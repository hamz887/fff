# -*- coding: utf-8 -*-
"""
detector.py — the deployable detection system: saved bundle and the hierarchical detector (binary decision -> attack
type for flagged records). The bundle holds only plain data and tensors and is loaded with
torch.load(weights_only=True), so loading it cannot execute code (unlike a pickle).
"""
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from .config import DEVICE, MODEL_NAMES, Config, arch_of, long_name
from .data_loader import FeatureScaler, FoldPreprocessor
from .models import build_model
from .reporting import write_json
from .trainer import predict_scores


class IDSBundle:
    @staticmethod
    def save(cfg, key, setting, fe, payload, thr, test_row, mc_model, classes, mc_summ, edge, subdir="",
             role="primary", note=""):
        """Saves one detection system (ids_bundle.pt) with the feature list and scaler the model was trained with."""
        md = os.path.join(cfg.dirs["models"], subdir)
        os.makedirs(md, exist_ok=True)
        sel = payload["features"]
        prep = fe["prep"].subset(sel)
        fed = key.startswith("fl_")
        mc_state = {k: v.cpu() for k, v in mc_model.state_dict().items()} if mc_model is not None else None
        bundle = {"cfg": asdict(cfg), "model_key": key, "arch": arch_of(key), "setting": setting, "features": sel,
                  "prep": prep.state(), "scaler": payload["scaler"], "binary_state": payload["state"],
                  "threshold": thr, "classes": classes, "type_state": mc_state}
        torch.save(bundle, os.path.join(md, "ids_bundle.pt"))
        torch.save(payload["state"], os.path.join(md, "binary_detector.pt"))
        if mc_state is not None:
            torch.save(mc_state, os.path.join(md, "attack_type_classifier.pt"))
        card = {"role": role, "selection_note": note, "model_key": key, "architecture": long_name(key),
                "network": MODEL_NAMES[arch_of(key)],
                "training_paradigm": f"federated learning — {cfg.fl_protocol()}" if fed else
                (f"centralized — AdamW (lr {cfg.lr:g}); number of epochs chosen in cross-validation by early "
                 f"stopping on the validation detection loss (patience {cfg.es_patience}), final model retrained on "
                 f"the development set for it"),
                "feature_selection": ("federated feature voting (client-side Boruta+XGBoost, server keeps features "
                                      f"chosen by >= {cfg.fs_vote_frac:.0%} of clients)") if (fed and setting == "fs")
                else ("Boruta+XGBoost (centralized)" if setting == "fs" else "all encoded features"),
                "leakage_columns_removed": cfg.leakage_cols,
                "n_features": len(sel), "features": sel, "raw_fields_needed": prep.raw_fields(), "threshold": thr,
                "threshold_policy": f"standard decision rule: attack if score >= {thr:g}",
                "test_metrics": {k: test_row[k] for k in ("accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "far")},
                "attack_types": classes, "attack_type_metrics": mc_summ, "edge_profile": edge}
        write_json(card, os.path.join(md, "model_card.json"))


class SchemaError(ValueError):
    """Raised when input records cannot be scored reliably (missing required fields, malformed values)."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class Detector:
    """Raw record(s) -> schema validation -> saved preprocessing -> selected features -> saved scaler -> binary
    detector (threshold) -> for attacks: softmax attack-type classifier -> decision, attack probability, attack type
    and confidence. Missing required fields are rejected (SchemaError) unless allow_missing=True."""

    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"detection bundle not found: {path} (run the pipeline first)")
        b = torch.load(path, map_location="cpu", weights_only=True)
        self.b = b
        self.cfg = Config.from_dict(b["cfg"])
        self.prep = FoldPreprocessor.from_state(self.cfg, b["prep"])
        self.scaler = FeatureScaler.from_state(b["scaler"])
        d = len(b["features"])
        self.bin = build_model(b["arch"], d, self.cfg)
        self.bin.load_state_dict(b["binary_state"])
        self.bin.to(DEVICE).eval()
        self.types = None
        if b["type_state"] is not None:
            self.types = build_model(b["arch"], d, self.cfg, n_out=len(b["classes"]))
            self.types.load_state_dict(b["type_state"])
            self.types.to(DEVICE).eval()
        card_path = os.path.join(os.path.dirname(path), "model_card.json")
        self.card = {}
        if os.path.isfile(card_path):
            with open(card_path, encoding="utf-8") as fh:
                self.card = json.load(fh)
        self.required = self.prep.raw_fields()
        self.name = self.card.get("architecture", MODEL_NAMES.get(b.get("model_key", b["arch"]), b["arch"]))

    def validate(self, df):
        """Schema check. Returns (missing fields, non-numeric counts); raises SchemaError for unusable input."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise SchemaError("no records were provided")
        if len(df) > self.cfg.max_rows:
            raise SchemaError(f"too many records ({len(df):,}); the limit is {self.cfg.max_rows:,} per request")
        if df.columns.duplicated().any():
            raise SchemaError("duplicate column names", {"duplicates": df.columns[df.columns.duplicated()].tolist()})
        missing = [c for c in self.required if c not in df.columns]
        tok, bad = set(self.cfg.placeholder_tokens), {}
        for c in self.prep.numeric_cols:
            if c in df.columns:
                raw = df[c].astype(str).str.strip()
                n_bad = int((pd.to_numeric(df[c], errors="coerce").isna() & ~raw.isin(tok) & raw.notna()).sum())
                if n_bad:
                    bad[c] = n_bad
        return missing, bad

    def detect(self, df, allow_missing=False):
        df = df.drop(columns=[c for c in (self.cfg.target_col, self.cfg.attack_type_col) if c in df.columns])
        missing, bad = self.validate(df)
        warns = []
        if missing and not allow_missing:
            raise SchemaError(f"{len(missing)} required field(s) missing; the detector will not impute them "
                              f"silently", {"missing_fields": missing, "required_fields": self.required,
                                            "hint": "send the missing fields, or set allow_missing=true to impute "
                                                    "them with 0 (predictions become unreliable)"})
        if missing:
            warns.append(f"{len(missing)} required field(s) missing and imputed with 0 — predictions are unreliable: "
                         f"{', '.join(missing)}")
        if bad:
            frac = sum(bad.values()) / (len(df) * max(len(bad), 1))
            if frac > 0.5:
                raise SchemaError("most values in numeric fields are not numbers", {"non_numeric_values": bad})
            warns.append("non-numeric values in numeric fields were treated as 0: " +
                         ", ".join(f"{c} ({n})" for c, n in bad.items()))
        x = self.scaler.transform(self.prep.transform(df.reset_index(drop=True)))
        p = predict_scores(self.bin, x)
        att = p >= self.b["threshold"]
        out = pd.DataFrame({"decision": np.where(att, "Attack", "Normal"), "attack_probability": p.round(6),
                            "attack_type": "—", "type_confidence": np.nan})
        if self.types is not None and att.any():
            P = predict_scores(self.types, x[att], multiclass=True)
            out.loc[att, "attack_type"] = np.array(self.b["classes"])[P.argmax(1)]
            out.loc[att, "type_confidence"] = P.max(1).round(4)
        return out, warns
