# -*- coding: utf-8 -*-
"""
transfer.py — STAGE 4: cross-dataset generalization (source Edge-IIoTset -> target CICIoT2023 CSV).

Only semantically equivalent features are aligned: the TCP flags and the protocol indicators (0/1 presence flags in
the source, window averages in [0, 1] in the target). Edge-IIoTset rows are single packets whereas CICIoT2023 rows
aggregate a fixed-size packet window — a stated limitation. Binary task only (the attack taxonomies differ).
The source model is trained with the deployed architecture and paradigm on the development set (early stopping on a
training-side hold-out, or on the clients' local validation parts when federated). It is then fine-tuned on the target
in two phases: the target is split 60/20/20 (stratified) into training, validation and test parts; phase 1 trains the
classification head with the encoder frozen, phase 2 fine-tunes every layer with a smaller learning rate; both phases
use early stopping on the target validation loss. The fine-tuned model is evaluated once on the target test part.
"""
import copy
import os
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MODEL_NAMES, arch_of, long_name
from .data_loader import TRANSFER_ALIGNMENT, FeatureScaler, holdout_split, stratification_key
from .evaluate import binary_metrics
from .feature_selection import stratified_subsample
from .models import build_model, count_params
from .reporting import fmt_int, write_json
from .trainer import FederatedTrainer, Trainer, predict_scores


class CrossDatasetTransfer:
    def __init__(self, cfg, R):
        self.cfg, self.R = cfg, R

    @staticmethod
    def _source(presence):
        """Protocol/flag indicators from the Stage 1 presence flags (0/1 only, never identifier values)."""
        feats = {}
        for k, (cols, _) in TRANSFER_ALIGNMENT.items():
            cols = [c for c in cols if c in presence.columns]
            if cols:
                feats[k] = presence[cols].any(axis=1).values.astype(np.float32)
        return pd.DataFrame(feats)

    def run(self, presence, y, atype, dev, test, key):
        cfg, R = self.cfg, self.R
        R.stage("STAGE 4 | CROSS-DATASET GENERALIZATION: Edge-IIoTset (source) -> CICIoT2023 CSV (target), "
                "two-phase fine-tuning")
        S = self._source(presence)
        hdr = pd.read_csv(cfg.target_path, nrows=0).columns.str.strip().tolist()
        lab = next((c for c in hdr if c.lower() == "label"), None)
        common = [k for k in S.columns if TRANSFER_ALIGNMENT[k][1] in hdr]
        if lab is None or not common:
            R.say("    WARNING: target file has no 'label' column or no aligned features; transfer skipped.")
            return None
        wanted = [TRANSFER_ALIGNMENT[k][1] for k in common] + [lab]
        T = pd.read_csv(cfg.target_path, usecols=lambda c: c.strip() in wanted)
        T.columns = T.columns.str.strip()
        yT = (~T[lab].astype(str).str.lower().str.contains("benign")).astype(np.int8).values
        if cfg.target_sample and len(T) > cfg.target_sample:      # 0 = use the full target file
            keep = stratified_subsample(yT, cfg.target_sample, cfg.seed)
            T, yT = T.iloc[keep].reset_index(drop=True), yT[keep]
        XT = T[[TRANSFER_ALIGNMENT[k][1] for k in common]].apply(pd.to_numeric, errors="coerce").fillna(0) \
            .values.astype(np.float32)
        XS = S[common].values
        R.kv("Aligned features", f"{len(common)}: " + ", ".join(
            f"{k} ({'/'.join(TRANSFER_ALIGNMENT[k][0])} <-> {TRANSFER_ALIGNMENT[k][1]})" for k in common))
        if len(np.unique(yT)) < 2:
            R.say("    WARNING: the target data contain a single class; transfer skipped.")
            return None

        # ---- source model: deployed architecture and training paradigm, aligned features only
        model = build_model(arch_of(key), len(common), cfg)
        if key.startswith("fl_"):
            ftr = FederatedTrainer(cfg, R)
            pp = [dev[p] for p in ftr.dirichlet_partition(y[dev], cfg.fl_clients, cfg.fl_alpha, cfg.seed,
                                                           cfg.fl_min_client_size)]
            sc = FeatureScaler.federated([XS[p] for p in pp], cfg.log_range_threshold)
            model, _, _ = ftr.fit(model, [(sc.transform(XS[p]), y[p]) for p in pp], cfg.seed,
                                  f"{MODEL_NAMES[key]} (source, aligned features)")
        else:
            strat, _ = stratification_key(atype[dev], y[dev], 10)
            fi, ei = holdout_split(strat, cfg.val_frac, cfg.seed)
            tr_i, va_i = dev[fi], dev[ei]
            sc = FeatureScaler(cfg.log_range_threshold).fit(XS[tr_i])           # training part only
            model, _, _ = Trainer(cfg, R).fit(model, sc.transform(XS[tr_i]), y[tr_i], sc.transform(XS[va_i]),
                                              y[va_i], cfg.seed, f"{MODEL_NAMES[key]} (source, aligned features)")

        # ---- target split: stratified 60 / 20 / 20 (training / validation / test)
        hold = cfg.transfer_val_frac + cfg.transfer_test_frac
        trn, rest = train_test_split(np.arange(len(yT)), test_size=hold, stratify=yT, random_state=cfg.seed)
        va, te = train_test_split(rest, test_size=cfg.transfer_test_frac / hold, stratify=yT[rest],
                                  random_state=cfg.seed)
        XTs = sc.transform(XT)
        R.table(["Target part", "Records", "Benign", "Attack", "Attack %"],
                [(n, fmt_int(len(i)), fmt_int(int((yT[i] == 0).sum())), fmt_int(int((yT[i] == 1).sum())),
                  f"{yT[i].mean():.2%}") for n, i in (("Fine-tuning training", trn), ("Fine-tuning validation", va),
                                                      ("Target test", te))], widths=[24, 12, 10, 12, 10])

        # ---- phase 1: classification head only (encoder frozen)
        fm = copy.deepcopy(model)
        for p in fm.parameters():
            p.requires_grad = False
        for p in fm.head().parameters():
            p.requires_grad = True
        n_head = count_params(fm)
        fm, _, i1 = Trainer(replace(cfg, es_patience=cfg.transfer_patience_head), R).fit(
            fm, XTs[trn], yT[trn], XTs[va], yT[va], cfg.seed, "fine-tuning phase 1 (head only)",
            lr=cfg.transfer_lr_head, max_epochs=cfg.transfer_epochs_head)
        # ---- phase 2: every layer
        for p in fm.parameters():
            p.requires_grad = True
        fm, _, i2 = Trainer(replace(cfg, es_patience=cfg.transfer_patience_full), R).fit(
            fm, XTs[trn], yT[trn], XTs[va], yT[va], cfg.seed, "fine-tuning phase 2 (all layers)",
            lr=cfg.transfer_lr_full, max_epochs=cfg.transfer_epochs_full)

        thr = cfg.decision_threshold
        r = binary_metrics(yT[te], predict_scores(fm, XTs[te]), thr)
        D = pd.DataFrame([{"scenario": "fine-tuned (two-phase), target test", **r}])
        D.to_csv(os.path.join(cfg.dirs["results"], "transfer_results.csv"), index=False)
        summ = {"source": "Edge-IIoTset", "target": os.path.basename(cfg.target_path), "architecture": long_name(key),
                "aligned_features": {k: [TRANSFER_ALIGNMENT[k][0], TRANSFER_ALIGNMENT[k][1]] for k in common},
                "target_records": int(len(yT)), "split": {"train": int(len(trn)), "validation": int(len(va)),
                                                          "test": int(len(te))},
                "phase1": {"trainable_params": int(n_head), "total_params": int(count_params(fm)),
                           "lr": cfg.transfer_lr_head, "epochs_run": i1["epochs_run"], "best_epoch": i1["best_epoch"]},
                "phase2": {"lr": cfg.transfer_lr_full, "epochs_run": i2["epochs_run"], "best_epoch": i2["best_epoch"]},
                **{f"test_{k}": float(r[k]) for k in ("accuracy", "precision", "recall", "f1", "balanced_accuracy",
                                                      "far", "mcc", "roc_auc")}}
        write_json(summ, os.path.join(cfg.dirs["results"], "transfer_summary.json"))
        R.kv("Phase 1 (head)", f"{n_head:,} of {count_params(fm):,} parameters trainable | lr {cfg.transfer_lr_head:g} | "
                               f"{i1['epochs_run']} epochs (best {i1['best_epoch']})")
        R.kv("Phase 2 (all layers)", f"lr {cfg.transfer_lr_full:g} | {i2['epochs_run']} epochs (best {i2['best_epoch']})")
        R.table(["Scenario", "Accuracy", "F1", "Recall", "FAR", "MCC", "ROC-AUC"],
                [(row.scenario, f"{row.accuracy:.4f}", f"{row.f1:.4f}", f"{row.recall:.4f}", f"{row.far:.4f}",
                  f"{row.mcc:.4f}", f"{row.roc_auc:.4f}") for row in D.itertuples()], widths=[40, 9, 9, 9, 9, 9, 9])
        return summ
