# -*- coding: utf-8 -*-
"""
data_loader.py — loading, cleaning, leakage removal, splitting, fold-level encoding and scaling.

STAGE 1 (rule-based, no statistics learned from the data)
  leakage columns dropped (config.LEAKAGE_COLUMNS) -> labels separated -> placeholder spellings unified
  ("0.0", "", NaN -> "0") -> numeric conversion -> exact duplicates removed -> conflicting-label records removed
  -> stratified 70/30 split by attack type.
FOLD LEVEL (fitted on training data only)
  FoldPreprocessor: one-hot / structural text encoding, constant-feature removal, single-feature leakage audit on an
  inner hold-out of the TRAINING part; FeatureScaler: signed-log1p on wide-range columns + Min-Max to [-1, 1];
  holdout_split: the validation hold-out of the training side used for early stopping.
"""
import copy
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .reporting import fmt_int

# Cross-dataset schema alignment (Stage 4): common name -> (Edge-IIoTset columns, "any non-zero"), CICIoT2023 column.
# Only 0/1 presence indicators of these columns are kept for transfer, never their raw values.
TRANSFER_ALIGNMENT = {
    "syn": (["tcp.connection.syn"], "syn_flag_number"), "fin": (["tcp.connection.fin"], "fin_flag_number"),
    "rst": (["tcp.connection.rst"], "rst_flag_number"), "ack": (["tcp.flags.ack"], "ack_flag_number"),
    "tcp": (["tcp.flags"], "TCP"), "udp": (["udp.stream", "udp.time_delta"], "UDP"),
    "icmp": (["icmp.checksum", "icmp.seq_le"], "ICMP"), "arp": (["arp.opcode", "arp.hw.size"], "ARP"),
    "dns": (["dns.qry.name.len", "dns.qry.qu", "dns.qry.type"], "DNS"),
    "http": (["http.request.method", "http.response", "http.content_length"], "HTTP")}


def clean_text(col, tokens):
    """Strips a text column and maps every placeholder spelling (and real missing values) to "0".
    Works with both pandas 2 (astype(str) -> "nan") and pandas 3 (astype(str) keeps NaN)."""
    raw = col.astype(str).str.strip()
    is_ph = raw.isna() | (raw.isin(tokens) & (raw != "0"))
    return raw.fillna("0").where(~raw.isin(tokens), "0"), int(is_ph.sum())


def stratification_key(atype, y, min_count):
    """Stratification labels for splits by attack type. Attack types with fewer than ``min_count`` records (after
    de-duplication some families collapse to a handful of unique packets) cannot be stratified; they are pooled per
    binary label, and a pool that is still too small joins the largest group with the same binary label.
    Returns (key, pooled attack types)."""
    atype, y = np.asarray(atype).astype(str), np.asarray(y)
    t, cnt = np.unique(atype, return_counts=True)
    rare = sorted(t[cnt < min_count].tolist())
    key = atype.astype(object).copy()
    if rare:
        m = np.isin(atype, rare)
        key[m] = np.where(y[m] == 1, "__rare_attack__", "__rare_normal__")
        for pool, lab in (("__rare_attack__", 1), ("__rare_normal__", 0)):
            sel = key == pool
            if 0 < sel.sum() < 2:
                vals, n = np.unique(key[(y == lab) & ~sel].astype(str), return_counts=True)
                if len(vals):
                    key[sel] = vals[np.argmax(n)]
    return key.astype(str), rare


def holdout_split(strat, frac, seed):
    """Positions (fit, val) of a stratified hold-out of ``frac`` of the rows (``strat`` = stratification labels).
    Falls back to a random hold-out when a label is too rare to stratify, and returns an empty validation part when the
    rows are too few to hold out any."""
    strat = np.asarray(strat)
    idx = np.arange(len(strat))
    n_val = int(round(frac * len(strat)))
    if n_val < 1 or n_val >= len(strat):
        return idx, idx[:0]
    _, cnt = np.unique(strat, return_counts=True)
    ok = cnt.min() >= 2 and len(cnt) <= min(n_val, len(strat) - n_val)
    fit, val = train_test_split(idx, test_size=n_val, stratify=strat if ok else None, random_state=seed)
    return np.sort(fit), np.sort(val)


def partition_table(R, y, parts):
    R.table(["Partition", "Rows", "Share", "Normal", "Attack", "Attack %"],
            [(n, fmt_int(len(i)), f"{len(i) / len(y):.1%}", fmt_int((y[i] == 0).sum()), fmt_int((y[i] == 1).sum()),
              f"{y[i].mean():.2%}") for n, i in parts], widths=[28, 12, 9, 12, 12, 10])


# =========================================================================================================
# STAGE 1 — dataset
# =========================================================================================================
class EdgeIIoTDataset:
    """Loads Edge-IIoTset and applies the global, rule-based cleaning of Stage 1."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.presence = None

    def load(self, R):
        cfg = self.cfg
        header = pd.read_csv(cfg.source_path, nrows=0).columns.tolist()
        for col in (cfg.target_col, cfg.attack_type_col):
            if col not in header:
                raise ValueError(f"label column '{col}' not found in {cfg.source_path}")
        removed = {**cfg.leakage_cols, **cfg.excluded_cols}
        need = sorted({c for cols, _ in TRANSFER_ALIGNMENT.values() for c in cols if c in header})
        # dropped columns are never read (saves memory), except those needed for the transfer presence flags
        usecols = [c for c in header if c not in removed or c in need]
        df = pd.read_csv(cfg.source_path, usecols=usecols, low_memory=False)
        rep = {"rows_raw": len(df), "cols_raw": len(header)}
        if cfg.sample_frac < 1.0:
            df = df.groupby(cfg.attack_type_col, group_keys=False).sample(frac=cfg.sample_frac, random_state=cfg.seed)
            rep["rows_after_sampling"] = len(df)
        rep["leakage_columns_removed"] = {c: r for c, r in cfg.leakage_cols.items() if c in header}
        rep["excluded_columns"] = {c: r for c, r in cfg.excluded_cols.items() if c in header}
        rep["leakage_columns_absent"] = [c for c in cfg.leakage_cols if c not in header]
        tokens = set(cfg.placeholder_tokens)
        # presence flags of the transfer-alignment columns, computed BEFORE the identifiers are dropped
        self.presence = pd.DataFrame({c: ((pd.to_numeric(df[c], errors="coerce").fillna(0).values != 0)
                                          if pd.api.types.is_numeric_dtype(df[c])
                                          else ~clean_text(df[c], tokens)[0].isin(["0"]).values) for c in need},
                                     index=df.index)
        df = df.drop(columns=[c for c in removed if c in df.columns])
        y = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0).astype(np.int8)
        atype = df[cfg.attack_type_col].astype(str).str.strip()
        X = df.drop(columns=[cfg.target_col, cfg.attack_type_col])
        del df
        num_cols, cat_cols, fixed, coerced, nan_filled = [], [], {}, {}, 0
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                nan_filled += int(X[c].isna().sum())
                X[c] = X[c].astype(np.float64).fillna(0.0)
                num_cols.append(c)
                continue
            s, n_ph = clean_text(X[c], tokens)
            if n_ph:
                fixed[c] = n_ph
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().mean() >= cfg.numeric_min_frac:
                if num.isna().any():
                    coerced[c] = int(num.isna().sum())
                X[c] = num.fillna(0.0).astype(np.float64)
                num_cols.append(c)
            else:
                X[c] = s
                cat_cols.append(c)
        n0 = len(X)
        feats = list(X.columns)
        X["__y"], X["__t"] = y.values, atype.values
        X = X.drop_duplicates(subset=feats + ["__y"], keep="first")
        n1 = len(X)
        # conflicting-label records: the same feature vector labelled both Normal and Attack (all copies removed)
        conf = X.duplicated(subset=feats, keep=False)
        conf_types = X.loc[conf, "__t"].value_counts()
        X = X[~conf]
        self.presence = self.presence.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
        rep.update(placeholders_unified=fixed, non_numeric_set_to_0=coerced, missing_filled=nan_filled,
                   duplicates_removed=n0 - n1, conflicting_removed=n1 - len(X),
                   conflicting_by_type={str(k): int(v) for k, v in conf_types.items()})
        y = X.pop("__y").astype(np.int8).values
        atype = X.pop("__t").values.astype(str)
        lim = float(np.finfo(np.float32).max)
        n_big = int((X[num_cols].abs() > lim).to_numpy().sum()) if num_cols else 0
        if n_big:                                   # would become inf in float32 and break Min-Max scaling
            rep["values_clipped_to_float32_range"] = n_big
            X[num_cols] = X[num_cols].clip(-lim, lim)
        X[num_cols] = X[num_cols].astype(np.float32)
        rep.update(rows_clean=len(X), numeric=len(num_cols), text=cat_cols,
                   class_counts={int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                   attack_type_counts={str(k): int(v) for k, v in zip(*np.unique(atype, return_counts=True))})
        self._report(R, rep, len(y), num_cols, cat_cols)
        return X, y, atype, num_cols, cat_cols, rep

    def _report(self, R, rep, n, num_cols, cat_cols):
        cfg = self.cfg
        base = rep.get("rows_after_sampling", rep["rows_raw"])
        R.stage("STAGE 1 | GLOBAL PREPROCESSING (rule-based, no statistics learned from the data)")
        R.kv("Raw dataset", f"{fmt_int(rep['rows_raw'])} rows x {rep['cols_raw']} columns")
        R.say(f"    Step 1 | Leakage columns removed ({len(rep['leakage_columns_removed'])})")
        for c, why in rep["leakage_columns_removed"].items():
            R.say(f"             - {c:<24} {why}")
        if rep["excluded_columns"]:
            R.say(f"    Step 1b| Other excluded columns ({len(rep['excluded_columns'])}, not leakage)")
            for c, why in rep["excluded_columns"].items():
                R.say(f"             - {c:<24} {why}")
        if rep["leakage_columns_absent"]:
            R.say(f"    WARNING: leakage columns not present in this file: {', '.join(rep['leakage_columns_absent'])}")
        R.say(f"    Step 2 | Labels separated: {cfg.target_col} (target), {cfg.attack_type_col} (type)")
        R.say(f"    Step 3 | Placeholder spellings / missing text unified (\"0.0\", \"\", NaN -> \"0\"): "
              f"{', '.join(f'{k} ({fmt_int(v)})' for k, v in rep['placeholders_unified'].items()) or 'none'}")
        if rep["non_numeric_set_to_0"]:
            R.say("    Step 3b| Non-numeric values in numeric columns set to 0: " +
                  ", ".join(f"{k} ({fmt_int(v)})" for k, v in rep["non_numeric_set_to_0"].items()))
        R.say(f"    Step 4 | Missing numeric values imputed with 0: {fmt_int(rep['missing_filled'])}")
        R.say(f"    Step 5 | Exact duplicates removed: {fmt_int(rep['duplicates_removed'])} "
              f"({rep['duplicates_removed'] / base:.2%})")
        R.say(f"    Step 6 | Conflicting-label records removed (identical features, both labels): "
              f"{fmt_int(rep['conflicting_removed'])} ({rep['conflicting_removed'] / base:.2%})")
        R.kv("Clean dataset", f"{fmt_int(n)} rows | {len(num_cols) + len(cat_cols)} features "
                              f"({len(num_cols)} numeric, {len(cat_cols)} text)")
        cc = rep["class_counts"]
        R.kv("Class distribution", f"Normal {fmt_int(cc.get(0, 0))} ({cc.get(0, 0) / n:.2%}) | "
                                   f"Attack {fmt_int(cc.get(1, 0))} ({cc.get(1, 0) / n:.2%})")

    def split(self, X, y, atype, R):
        """Stratified 70/30 split by Attack_type (exact duplicates and conflicting records were removed in Stage 1)."""
        cfg = self.cfg
        # types need >= 2·n_folds records so that both partitions and every CV fold can receive them
        key, rare = stratification_key(atype, y, 2 * cfg.n_folds)
        dev, test = train_test_split(np.arange(len(y)), test_size=cfg.test_frac, stratify=key,
                                     random_state=cfg.seed)
        dev, test = np.sort(dev), np.sort(test)
        R.stage(f"STAGE 1 | STRATIFIED SPLIT BY ATTACK TYPE ({1 - cfg.test_frac:.0%} development / "
                f"{cfg.test_frac:.0%} independent test)")
        if rare:
            counts = dict(zip(*np.unique(atype, return_counts=True)))
            R.say(f"    WARNING: attack types with < {2 * cfg.n_folds} unique records after de-duplication are "
                  f"stratified as a pooled group: " + ", ".join(f"{t} ({counts[t]})" for t in rare))
        partition_table(R, y, [("Development", dev), ("Independent test (locked)", test)])
        return dev, test


# =========================================================================================================
# FOLD LEVEL — encoding and in-fold leakage audit (fitted on training data only)
# =========================================================================================================
def _entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in Counter(s).values())


_STAT = {"len": len,
         "digit_ratio": lambda s: sum(ch.isdigit() for ch in s) / max(len(s), 1),
         "alpha_ratio": lambda s: sum(ch.isalpha() for ch in s) / max(len(s), 1),
         "special_ratio": lambda s: sum((not ch.isalnum()) and (not ch.isspace()) for ch in s) / max(len(s), 1),
         "entropy": _entropy}


class FoldPreprocessor:
    """Text encoding, constant-feature removal and a single-feature leakage audit (depth-1 tree, balanced accuracy >=
    leakage_threshold) evaluated on an inner hold-out of the TRAINING data — the fold validation part is never used to
    decide which features exist. Text encoding (cfg.text_encoding):
      "ordinal" — label encoding: every text column stays ONE feature with the same name; categories are coded in
                  alphabetical order (1, 2, ...), categories seen fewer than onehot_min_freq times in the training data
                  and unseen categories share the code 0 ("other");
      "onehot"  — one-hot encoding for low-cardinality columns and structural features (length, character ratios,
                  entropy) for high-cardinality ones."""

    def __init__(self, cfg, num_cols, cat_cols):
        self.cfg, self.numeric_cols, self.cat_cols = cfg, list(num_cols), list(cat_cols)
        self.onehot, self.structural, self.ordinal, self.feature_names = {}, [], {}, []
        self.dropped_constant, self.leakage_report = [], {}

    def _encode(self, X):
        parts = [X[self.numeric_cols].reset_index(drop=True).astype(np.float64)]
        for c, cats in self.ordinal.items():
            codes = {k: float(i + 1) for i, k in enumerate(cats)}
            s = X[c].astype(str).reset_index(drop=True)
            parts.append(pd.DataFrame({c: s.map(codes).fillna(0.0).astype(np.float64)}))
        for c, cats in self.onehot.items():
            s = X[c].astype(str).reset_index(drop=True)
            s = s.where(s.isin(cats), "__other__")
            parts.append(pd.DataFrame({f"{c}={k}": (s == k).astype(np.float64) for k in cats + ["__other__"]}))
        for c in self.structural:
            s = X[c].astype(str).reset_index(drop=True)
            u = pd.Series(s.unique())
            t = pd.DataFrame({f"{c}__{k}": u.map(f) for k, f in _STAT.items()})
            t.index = u.values
            parts.append(t.reindex(s.values).reset_index(drop=True).astype(np.float64))
        return pd.concat(parts, axis=1)

    def fit(self, Xtr, ytr, forced_drop=None):
        """forced_drop=None -> run the in-fold leakage audit; otherwise drop the given features (final model: union
        of the features flagged in any CV fold)."""
        cfg = self.cfg
        for c in self.cat_cols:
            vc = Xtr[c].astype(str).value_counts()
            if getattr(cfg, "text_encoding", "onehot") == "ordinal":
                self.ordinal[c] = sorted(k for k, v in vc.items() if v >= cfg.onehot_min_freq)
            elif len(vc) <= cfg.onehot_max_categories:
                self.onehot[c] = [k for k, v in vc.items() if v >= cfg.onehot_min_freq]
            else:
                self.structural.append(c)
        E = self._encode(Xtr)
        self.dropped_constant = ([c for c in E.columns if E[c].nunique() <= 1]
                                 if getattr(cfg, "drop_constant_features", True) else [])
        E = E.drop(columns=self.dropped_constant)
        if forced_drop is not None:
            self.leakage_report = {c: "flagged in CV" for c in forced_drop if c in E.columns}
        elif len(np.unique(ytr)) == 2:
            fit_i, aud_i = train_test_split(np.arange(len(E)), test_size=cfg.leakage_audit_frac, stratify=ytr,
                                            random_state=cfg.seed)
            if len(fit_i) > 200_000:
                fit_i = np.random.RandomState(cfg.seed).choice(fit_i, 200_000, replace=False)
            V = E.values
            for j, c in enumerate(E.columns):
                st = DecisionTreeClassifier(max_depth=1, random_state=0).fit(V[fit_i, j:j + 1], ytr[fit_i])
                b = balanced_accuracy_score(ytr[aud_i], st.predict(V[aud_i, j:j + 1]))
                if b >= cfg.leakage_threshold:
                    self.leakage_report[c] = round(float(b), 5)
        E = E.drop(columns=list(self.leakage_report))
        self.feature_names = list(E.columns)
        if not self.feature_names:
            raise RuntimeError("no features left after constant removal and the leakage audit")
        return self

    def transform(self, X):
        """Raw frame -> float32 matrix with exactly ``feature_names`` columns (missing raw fields -> 0)."""
        tok = set(self.cfg.placeholder_tokens)
        cols = {}
        for c in self.numeric_cols:
            cols[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).values if c in X else np.zeros(len(X))
        for c in self.cat_cols:
            cols[c] = clean_text(X[c], tok)[0].values if c in X else np.full(len(X), "0", dtype=object)
        Z = pd.DataFrame(cols, index=range(len(X)))
        return self._encode(Z).reindex(columns=self.feature_names, fill_value=0.0).values.astype(np.float32)

    def subset(self, selected):
        """Lean copy for deployment: computes only the selected features from only the needed raw fields."""
        lean = copy.copy(self)
        sel = list(selected)
        lean.numeric_cols = [c for c in self.numeric_cols if c in sel]
        lean.onehot = {c: v for c, v in self.onehot.items() if any(n.startswith(c + "=") for n in sel)}
        lean.structural = [c for c in self.structural if any(n.startswith(c + "__") for n in sel)]
        lean.ordinal = {c: v for c, v in self.ordinal.items() if c in sel}
        lean.cat_cols = list(lean.onehot) + lean.structural + list(lean.ordinal)
        lean.feature_names = sel
        return lean

    def raw_fields(self):
        return list(self.numeric_cols) + list(self.onehot) + list(self.structural) + list(self.ordinal)

    def state(self):          # plain-data state (portable bundle, no pickled classes)
        return {"numeric_cols": self.numeric_cols, "cat_cols": self.cat_cols, "onehot": self.onehot,
                "structural": self.structural, "ordinal": self.ordinal, "feature_names": self.feature_names}

    @classmethod
    def from_state(cls, cfg, st):
        p = cls(cfg, st["numeric_cols"], st["cat_cols"])
        p.onehot, p.structural, p.feature_names = st["onehot"], st["structural"], st["feature_names"]
        p.ordinal = st.get("ordinal", {})          # bundles saved before the ordinal option have no such key
        return p


# =========================================================================================================
# FOLD LEVEL — scaling
# =========================================================================================================
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


class FeatureScaler:
    """signed-log1p on wide-range columns (max(|min|,|max|) > thr) followed by Min-Max to [-1, 1] with clipping.
    Used centrally (min/max of the training data) and federally (aggregated client min/max only), so both paths
    transform identical columns. Pure NumPy (vectorised, float32 output)."""

    def __init__(self, thr=100.0):
        self.thr, self.log_idx = thr, np.array([], dtype=int)

    def _set(self, lo, hi):
        self.data_min, self.data_max = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        rng = self.data_max - self.data_min
        rng[rng == 0] = 1.0                                     # constant column -> maps to -1
        self.mul = 2.0 / rng
        self.add = -1.0 - self.data_min * self.mul
        return self

    def fit_minmax(self, mn, mx):
        mn, mx = np.asarray(mn, np.float64), np.asarray(mx, np.float64)
        self.log_idx = np.where(np.maximum(np.abs(mn), np.abs(mx)) > self.thr)[0].astype(int)
        lo, hi = mn.copy(), mx.copy()
        lo[self.log_idx], hi[self.log_idx] = signed_log1p(lo[self.log_idx]), signed_log1p(hi[self.log_idx])
        return self._set(lo, hi)

    def fit(self, X):                      # centralized
        return self.fit_minmax(X.min(0), X.max(0))

    @classmethod
    def federated(cls, client_arrays, thr):   # clients send only their min and max
        return cls(thr).fit_minmax(np.min([a.min(0) for a in client_arrays], 0),
                                   np.max([a.max(0) for a in client_arrays], 0))

    def transform(self, X):
        x = np.array(X, dtype=np.float64, copy=True)
        if len(self.log_idx):
            x[:, self.log_idx] = signed_log1p(x[:, self.log_idx])
        x *= self.mul
        x += self.add
        return np.clip(x, -1.0, 1.0, out=x).astype(np.float32)

    def state(self):
        return {"thr": self.thr, "log_idx": self.log_idx.tolist(), "data_min": self.data_min.tolist(),
                "data_max": self.data_max.tolist()}

    @classmethod
    def from_state(cls, st):
        f = cls(st["thr"])
        f.log_idx = np.array(st["log_idx"], dtype=int)
        return f._set(st["data_min"], st["data_max"])
