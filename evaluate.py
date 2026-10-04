# -*- coding: utf-8 -*-
"""
evaluate.py — performance metrics, statistical tests and the edge (deployment) profile.

Every model uses the standard decision rule: attack if the sigmoid score >= 0.5 (config.decision_threshold); with
threshold_mode "far" a recall-oriented threshold (select_threshold) is reported as well.
Metrics: accuracy, precision, recall (= detection rate), F1, FPR (= FAR), specificity, balanced accuracy, MCC,
Cohen's kappa, G-mean, ROC-AUC, PR-AUC (threshold-independent) and the confusion matrix (tp, tn, fp, fn).
"""
import copy
import io
import time

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from .models import count_params


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def binary_metrics(y, s, thr=0.5):
    y, s = np.asarray(y), np.asarray(s)
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    both = len(np.unique(y)) == 2
    return {"threshold": float(thr), "accuracy": accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0), "micro_f1": f1_score(y, p, average="micro", zero_division=0),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "roc_auc": roc_auc_score(y, s) if both else np.nan,
            "pr_auc": average_precision_score(y, s) if both else np.nan,
            "far": fp / max(fp + tn, 1), "fpr": fp / max(fp + tn, 1), "specificity": tnr, "detection_rate": tpr,
            "mcc": matthews_corrcoef(y, p), "kappa": cohen_kappa_score(y, p), "g_mean": float(np.sqrt(tpr * tnr)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def select_threshold(y, s, target_far, cap=0.5):
    """Recall-oriented decision threshold: the lowest threshold whose false-alarm rate on (y, s) is <= target_far, i.e.
    the highest detection rate the false-alarm budget allows, capped at `cap` (it may only lower the standard 0.5).
    (y, s) must not be the data the threshold is evaluated on (out-of-fold development scores). Only the per-class
    score distribution is needed, so in a federated deployment it follows from score histograms summed over clients."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    ok = ~np.isnan(s)
    neg = np.sort(s[ok & (y == 0)])[::-1]                      # Normal scores, highest first
    if not len(neg):
        return float(cap)
    k = int(np.floor(target_far * len(neg) + 1e-9))            # false alarms the budget allows
    if k >= len(neg):
        return float(min(cap, neg[-1]))
    return float(min(cap, np.nextafter(neg[k], np.inf)))       # just above the (k+1)-th highest Normal score


def per_attack_dr(atype, y, s, thr):
    """Detection rate per attack family (attack records only)."""
    p = s >= thr
    return {str(t): float(p[(atype == t) & (y == 1)].mean()) for t in np.unique(atype[y == 1])}


# ---------------------------------------------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------------------------------------------
def paired_tests(a, b, n_train, n_test):
    """Paired t-test and the corrected resampled t-test (Nadeau & Bengio 2003) over CV folds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    v = d.var(ddof=1)
    if v == 0 or not np.isfinite(v):
        return {"p": np.nan, "p_corr": np.nan}
    _, p = stats.ttest_rel(a, b)
    tc = d.mean() / np.sqrt((1 / n + n_test / n_train) * v)
    return {"p": float(p), "p_corr": float(2 * stats.t.sf(abs(tc), df=n - 1))}


def holm(p):
    """Holm-Bonferroni step-down adjustment (NaN entries are ignored)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    run = 0.0
    for r, j in enumerate(idx[np.argsort(p[idx])]):
        run = max(run, min(1.0, (len(idx) - r) * p[j]))
        out[j] = run
    return out


def mcnemar(y, pa, pb):
    """McNemar's test on a single test set (Dietterich 1998): exact binomial for < 25 discordant pairs."""
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p": 1.0}
    if b + c < 25:
        return {"b": b, "c": c, "p": float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))}
    return {"b": b, "c": c, "p": float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))}


# ---------------------------------------------------------------------------------------------------------------
# Edge profile — parameters, executed FLOPs, memory, batch-1 single-thread latency
# ---------------------------------------------------------------------------------------------------------------
def edge_profile(model, d, X, n=500):
    from torch.utils._python_dispatch import TorchDispatchMode
    m = copy.deepcopy(model).cpu().eval()
    PER = {"add": 1, "sub": 1, "rsub": 1, "mul": 1, "div": 1, "neg": 1, "exp": 1, "tanh": 1, "sigmoid": 1, "silu": 4,
           "layer_norm": 5, "native_layer_norm": 5, "batch_norm": 4, "native_batch_norm": 4}

    class Flops(TorchDispatchMode):          # counts executed ops (Yu, Yu & Wang 2024 convention)
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            nm = func.overloadpacket.__name__.rstrip("_")
            if nm == "linear":
                x, w = args[0], args[1]
                self.n += 2 * x.numel() * w.shape[0] + (x.numel() // x.shape[-1] * w.shape[0] if len(args) > 2 else 0)
            elif nm in ("mm", "bmm"):
                self.n += 2 * args[0].numel() * args[1].shape[-1]
            elif nm == "addmm":
                self.n += 2 * args[1].numel() * args[2].shape[-1]
            elif nm in PER:
                ref = out[0] if isinstance(out, (tuple, list)) else out
                self.n += PER[nm] * ref.numel() if isinstance(ref, torch.Tensor) else 0
            return out
    acts = []
    hooks = [mod.register_forward_hook(lambda mm, i, o: acts.append(o[0].numel() if isinstance(o, tuple) else o.numel()))
             for mod in m.modules() if len(list(mod.children())) == 0]
    with torch.inference_mode(), Flops() as fc:
        m(torch.zeros(1, d))
    for hk in hooks:
        hk.remove()
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        idx = np.random.RandomState(0).randint(0, len(X), n + 50)
        ts = []
        with torch.inference_mode():
            for k, i in enumerate(idx):
                t0 = time.perf_counter()
                m(torch.from_numpy(X[i:i + 1]))
                if k >= 50:                                         # 50 warm-up calls
                    ts.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(nt)
    return {"params": count_params(m), "size_kb": buf.tell() / 1024, "flops": int(fc.n),
            "memory_kb": count_params(m) * 4 / 1024 + (max(acts) if acts else 0) * 4 / 1024,
            "latency_median_ms": float(np.median(ts)), "latency_p95_ms": float(np.percentile(ts, 95))}
