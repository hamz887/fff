# -*- coding: utf-8 -*-
"""threshold_check.py — synthetic check of the recall-oriented threshold (evaluate.select_threshold): unit checks, then
5-fold CV + final retrain of MLP / KAN / KAN-AE on Edge-IIoT-like synthetic data in which some attack packets are
camouflaged as Normal ACK traffic, reported at 0.5 and at several false-alarm budgets. The numbers have no scientific
meaning. Run from the project root:  python -m <package>.threshold_check [rows]   (env: EPOCHS, TARGET_FARS)"""
import json, os, sys, tempfile

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Config, CENTRAL_MODELS, MODEL_NAMES
from .data_loader import FeatureScaler, holdout_split, stratification_key
from .evaluate import binary_metrics, per_attack_dr, select_threshold
from .experiment import CrossValidator, FinalEvaluator, threshold_table
from .models import build_model
from .reporting import Reporter
from .trainer import Trainer, predict_scores

# ---------------- unit checks of select_threshold
rng = np.random.RandomState(0)
y = np.r_[np.zeros(10000), np.ones(3000)].astype(int)
s = np.r_[rng.beta(1, 8, 10000), rng.beta(5, 2, 3000)]
for far in (0.0, 0.0005, 0.001, 0.01):
    t = select_threshold(y, s, far)
    m = binary_metrics(y, s, t)
    assert t <= 0.5 and (m["far"] <= far + 1e-12 or t == 0.5), (far, t, m["far"])
    assert t == 0.5 or binary_metrics(y, s, np.nextafter(t, -1))["far"] > far   # lowest threshold allowed
    # lowering t by any amount below the next Normal score must break the budget (t is the lowest allowed)
    print(f"unit far<={far:<7g} thr={t:.4f} FAR={m['far']:.5f} recall={m['recall']:.4f}")
assert select_threshold(np.ones(5, int), np.ones(5), 0.001) == 0.5           # no Normal rows -> cap
assert select_threshold(y, np.where(y == 1, .99, .01), 0.001) == np.nextafter(.01, 1)
print("unit checks passed")

# ---------------- synthetic Edge-IIoT-like data with "camouflaged" attack packets
N, D = int(sys.argv[1]) if len(sys.argv) > 1 else 60000, 20
TYPES = {  # share of attack rows, fraction camouflaged as Normal ACK traffic (ordering follows the real overlap)
    "DDoS_UDP": (.233, 0), "DDoS_ICMP": (.130, 0), "DDoS_TCP": (.096, 0), "Port_Scanning": (.038, 0),
    "Vulnerability_scanner": (.095, .02), "SQL_injection": (.091, .12), "Password": (.089, .10),
    "DDoS_HTTP": (.075, .30), "Uploading": (.061, .27), "Backdoor": (.045, .04), "XSS": (.027, .07),
    "Ransomware": (.018, .08), "Fingerprinting": (.0014, .12), "MITM": (.0008, .01)}
rng = np.random.RandomState(42)
y = (rng.rand(N) < .28).astype(int)
names = list(TYPES)
p = np.array([v[0] for v in TYPES.values()]); p /= p.sum()
atype = np.where(y == 1, rng.choice(names, N, p=p), "Normal")
centers = {t: rng.normal(0, 1.6, D) for t in names}
ack = rng.normal(0, 1, D)                                   # the Normal "pure ACK" profile
X = np.empty((N, D))
nrm = y == 0
prof = rng.rand(N)
X[nrm] = rng.normal(0, 1, (nrm.sum(), D)) + np.where(prof[nrm, None] < .5, ack, -ack)
for t, (_, h) in TYPES.items():
    m = atype == t
    cam = m & (rng.rand(N) < h)
    X[m & ~cam] = centers[t] + rng.normal(0, 1, ((m & ~cam).sum(), D))
    shift = np.zeros(D); shift[:3] = .6                     # weak residual signal in 3 features
    X[cam] = ack + shift + rng.normal(0, 1, (cam.sum(), D))
X = X.astype(np.float32)

cfg = Config(target_far=float(os.environ.get("TARGET_FAR", 0.001)), max_epochs=int(os.environ.get("EPOCHS", 12)), es_patience=4, batch_size=512, ui="plain",
             out_dir=tempfile.mkdtemp())
for d in cfg.dirs.values():
    os.makedirs(d, exist_ok=True)
R = Reporter(os.path.join(cfg.dirs["logs"], "run.log"), ui="plain")
idx = np.arange(N)
dev, test = train_test_split(idx, test_size=.3, stratify=stratification_key(atype, y, 2)[0], random_state=cfg.seed)
yd, ad = y[dev], atype[dev]
strat, _ = stratification_key(ad, yd, cfg.n_folds)
splits = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(len(dev)), strat))
tr = Trainer(cfg, R)
rows, oof, epochs = [], {}, {}
for fold, (tp, vp) in enumerate(splits, 1):
    sc = FeatureScaler(cfg.log_range_threshold).fit(X[dev[tp]])
    a, v = sc.transform(X[dev[tp]]), sc.transform(X[dev[vp]])
    key, _ = stratification_key(ad[tp], yd[tp], 10)
    fi, ei = holdout_split(key, cfg.val_frac, cfg.seed)
    for m in CENTRAL_MODELS:
        model, h, info = tr.fit(build_model(m, D, cfg), a[fi], yd[tp][fi], a[ei], yd[tp][ei], cfg.seed, MODEL_NAMES[m])
        sv = predict_scores(model, v)
        oof.setdefault(("all", m), np.full(len(dev), np.nan, np.float32))[vp] = sv
        epochs.setdefault(m, []).append(info["best_epoch"])
        rows.append({"fold": fold, "setting": "all", "model": m, **binary_metrics(yd[vp], sv, cfg.decision_threshold)})
finals = {}
sc = FeatureScaler(cfg.log_range_threshold).fit(X[dev])
for m in CENTRAL_MODELS:
    model, _, _ = tr.fit_fixed(build_model(m, D, cfg), sc.transform(X[dev]), yd, cfg.seed,
                               max(1, int(round(np.mean(epochs[m])))), MODEL_NAMES[m])
    finals[m] = predict_scores(model, sc.transform(X[test]))
for far in [float(f) for f in os.environ.get("TARGET_FARS", "0.001,0.01,0.02").split(",")]:
    cfg.target_far = far
    R.say(f"\n======== TARGET FAR <= {far:g} ========")
    tuned = []
    CrossValidator(cfg, R, None).tuned_phase(R, rows, oof, yd, splits, "all", CENTRAL_MODELS, tuned)
    TR, TT = [], []
    for m in CENTRAL_MODELS:
        st = finals[m]
        t = select_threshold(yd, oof[("all", m)], cfg.target_far, cfg.decision_threshold)
        for out, thr in ((TR, cfg.decision_threshold), (TT, t)):
            out.append({"setting": "all", "model": m, "per_attack_dr": json.dumps(per_attack_dr(atype[test], y[test], st, thr)),
                        **binary_metrics(y[test], st, thr)})
    TR, TT = pd.DataFrame(TR), pd.DataFrame(TT)
    threshold_table(R, f"TEST: recall-oriented threshold (OOF FAR <= {far:g}) vs 0.5", TR, TT, CENTRAL_MODELS)
    FinalEvaluator(cfg, R, None)._per_attack_table(TR, TT, "all")
R.close()
