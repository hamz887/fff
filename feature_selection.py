# -*- coding: utf-8 -*-
"""
feature_selection.py — Boruta with XGBoost importance (Kursa & Rudnicki 2010; Chen & Guestrin 2016) and federated
feature voting. Always fitted on training data only (a CV fold's training part, or the development set).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from xgboost import XGBClassifier

from .reporting import TrainProgress
from .trainer import FederatedTrainer

MIN_SELECTED = 2          # the KAN-AE needs d >= 2 (latent < d)


def stratified_subsample(y, n, seed):
    """Class-proportional subsample of at most n indices."""
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    return np.sort(np.concatenate([rng.choice(np.where(y == c)[0], size=max(1, int(round(n * (y == c).mean()))),
                                              replace=False) for c in np.unique(y)]))


class BorutaXGB:
    """All-relevant selection: shadow features, binomial test with Bonferroni, TentativeRoughFix.
    The number of features is decided by the test (no fixed k); accepted features are ranked by mean gain."""

    def __init__(self, cfg, seed, R=None):
        self.cfg, self.seed, self.R = cfg, seed, R

    def _xgb(self, s):
        return XGBClassifier(n_estimators=100, max_depth=self.cfg.boruta_depth, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", importance_type="gain",
                             eval_metric="logloss", random_state=s, n_jobs=-1, verbosity=0)

    def select(self, X, y, names, label="Boruta"):
        cfg = self.cfg
        idx = stratified_subsample(y, cfg.boruta_sample, self.seed)
        X, y = X[idx], y[idx]
        d, rng = X.shape[1], np.random.RandomState(self.seed)
        status, hits = np.zeros(d, int), np.zeros(d)
        imp_h, sh_max = np.full((cfg.boruta_max_iter, d), np.nan), np.zeros(cfg.boruta_max_iter)
        it = 0
        with TrainProgress(self.R, label, cfg.boruta_max_iter, "run", "shadow test") as pb:
            for it in range(1, cfg.boruta_max_iter + 1):
                pb.start(it, 1)
                act = np.where(status >= 0)[0]
                Xa = X[:, act]
                sh = Xa.copy()                                         # column-wise shuffled shadow copies
                for j in range(sh.shape[1]):
                    rng.shuffle(sh[:, j])
                imp = self._xgb(self.seed + it).fit(np.hstack([Xa, sh]), y).feature_importances_
                real, shad = imp[:len(act)], imp[len(act):]
                imp_h[it - 1, act], sh_max[it - 1] = real, shad.max()
                hits[act[real > shad.max()]] += 1
                tent = np.where(status == 0)[0]
                a = cfg.boruta_alpha / max(len(tent), 1)
                status[tent[binom.sf(hits[tent] - 1, it, 0.5) < a]] = 1
                status[tent[binom.cdf(hits[tent], it, 0.5) < a]] = -1
                info = (f"confirmed {(status == 1).sum()} · tentative {(status == 0).sum()} · "
                        f"rejected {(status == -1).sum()}")
                pb.step(1, info)
                pb.end(info)
                if (status == 0).sum() == 0:
                    break
        imp_h, med_sh = imp_h[:it], float(np.median(sh_max[:it]))
        rows = []
        for j in range(d):
            h = imp_h[:, j][~np.isnan(imp_h[:, j])]
            med = float(np.median(h)) if len(h) else 0.0
            dec = {1: "Confirmed", -1: "Rejected"}.get(status[j]) or \
                ("Tentative->Accepted" if med > med_sh else "Tentative->Rejected")
            rows.append({"feature": names[j], "mean_importance": float(h.mean()) if len(h) else 0.0,
                         "hits": int(hits[j]), "runs": len(h), "decision": dec})
        T = pd.DataFrame(rows).sort_values("mean_importance", ascending=False, kind="stable").reset_index(drop=True)
        T["importance_%"] = 100 * T.mean_importance / max(T.mean_importance.sum(), 1e-12)
        T.insert(0, "rank", np.arange(1, len(T) + 1))
        sel = T[T.decision.isin(["Confirmed", "Tentative->Accepted"])].feature.tolist()
        if len(sel) < MIN_SELECTED:                                    # fallback: top-ranked features
            sel = T.feature.tolist()[:max(MIN_SELECTED, d // 2)]
        return sel, T, it


def federated_feature_voting(cfg, X, y, names, seed, R, csv_path):
    """Each eligible client runs Boruta on its own data and sends only its feature list; the server keeps features
    selected by >= fs_vote_frac of the voting clients. Fallback when fewer than fs_min_features pass: the most-voted
    features (at least one vote); when no client is eligible to vote, no selection is possible and all features are
    kept. Returns (selected features, the client partition used)."""
    parts = FederatedTrainer.dirichlet_partition(y, cfg.fl_clients, cfg.fl_alpha, seed, cfg.fl_min_client_size)
    pos = {n: j for j, n in enumerate(names)}
    votes, voters, excl = np.zeros(len(names), int), 0, []
    for i, p in enumerate(parts, 1):
        if len(p) < cfg.fs_min_client_samples or len(np.unique(y[p])) < 2:
            excl.append(f"C{i}")
            continue
        sel_i, _, _ = BorutaXGB(cfg, cfg.seed, R).select(X[p], y[p], names, label=f"Client C{i} Boruta")
        voters += 1
        votes[[pos[f] for f in sel_i]] += 1
    chosen = [n for n, v in zip(names, votes) if voters and v >= cfg.fs_vote_frac * voters]
    fb = len(chosen) < max(cfg.fs_min_features, MIN_SELECTED)
    if fb:
        voted = sorted((j for j in range(len(names)) if votes[j] > 0), key=lambda j: (-votes[j], j))
        top = voted[:max(cfg.fs_min_features, MIN_SELECTED)]
        chosen = [names[j] for j in sorted(top)] if len(top) >= MIN_SELECTED else list(names)
    pd.DataFrame({"feature": names, "votes": votes, "voters": voters, "selected": [n in chosen for n in names]}) \
        .to_csv(csv_path, index=False)
    R.kv("Federated voting", f"{voters} voting clients (excluded: {', '.join(excl) or 'none'}) -> "
                             f"{len(chosen)} features{' (fallback used)' if fb else ''}"
                             f"{'; no eligible client, all features kept' if not voters else ''}")
    return chosen, parts
