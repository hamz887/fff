# -*- coding: utf-8 -*-
"""
experiment.py — the evaluation protocol.

STAGE 2  CrossValidator: stratified 5-fold CV on the development set, executed phase by phase
         (1 central/all, 2 central/Boruta, 3 federated/all, 4 federated/voting). Everything that learns from data is
         fitted on the fold's training part: encoding, leakage audit, feature selection, scaling, and early stopping
         (a stratified val_frac hold-out of the fold training part for centralized models; each client's local
         validation part for federated models). The fold's validation part is used ONLY for evaluation (out-of-fold
         scores), so the CV metrics are unbiased estimates.
STAGE 3  FinalEvaluator: every model retrained on the full development set, evaluated ONCE on the held-out test set.
         AttackTypeClassifier: second-stage softmax over attack types (hierarchical detection).
All binary decisions use the standard rule score >= 0.5 (config.decision_threshold).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .config import (CENTRAL_MODELS, COMPARISONS, MODEL_NAMES, ORDER, PROPOSED, SET_NAMES, arch_of, long_name)
from .data_loader import FeatureScaler, FoldPreprocessor, holdout_split, partition_table, stratification_key
from .evaluate import binary_metrics, holm, mcnemar, paired_tests, per_attack_dr
from .feature_selection import BorutaXGB, federated_feature_voting
from .models import build_model, count_params
from .reporting import fmt_int, write_json
from .trainer import FederatedTrainer, Trainer, predict_scores

METRIC_COLS = [("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec=DR"), ("f1", "F1"),
               ("balanced_accuracy", "BalAcc"), ("roc_auc", "ROC-AUC"), ("far", "FAR"), ("mcc", "MCC"),
               ("g_mean", "G-mean")]


def metric_table(R, rows):
    R.table(["Features", "Model"] + [h for _, h in METRIC_COLS],
            [(r["setting"], MODEL_NAMES.get(r["model"], r["model"])) + tuple(f"{r[k]:.4f}" for k, _ in METRIC_COLS)
             for r in rows], widths=[10, 26] + [8] * len(METRIC_COLS))


def _state_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# =========================================================================================================
# STAGE 2 — cross-validation
# =========================================================================================================
class CrossValidator:
    PHASES = [(1, "central", "all"), (2, "central", "fs"), (3, "fl", "all"), (4, "fl", "fs")]

    def __init__(self, cfg, R, cache):
        self.cfg, self.R, self.cache = cfg, R, cache
        self.tr, self.ftr = Trainer(cfg, R), FederatedTrainer(cfg, R)
        self._folds = {}                       # fold -> (prep, Xtr, Xva): fitted once, reused by every phase
        self._es = {}                          # fold -> (fit, hold-out) positions inside the fold training part

    def fl_jobs(self, setting):
        jobs = ["fl_kan_ae"]
        if setting == "all" and self.cfg.fl_iid_baseline:
            jobs.append("fl_iid_kan_ae")
        if setting == "fs" and self.cfg.fl_central_fs_ablation:
            jobs.append("fl_kan_ae_cfs")
        return jobs

    def _fold_data(self, fold, ftr, fva, X, y, num_cols, cat_cols):
        """Encoding + in-fold leakage audit fitted on the fold's training part only (computed once per fold; the
        procedure is deterministic, so every phase sees the same features)."""
        if fold not in self._folds:
            R = self.R
            prep = FoldPreprocessor(self.cfg, num_cols, cat_cols).fit(X.iloc[ftr], y[ftr])
            self._folds[fold] = (prep, prep.transform(X.iloc[ftr]), prep.transform(X.iloc[fva]))
            partition_table(R, y, [("Fold training", ftr), ("Fold validation", fva)])
            R.kv("Encoded features", f"{len(prep.feature_names)} (label-encoded text: {', '.join(prep.ordinal) or 'none'})"
                 if prep.ordinal else f"{len(prep.feature_names)} (one-hot: {', '.join(prep.onehot) or 'none'}; "
                                      f"structural: {', '.join(prep.structural) or 'none'})")
            R.kv("Constant features removed", ", ".join(prep.dropped_constant) or "none")
            R.kv(f"Leakage audit (>= {self.cfg.leakage_threshold})",
                 ", ".join(f"{k} ({v})" for k, v in prep.leakage_report.items()) or
                 "no single-column separator (audited on an inner hold-out of the fold training part)")
        else:
            self.R.kv("Fold encoding", f"reused (fitted on fold training, {len(self._folds[fold][0].feature_names)} "
                                       f"features)")
        return self._folds[fold]

    def _es_split(self, fold, ftr, y, atype):
        """Early-stopping hold-out of the fold training part, stratified by attack type (same rows in every phase)."""
        if fold not in self._es:
            key, _ = stratification_key(atype[ftr], y[ftr], 10)
            self._es[fold] = holdout_split(key, self.cfg.val_frac, self.cfg.seed)
        return self._es[fold]

    def run(self, X, y, atype, dev, num_cols, cat_cols):
        cfg, R = self.cfg, self.R
        strat, _ = stratification_key(atype[dev], y[dev], cfg.n_folds)
        splits = list(StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
                      .split(np.zeros(len(dev)), strat))
        rows, oof, art, curves = [], {}, {}, []
        for phase, mode, setting in self.PHASES:
            for fold, (tp, vp) in enumerate(splits, 1):
                ftr, fva = dev[tp], dev[vp]
                R.stage(f"STAGE 2 | PHASE {phase}/4 | {'CENTRALIZED' if mode == 'central' else 'FEDERATED'} | "
                        f"{SET_NAMES[setting]} | FOLD {fold}/{cfg.n_folds}")
                prep, Xtr, Xva = self._fold_data(fold, ftr, fva, X, y, num_cols, cat_cols)
                names = prep.feature_names
                pos = {n: j for j, n in enumerate(names)}
                art.setdefault(fold, {"features_all": names, "leakage": prep.leakage_report})
                if setting == "fs" and "fs" not in art[fold]:
                    (sel, T, it), hit = self.cache.run(f"boruta_f{fold}", lambda: BorutaXGB(cfg, cfg.seed, R)
                                                        .select(Xtr, y[ftr], names))
                    T.to_csv(os.path.join(cfg.dirs["results"], f"fs_importance_fold{fold}.csv"), index=False)
                    art[fold]["fs"] = sel
                    R.kv("Boruta + XGBoost", f"{it} runs -> {len(sel)} of {len(names)} features selected"
                                             f"{' (cached)' if hit else ''}", style="green")
                    R.kv("Selected (ranked)", ", ".join(sel))
                cols = list(range(len(names))) if setting == "all" else [pos[n] for n in art[fold]["fs"]]
                if mode == "central":
                    A, V = Xtr[:, cols], Xva[:, cols]
                    sc = FeatureScaler(cfg.log_range_threshold).fit(A)
                    a, v = sc.transform(A), sc.transform(V)
                    fi, ei = self._es_split(fold, ftr, y, atype)
                    R.kv(f"Scaler (central, {setting})", f"signed-log1p on {len(sc.log_idx)} of {len(cols)} columns, "
                                                        f"Min-Max [-1,1] fitted on fold training")
                    R.kv("Early-stopping hold-out", f"{fmt_int(len(ei))} rows ({cfg.val_frac:.0%} of fold training, "
                                                    f"stratified); fold validation used for evaluation only")
                    for m in CENTRAL_MODELS:
                        def job(m=m):
                            model = build_model(m, len(cols), cfg)
                            yf = y[ftr]
                            model, h, info = self.tr.fit(model, a[fi], yf[fi], a[ei], yf[ei], cfg.seed, MODEL_NAMES[m])
                            return self._payload(model, h, info, v, y[fva])
                        P, hit = self.cache.run(f"p{phase}_f{fold}_{m}", job)
                        self._record(R, rows, oof, curves, P, hit, fold, setting, m, vp, len(dev))
                else:
                    fl_sel, parts = None, None
                    if setting == "fs":
                        (fl_sel, parts), hit = self.cache.run(
                            f"voting_f{fold}", lambda: federated_feature_voting(
                                cfg, Xtr, y[ftr], names, cfg.seed, R,
                                os.path.join(cfg.dirs["results"], f"fl_votes_fold{fold}.csv")))
                        art[fold]["fl_fs"] = fl_sel
                    for key in self.fl_jobs(setting):
                        def job(key=key):
                            c2 = [pos[n] for n in fl_sel] if (key == "fl_kan_ae" and setting == "fs") else cols
                            A2, V2 = Xtr[:, c2], Xva[:, c2]
                            pp = self.ftr.iid_partition(y[ftr], cfg.fl_clients, cfg.seed) if key == "fl_iid_kan_ae" \
                                else (parts if parts is not None else self.ftr.dirichlet_partition(
                                    y[ftr], cfg.fl_clients, cfg.fl_alpha, cfg.seed, cfg.fl_min_client_size))
                            fsc = FeatureScaler.federated([A2[p] for p in pp], cfg.log_range_threshold)
                            clients = [(fsc.transform(A2[p]), y[ftr][p]) for p in pp]
                            vv = fsc.transform(V2)
                            model = build_model("kan_ae", A2.shape[1], cfg)
                            model, h, info = self.ftr.fit(model, clients, cfg.seed, MODEL_NAMES[key])
                            P = self._payload(model, h, info, vv, y[fva])
                            P["dist"] = [(int(y[ftr][p].sum()), int(len(p))) for p in pp]
                            return P
                        P, hit = self.cache.run(f"p{phase}_f{fold}_{key}", job)
                        R.kv(f"Clients ({MODEL_NAMES[key]})", " | ".join(f"C{i + 1} {n} ({a_ / max(n, 1):.0%})"
                                                                          for i, (a_, n) in enumerate(P["dist"])))
                        self._record(R, rows, oof, curves, P, hit, fold, setting, key, vp, len(dev))
                pd.DataFrame(rows).to_csv(os.path.join(cfg.dirs["results"], "cv_results.csv"), index=False)
            models = CENTRAL_MODELS if mode == "central" else self.fl_jobs(setting)
            self.phase_table(R, rows, phase, mode, setting, models)
        self._folds.clear()                    # free the fold matrices before Stage 3
        self._es.clear()
        return pd.DataFrame(rows), oof, art, curves

    def _payload(self, model, h, info, v, yva):
        """Fold-validation scores (OOF) and CV metrics at the decision threshold (the fold validation part took no part
        in training or early stopping)."""
        sv = predict_scores(model, v)
        return {"sv": sv, "hist": h, "info": info, "params": count_params(model),
                "m": binary_metrics(yva, sv, self.cfg.decision_threshold)}

    @staticmethod
    def _record(R, rows, oof, curves, P, hit, fold, setting, m, vp, n_dev):
        i = P["info"]
        R.model_summary(f">> {MODEL_NAMES[m]} [{setting}] fold {fold} · {fmt_int(P['params'])} params", P["hist"], i,
                        P["m"], cached=hit, unit="Round" if m.startswith("fl_") else "Epoch")
        oof.setdefault((setting, m), np.full(n_dev, np.nan, dtype=np.float32))[vp] = P["sv"]
        rows.append({"fold": fold, "setting": setting, "model": m, "params": P["params"],
                     "epochs_run": i["epochs_run"], "best_epoch": i["best_epoch"], "train_time_s": i["time_s"],
                     **P["m"]})
        curves.append({"fold": fold, "setting": setting, "model": m, "history": P["hist"]})

    @staticmethod
    def phase_table(R, rows, phase, mode, setting, models):
        C = pd.DataFrame(rows)
        C = C[(C.setting == setting) & C.model.isin(models)]
        R.stage(f"STAGE 2 | PHASE {phase}/4 RESULT | {'CENTRALIZED' if mode == 'central' else 'FEDERATED'} | "
                f"{SET_NAMES[setting]} (fold validation, per fold and mean ± std)")
        out = []
        for m in [x for x in ORDER if x in C.model.unique()]:
            for met, lab in (("f1", "F1"), ("recall", "Recall"), ("mcc", "MCC"), ("far", "FAR"), ("roc_auc", "AUC")):
                v = C[C.model == m].sort_values("fold")[met].values
                out.append((MODEL_NAMES[m], lab) + tuple(f"{x:.4f}" for x in v) +
                           (f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}",))
        R.table(["Model", "Metric"] + [f"Fold {f}" for f in sorted(C.fold.unique())] + ["Mean ± std"], out,
                widths=[26, 8] + [9] * C.fold.nunique() + [18])

    @staticmethod
    def summary(CV, art, cfg, R):
        R.stage(f"STAGE 2 | CROSS-VALIDATION SUMMARY (fold validation, mean ± std over {cfg.n_folds} folds)")
        keys = [k for k, _ in METRIC_COLS]
        g = CV.groupby(["setting", "model"])
        M, S = g[keys].mean(), g[keys].std()
        R.table(["Features", "Model"] + [h for _, h in METRIC_COLS[:6]],
                [(s, MODEL_NAMES[m]) + tuple(f"{M.loc[(s, m), k]:.4f}±{S.loc[(s, m), k]:.4f}" for k, _ in METRIC_COLS[:6])
                 for s in ("all", "fs") for m in ORDER if (s, m) in M.index], widths=[10, 26] + [14] * 6)
        (M.round(6).astype(str) + " ± " + S.round(6).astype(str)).to_csv(os.path.join(cfg.dirs["results"],
                                                                                    "cv_summary.csv"))
        epochs = {(s, m): max(1, int(round(grp.best_epoch.mean()))) for (s, m), grp in g}
        R.section("Epochs (FL: rounds) for the final models = mean best epoch across folds")
        R.table(["Features", "Model", "Best per fold", "Final"],
                [(s, MODEL_NAMES[m], ", ".join(str(int(v)) for v in CV[(CV.setting == s) & (CV.model == m)]
                                              .sort_values("fold").best_epoch), epochs[(s, m)])
                 for s in ("all", "fs") for m in ORDER if (s, m) in epochs], widths=[10, 26, 24, 8])
        n_test, n_train = 1 / cfg.n_folds, 1 - 1 / cfg.n_folds     # corrected t-test: validation vs training share
        tests = []
        for ma, sa, mb, sb in COMPARISONS:
            A_ = CV[(CV.setting == sa) & (CV.model == ma)].sort_values("fold")
            B_ = CV[(CV.setting == sb) & (CV.model == mb)].sort_values("fold")
            if len(A_) == len(B_) > 1:
                for met in ("f1", "mcc"):
                    d = A_[met].values - B_[met].values
                    tests.append({"comparison": f"{MODEL_NAMES[ma]} [{sa}] vs {MODEL_NAMES[mb]} [{sb}]", "metric": met,
                                  "mean_diff": d.mean(), "std_diff": d.std(ddof=1),
                                  **paired_tests(A_[met].values, B_[met].values, n_train, n_test)})
        T = pd.DataFrame(tests)
        if not T.empty:
            for met in T.metric.unique():
                k = T.metric == met
                T.loc[k, "p_holm"], T.loc[k, "p_corr_holm"] = holm(T.loc[k, "p"]), holm(T.loc[k, "p_corr"])
            T.to_csv(os.path.join(cfg.dirs["results"], "cv_statistical_tests.csv"), index=False)
            R.section("Statistical tests over folds: corrected resampled t-test (Nadeau & Bengio 2003) + Holm "
                      "(reference p = p_corr_holm)")

            def f4(v):
                return "nan" if pd.isna(v) else f"{v:.4f}"
            R.table(["Comparison", "Metric", "Mean diff", "p corr", "p corr Holm"],
                    [(r.comparison, r.metric.upper(), f"{r.mean_diff:+.4f}", f4(r.p_corr), f4(r.p_corr_holm))
                     for r in T.itertuples()], widths=[62, 7, 10, 9, 12])
        fq = pd.Series([f for a in art.values() for f in a.get("fs", [])]).value_counts()
        fq.rename("folds_selected").to_csv(os.path.join(cfg.dirs["results"], "feature_selection_stability.csv"))
        leak = {str(f): a["leakage"] for f, a in art.items()}
        write_json(leak, os.path.join(cfg.dirs["results"], "cv_leakage_audit.json"))
        deploy = CrossValidator.select_deployment(M, cfg, R)
        return {"epochs": epochs, "leakage_union": sorted({c for a in art.values() for c in a["leakage"]}), **deploy}

    @staticmethod
    def select_deployment(M, cfg, R):
        """Deployment rule, fixed in advance and based on mean cross-validation MCC only (the test set is never used):
        'auto' deploys Fed-KAN-AE when it is non-inferior to the best centralized model, i.e. the centralized model
        exceeds it by at most deploy_margin_mcc; otherwise the best centralized model is deployed. An explicit model key
        forces that model. The other one of {Fed-KAN-AE, best centralized model} is retained for comparison."""
        def best_of(m, settings=("all", "fs")):
            c = [(M.loc[(s, m), "mcc"], s) for s in settings if (s, m) in M.index]
            return max(c) if c else (np.nan, "fs")
        p_set = ("all", "fs") if cfg.proposed_setting == "auto" else (cfg.proposed_setting,)
        f_mcc, f_set = best_of(PROPOSED, p_set)
        b_mcc, b_set, b_m = max((best_of(m) + (m,) for m in CENTRAL_MODELS), key=lambda t: t[0])
        margin = b_mcc - f_mcc
        if cfg.deploy_model == "auto":
            if margin <= cfg.deploy_margin_mcc:
                primary, why = (PROPOSED, f_set), f"non-inferior to the best centralized model (margin " \
                                                  f"{cfg.deploy_margin_mcc:g} MCC)"
            else:
                primary, why = (b_m, b_set), f"best centralized model exceeds Fed-KAN-AE by {margin:.4f} MCC " \
                                             f"(> margin {cfg.deploy_margin_mcc:g})"
        else:
            st = f_set if cfg.deploy_model == PROPOSED else best_of(cfg.deploy_model)[1]
            primary, why = (cfg.deploy_model, st), "configured (deploy_model)"
        secondary = (b_m, b_set) if primary[0] == PROPOSED else (PROPOSED, f_set)
        R.section("Deployment decision (mean cross-validation MCC; the test set is not used)")
        R.table(["Role", "Model", "Paradigm", "Features", "Mean CV MCC"],
                [(role, long_name(k), "federated" if k.startswith("fl_") else "centralized",
                  "federated voting" if (k == PROPOSED and st == "fs") else st, f"{M.loc[(st, k), 'mcc']:.4f}")
                 for role, (k, st) in (("primary", primary), ("retained", secondary))], widths=[10, 32, 13, 18, 12])
        R.kv("Primary deployed architecture", f"{long_name(primary[0])} — {why}", style="green")
        R.kv("Best centralized - Fed-KAN-AE", f"{margin:+.4f} mean CV MCC")
        return {"deploy": primary, "secondary": secondary, "deploy_reason": why, "deploy_margin": float(margin)}


# =========================================================================================================
# STAGE 3 — in-domain evaluation on the independent test set
# =========================================================================================================
class FinalEvaluator:
    def __init__(self, cfg, R, cache):
        self.cfg, self.R, self.cache = cfg, R, cache
        self.tr, self.ftr = Trainer(cfg, R), FederatedTrainer(cfg, R)

    def run(self, X, y, atype, dev, test, num_cols, cat_cols, oof, dec):
        cfg, R = self.cfg, self.R
        R.stage(f"STAGE 3 | IN-DOMAIN EVALUATION: RETRAIN ON THE {1 - cfg.test_frac:.0%} DEVELOPMENT SET, "
                f"EVALUATE ONCE ON THE {cfg.test_frac:.0%} TEST SET")
        yd, yt, at = y[dev], y[test], atype[test]
        prep = FoldPreprocessor(cfg, num_cols, cat_cols).fit(X.iloc[dev], yd, forced_drop=dec["leakage_union"])
        Xd, Xt = prep.transform(X.iloc[dev]), prep.transform(X.iloc[test])
        names = prep.feature_names
        pos = {n: j for j, n in enumerate(names)}
        R.kv("Encoded features (development)", len(names))
        (sel, T, it), _ = self.cache.run("boruta_final", lambda: BorutaXGB(cfg, cfg.seed, R).select(Xd, yd, names))
        T.to_csv(os.path.join(cfg.dirs["results"], "fs_importance_final.csv"), index=False)
        R.kv("Boruta + XGBoost (final)", f"{len(sel)} of {len(names)} features: {', '.join(sel)}")
        (fl_sel, parts), _ = self.cache.run("voting_final", lambda: federated_feature_voting(
            cfg, Xd, yd, names, cfg.seed, R, os.path.join(cfg.dirs["results"], "fl_votes_final.csv")))
        cols = {"all": list(range(len(names))), "fs": [pos[n] for n in sel]}
        scalers = {s: FeatureScaler(cfg.log_range_threshold).fit(Xd[:, c]) for s, c in cols.items()}
        jobs = [(s, m) for s in ("all", "fs") for m in CENTRAL_MODELS] + \
               [("all", "fl_kan_ae")] + ([("all", "fl_iid_kan_ae")] if cfg.fl_iid_baseline else []) + \
               [("fs", "fl_kan_ae")] + ([("fs", "fl_kan_ae_cfs")] if cfg.fl_central_fs_ablation else [])
        rows, preds, scores, states = [], {}, {}, {}
        thr = cfg.decision_threshold
        for s, m in jobs:
            ep = dec["epochs"][(s, m)]

            def job(s=s, m=m, ep=ep):
                if not m.startswith("fl_"):
                    sc = scalers[s]
                    model = build_model(m, len(cols[s]), cfg)
                    model, h, info = self.tr.fit_fixed(model, sc.transform(Xd[:, cols[s]]), yd, cfg.seed, ep,
                                                       MODEL_NAMES[m])
                    return {"st": predict_scores(model, sc.transform(Xt[:, cols[s]])), "state": _state_cpu(model),
                            "d": len(cols[s]), "info": info, "hist": h, "params": count_params(model),
                            "features": [names[i] for i in cols[s]], "scaler": sc.state(), "parts": None}
                c2 = [pos[n] for n in fl_sel] if (m == "fl_kan_ae" and s == "fs") else cols[s]
                pp = self.ftr.iid_partition(yd, cfg.fl_clients, cfg.seed) if m == "fl_iid_kan_ae" else parts
                fsc = FeatureScaler.federated([Xd[p][:, c2] for p in pp], cfg.log_range_threshold)
                clients = [(fsc.transform(Xd[p][:, c2]), yd[p]) for p in pp]
                model = build_model("kan_ae", len(c2), cfg)
                model, h, info = self.ftr.fit_fixed(model, clients, cfg.seed, ep, MODEL_NAMES[m])
                return {"st": predict_scores(model, fsc.transform(Xt[:, c2])), "state": _state_cpu(model),
                        "d": len(c2), "info": info, "hist": h, "params": count_params(model),
                        "features": [names[i] for i in c2], "scaler": fsc.state(), "parts": pp}
            P, hit = self.cache.run(f"final_{s}_{m}", job)
            R.model_summary(f">> FINAL {MODEL_NAMES[m]} [{s}] · retrained on the development set · "
                            f"{fmt_int(P['params'])} params", P["hist"], P["info"], cached=hit,
                            unit="Round" if m.startswith("fl_") else "Epoch")
            mt = binary_metrics(yt, P["st"], thr)
            r = {"setting": s, "model": m, "d": P["d"], "params": P["params"], "epochs": ep,
                 "per_attack_dr": json.dumps(per_attack_dr(at, yt, P["st"], thr)), **mt}
            rows.append(r)
            preds[(s, m)] = (P["st"] >= thr).astype(np.int8)
            scores[(s, m)], states[(s, m)] = P["st"], P
            R.say(f"      TEST (threshold {thr:g}): Accuracy {r['accuracy']:.4f} | F1 {r['f1']:.4f} | "
                  f"Recall {r['recall']:.4f} | Precision {r['precision']:.4f} | FAR {r['far']:.4f} | MCC {r['mcc']:.4f} | "
                  f"AUC {r['roc_auc']:.4f}", "cyan")
        TR = pd.DataFrame(rows)
        TR.to_csv(os.path.join(cfg.dirs["results"], "test_results.csv"), index=False)
        R.section(f"INDEPENDENT TEST RESULTS (decision threshold {thr:g})")
        metric_table(R, [r for s in ("all", "fs") for m in ORDER
                         for r in TR[(TR.setting == s) & (TR.model == m)].to_dict("records")])
        self._mcnemar(TR, preds, yt)
        self._decomposition(TR)
        arr = {"y_dev": yd.astype(np.int8), "atype_dev": np.asarray(atype[dev], str), "y_test": yt.astype(np.int8),
               "atype_test": np.asarray(at, str)}
        arr.update({f"oof__{s}__{m}": v.astype(np.float32) for (s, m), v in oof.items()})
        arr.update({f"test__{s}__{m}": v.astype(np.float32) for (s, m), v in scores.items()})
        np.savez_compressed(os.path.join(cfg.dirs["results"], "scores.npz"), **arr)
        return {"prep": prep, "names": names, "sel": sel, "cols": cols, "scalers": scalers, "Xd": Xd, "Xt": Xt,
                "states": states, "TR": TR, "oof": oof}

    def _mcnemar(self, TR, preds, yt):
        Y = TR.set_index(["setting", "model"])
        mc = []
        for ma, sa, mb, sb in COMPARISONS:
            if (sa, ma) in preds and (sb, mb) in preds:
                mc.append({"comparison": f"{MODEL_NAMES[ma]} [{sa}] vs {MODEL_NAMES[mb]} [{sb}]",
                           **mcnemar(yt, preds[(sa, ma)], preds[(sb, mb)]),
                           "d_mcc": Y.loc[(sa, ma)].mcc - Y.loc[(sb, mb)].mcc, "d_f1": Y.loc[(sa, ma)].f1 - Y.loc[(sb, mb)].f1})
        if not mc:
            return
        MC = pd.DataFrame(mc)
        MC["p_holm"] = holm(MC.p.values)
        MC.to_csv(os.path.join(self.cfg.dirs["results"], "test_mcnemar.csv"), index=False)
        self.R.section("McNemar test on the independent test set, Holm-adjusted (Dietterich 1998)")
        self.R.table(["Comparison", "d MCC", "d F1", "p Holm"],
                     [(r.comparison, f"{r.d_mcc:+.4f}", f"{r.d_f1:+.4f}", f"{r.p_holm:.3g}") for r in MC.itertuples()],
                     widths=[64, 10, 10, 10])

    def _decomposition(self, TR):
        chains = {"A": [("kan_ae", "all"), ("fl_iid_kan_ae", "all"), ("fl_kan_ae", "all")],
                  "B": [("kan_ae", "fs"), ("fl_kan_ae_cfs", "fs"), ("fl_kan_ae", "fs")]}
        Y = TR.set_index(["model", "setting"])
        out = []
        for ch, steps in chains.items():
            prev = None
            for k, (m, s) in enumerate(steps, 1):
                if (m, s) not in Y.index:
                    continue
                r = {"chain": ch, "step": k, "model": m, "setting": s, **{x: float(Y.loc[(m, s)][x]) for x in ("mcc", "f1", "far")}}
                for x in ("mcc", "f1", "far"):
                    r[f"delta_{x}"] = r[x] - prev[x] if prev else np.nan
                out.append(r)
                prev = r
        pd.DataFrame(out).to_csv(os.path.join(self.cfg.dirs["results"], "fl_decomposition.csv"), index=False)


class AttackTypeClassifier:
    """Second stage of the hierarchical detector: a softmax classifier over the attack types, trained on the attack
    records of the development set only, with the same architecture, features and scaler as the deployed binary
    detector (federally, over the same clients, when the deployed model is federated). Early stopping on a stratified
    val_frac hold-out of those records (federated: each client's local validation part). Applied only to records the
    binary stage flags as Attack."""

    def __init__(self, cfg, R):
        self.cfg, self.R = cfg, R

    def run(self, key, setting, fe, y, atype, dev, test, thr_deploy, bin_scores_test):
        cfg, R = self.cfg, self.R
        P0 = fe["states"][(setting, key)]
        federated = key.startswith("fl_")
        R.stage(f"STAGE 3 | ATTACK-TYPE CLASSIFICATION ({long_name(key)}: binary detector -> softmax over attack "
                f"types, {'federated' if federated else 'centralized'} training)")
        pos = {n: j for j, n in enumerate(fe["names"])}
        cols = [pos[n] for n in P0["features"]]
        sc = FeatureScaler.from_state(P0["scaler"])
        a_dev = y[dev] == 1
        classes = sorted(np.unique(atype[dev][a_dev]).tolist())
        cid = {c: i for i, c in enumerate(classes)}
        lab = np.full(len(dev), -1, dtype=np.int64)
        lab[a_dev] = [cid[t] for t in atype[dev][a_dev]]
        att_idx = np.where(a_dev)[0]
        Xs = sc.transform(fe["Xd"][:, cols])
        model = build_model(arch_of(key), len(cols), cfg, n_out=len(classes))
        label = f"{MODEL_NAMES[key]} (types)"
        if federated:
            clients = []
            for p in P0["parts"]:                       # same clients as the binary Fed-KAN-AE, attack rows only
                q = np.asarray(p)[a_dev[p]]
                if len(q) >= 2:
                    clients.append((Xs[q], lab[q]))
            model, h, info = FederatedTrainer(cfg, R).fit(model, clients, cfg.seed, label, n_classes=len(classes))
            R.kv("Federated clients (attack records)", f"{len(clients)} clients | local validation {cfg.val_frac:.0%} "
                                                       f"per client")
        else:
            skey, _ = stratification_key(lab[att_idx], np.ones(len(att_idx)), 10)
            fi, ei = holdout_split(skey, cfg.val_frac, cfg.seed)
            tr_i, va_i = att_idx[fi], att_idx[ei]
            model, h, info = Trainer(cfg, R).fit(model, Xs[tr_i], lab[tr_i], Xs[va_i], lab[va_i], cfg.seed, label,
                                                 n_classes=len(classes))
        R.model_summary(f">> Attack-type classifier ({long_name(key)}, softmax over {len(classes)} types)", h, info,
                        unit="Round" if federated else "Epoch")
        R.kv("Classes", f"{len(classes)} attack types | {info['epochs_run']} {'rounds' if federated else 'epochs'} "
                        f"(best {info['best_epoch']}) | {count_params(model):,} params")
        Xt = sc.transform(fe["Xt"][:, cols])
        at, yt = atype[test], y[test]
        m_att = yt == 1
        Pall = predict_scores(model, Xt, multiclass=True)
        pred_all = np.array(classes)[Pall.argmax(1)]
        pred = pred_all[m_att]
        rep = classification_report(at[m_att], pred, labels=classes, output_dict=True, zero_division=0)
        per = pd.DataFrame(rep).T.loc[classes, ["precision", "recall", "f1-score", "support"]]
        per.to_csv(os.path.join(cfg.dirs["results"], "attack_type_per_class.csv"))
        # end-to-end hierarchical evaluation on the whole test set (Normal + attack types)
        e2e = np.where(bin_scores_test >= thr_deploy, pred_all, "Normal")
        labels = ["Normal"] + classes
        e2e_true = np.where(yt == 1, at, "Normal")
        summ = {"attack_only_accuracy": accuracy_score(at[m_att], pred),
                "attack_only_macro_f1": f1_score(at[m_att], pred, average="macro", zero_division=0),
                "attack_only_weighted_f1": f1_score(at[m_att], pred, average="weighted", zero_division=0),
                "end_to_end_accuracy": accuracy_score(e2e_true, e2e),
                "end_to_end_macro_f1": f1_score(e2e_true, e2e, labels=labels, average="macro", zero_division=0),
                "n_classes": len(classes)}
        write_json(summ, os.path.join(cfg.dirs["results"], "attack_type_summary.json"))
        pd.DataFrame(confusion_matrix(e2e_true, e2e, labels=labels), index=labels, columns=labels) \
            .to_csv(os.path.join(cfg.dirs["results"], "hierarchical_confusion_matrix.csv"))
        R.table(["Attack type", "Precision", "Recall", "F1", "Support"],
                [(c, f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r['f1-score']:.4f}", fmt_int(r.support))
                 for c, r in per.iterrows()], widths=[26, 11, 10, 10, 10])
        R.kv("Attack-only (true attacks)", f"accuracy {summ['attack_only_accuracy']:.4f} | macro-F1 "
                                           f"{summ['attack_only_macro_f1']:.4f}")
        R.kv("End-to-end (Normal + types)", f"accuracy {summ['end_to_end_accuracy']:.4f} | macro-F1 "
                                            f"{summ['end_to_end_macro_f1']:.4f}")
        return model, classes, summ
