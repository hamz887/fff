# -*- coding: utf-8 -*-
"""figures.py — publication figures (PNG, 300 dpi) written to outputs/figures only."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .config import CENTRAL_MODELS, MODEL_NAMES, ORDER, SET_NAMES, long_name


def make_figures(cfg, CV, curves, fe, arch, setting, thr, fs_table, edge_df, transfer, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
                         "grid.alpha": 0.3})
    fd, rd = cfg.dirs["figures"], cfg.dirs["results"]
    TR = fe["TR"]
    made = []

    def save(fig, name):
        fig.savefig(os.path.join(fd, name))
        plt.close(fig)
        made.append(name)

    def attempt(name, fn):
        """Each figure is independent: one failure does not prevent the others."""
        try:
            fn()
        except Exception as e:
            plt.close("all")
            R.say(f"    WARNING: figure '{name}' failed ({type(e).__name__}: {e}); the others were saved.")

    with np.load(os.path.join(rd, "scores.npz")) as npz:
        S = {k: npz[k] for k in npz.files}
    yt = S["y_test"]

    def roc():
        for s in ("all", "fs"):
            fig, ax = plt.subplots(figsize=(6, 5))
            for m in ORDER:
                k = f"test__{s}__{m}"
                if k in S:
                    fpr, tpr, _ = roc_curve(yt, S[k])
                    ax.plot(fpr, tpr, label=f"{MODEL_NAMES[m]} (AUC {roc_auc_score(yt, S[k]):.4f})")
            ax.set(xscale="log", xlim=(1e-5, 1), xlabel="False alarm rate (log)", ylabel="Detection rate",
                   title=f"ROC on the independent test set — {SET_NAMES[s].title()}")
            ax.legend(fontsize=8, loc="lower right")
            save(fig, f"fig01_roc_test_{s}.png")

    def cm_deployed():
        st = S[f"test__{setting}__{arch}"]
        cm = confusion_matrix(yt, (st >= thr).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        pc = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(pc, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pc[i, j]:.2%})", ha="center", va="center",
                        color="white" if pc[i, j] > .5 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
               xlabel="Predicted", ylabel="Actual", title=f"Deployed {long_name(arch)} (threshold {thr:g})")
        ax.grid(False)
        save(fig, "fig02_confusion_deployed.png")

    def cv_box():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, met in zip(axes, ("f1", "mcc")):
            data, lab = [], []
            for s in ("all", "fs"):
                for m in ORDER:
                    v = CV[(CV.setting == s) & (CV.model == m)][met].values
                    if len(v):
                        data.append(v)
                        lab.append(f"{MODEL_NAMES[m]}\n[{s}]")
            ax.boxplot(data)
            ax.set_xticks(range(1, len(lab) + 1))
            ax.set_xticklabels(lab, rotation=60, fontsize=7)
            ax.set_title(f"Cross-validation {met.upper()} (validation folds)")
        save(fig, "fig03_cv_boxplot.png")

    def importance():
        T = fs_table.sort_values("importance_%")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(T))))
        ax.barh(T.feature, T["importance_%"], color=["#31a354" if d_.startswith(("Conf", "Tentative->A")) else "#bdbdbd"
                                                      for d_ in T.decision])
        ax.tick_params(axis="y", labelsize=7)
        ax.set(xlabel="Mean XGBoost gain (%)", title="Boruta + XGBoost importance (green = selected)")
        save(fig, "fig04_feature_importance.png")

    def per_attack():
        rows = [{"k": f"{MODEL_NAMES[r.model]} [{r.setting}]", "t": t, "v": v}
                for r in TR.itertuples() for t, v in json.loads(r.per_attack_dr).items()]
        Pm = pd.DataFrame(rows).pivot(index="k", columns="t", values="v")
        fig, ax = plt.subplots(figsize=(1 + .7 * Pm.shape[1], 1 + .42 * Pm.shape[0]))
        im = ax.imshow(Pm.values, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(Pm.shape[1]))
        ax.set_xticklabels(Pm.columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(Pm.shape[0]))
        ax.set_yticklabels(Pm.index, fontsize=8)
        for i in range(Pm.shape[0]):
            for j in range(Pm.shape[1]):
                ax.text(j, i, f"{Pm.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=.03)
        ax.set_title("Detection rate per attack type (test)")
        save(fig, "fig05_per_attack_detection.png")

    def training_curves():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for c in curves:
            if c["fold"] == 1 and c["setting"] == "all":
                h = pd.DataFrame(c["history"])
                ax[0].plot(h.epoch, h.val_loss, marker=".", label=MODEL_NAMES[c["model"]])
                ax[1].plot(h.epoch, h.val_f1, marker=".", label=MODEL_NAMES[c["model"]])
        ax[0].set(title="Validation detection loss (fold 1, all features)", xlabel="Epoch / round")
        ax[1].set(title="Validation F1 (fold 1, all features)", xlabel="Epoch / round")
        ax[1].legend(fontsize=7)
        save(fig, "fig06_training_curves.png")

    def edge_cost():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (c, t) in zip(axes, [("flops", "FLOPs / sample"), ("memory_kb", "Memory (KB)"),
                                     ("latency_median_ms", "Latency, batch=1 (ms)")]):
            for j, s in enumerate(("all", "fs")):
                e = edge_df[edge_df.setting == s].set_index("model").reindex(CENTRAL_MODELS)
                ax.bar(np.arange(3) + j * .4, e[c].values, .4, label=SET_NAMES[s].title())
            ax.set_xticks(np.arange(3) + .2)
            ax.set_xticklabels([MODEL_NAMES[m] for m in CENTRAL_MODELS], fontsize=8)
            ax.set_title(t)
        axes[0].legend(fontsize=8)
        save(fig, "fig07_edge_cost.png")

    def type_confusion():
        C = pd.read_csv(os.path.join(rd, "hierarchical_confusion_matrix.csv"), index_col=0)
        Cn = C.div(C.sum(1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(1 + .55 * len(C), 1 + .5 * len(C)))
        ax.imshow(Cn.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(C)))
        ax.set_xticklabels(C.columns, rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(len(C)))
        ax.set_yticklabels(C.index, fontsize=7)
        ax.grid(False)
        ax.set(xlabel="Predicted", ylabel="Actual", title="Hierarchical detection: Normal + attack types (row-normalised)")
        save(fig, "fig08_attack_type_confusion.png")

    def transfer_fig():
        D = pd.read_csv(os.path.join(rd, "transfer_results.csv"))
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(D))
        for j, met in enumerate(("f1", "mcc", "recall")):
            ax.bar(x + j * .27, D[met], .27, label=met.upper())
        ax.set_xticks(x + .27)
        ax.set_xticklabels([s.replace(", ", "\n") for s in D.scenario])
        ax.set(ylim=(min(0, D[["f1", "mcc", "recall"]].min().min()), 1),
               title="Cross-dataset generalization (CICIoT2023, two-phase fine-tuning)")
        ax.legend()
        save(fig, "fig09_transfer.png")

    attempt("ROC", roc)
    attempt("confusion matrix", cm_deployed)
    attempt("CV boxplot", cv_box)
    attempt("feature importance", importance)
    attempt("per-attack detection", per_attack)
    attempt("training curves", training_curves)
    if edge_df is not None:
        attempt("edge cost", edge_cost)
    if os.path.isfile(os.path.join(rd, "hierarchical_confusion_matrix.csv")):
        attempt("attack-type confusion", type_confusion)
    if transfer is not None:
        attempt("transfer", transfer_fig)
    return made
