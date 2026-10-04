# -*- coding: utf-8 -*-
"""
config.py — hyper-parameters, dataset paths, seeds, device and project-wide constants.

Every experiment setting lives in the ``Config`` dataclass. ``Cache`` (reporting.py) snapshots it, so a run can only
be resumed with identical settings.
"""
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # project root (data/, outputs/)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # dynamic GPU/CPU allocation

# ---------------------------------------------------------------------------------------------------------------
# Data-leakage registry — capture timestamps, host identities, per-session identifiers/counters and testbed
# fingerprints. Such columns let a model recognise WHEN/WHO/WHICH SESSION instead of learning attack behaviour, so they
# are removed before any learning (spurious correlations, Arp et al. 2022). A further single-feature audit runs inside
# every cross-validation fold (FoldPreprocessor, leakage_threshold).
# ---------------------------------------------------------------------------------------------------------------
LEAKAGE_COLUMNS: Dict[str, str] = {
    "frame.time": "capture timestamp: attack families were recorded in separate, non-overlapping time windows",
    "ip.src_host": "source IP: host identity (attacker host 192.168.0.170 is 100% attack)",
    "ip.dst_host": "destination IP: host identity (192.168.0.170 as destination is 100% attack)",
    "tcp.options": "TCP options: fingerprint of the attacker host's TCP stack (timestamp option in attack traffic, "
                   "absent in normal IoT devices), a testbed artefact",
    "tcp.srcport": "TCP source port: per-connection port (session identity)",
    "tcp.dstport": "TCP destination port: port on reply packets (session identity)",
}

# Columns excluded for reasons other than leakage (none by default), e.g.
#   "some.column": "reason for the exclusion"
NON_LEAKAGE_EXCLUSIONS: Dict[str, str] = {}


@dataclass
class Config:
    # ---- data
    source_path: str = os.path.join(BASE, "data", "DNN-EdgeIIoT-dataset.csv")   # Edge-IIoTset (source)
    target_path: str = os.path.join(BASE, "data", "CICIoT2023.csv")             # CICIoT2023 CSV (target)
    out_dir: str = os.path.join(BASE, "outputs")
    target_col: str = "Attack_label"
    attack_type_col: str = "Attack_type"
    sample_frac: float = 1.0          # stratified sample of the source (1.0 = full file)
    target_sample: int = 0            # rows sampled from the target CSV (0 = the full file)
    leakage_cols: Dict[str, str] = field(default_factory=lambda: dict(LEAKAGE_COLUMNS))
    excluded_cols: Dict[str, str] = field(default_factory=lambda: dict(NON_LEAKAGE_EXCLUSIONS))
    placeholder_tokens: List[str] = field(default_factory=lambda: [
        "0", "0.0", "", "nan", "NaN", "None", "none", "null", "NULL", "-", "0x00000000"])
    numeric_min_frac: float = 0.99    # a text column becomes numeric if >= 99 % of its values parse as numbers
    text_encoding: str = "ordinal"    # "ordinal" (label encoding, one feature per text column) | "onehot"
    drop_constant_features: bool = False   # False: every cleaned feature is kept, constant ones included
    onehot_max_categories: int = 30   # "onehot" mode only
    onehot_min_freq: int = 20         # categories rarer than this in the training data are coded as "other"
    leakage_threshold: float = 0.999  # in-fold single-feature audit (balanced accuracy of a depth-1 tree)
    leakage_audit_frac: float = 0.2   # inner hold-out of the fold TRAINING part used by that audit
    # ---- protocol
    seed: int = 42
    test_frac: float = 0.30           # 70 % development / 30 % independent held-out test
    n_folds: int = 5
    # ---- feature selection (Boruta with XGBoost importance)
    boruta_sample: int = 200_000
    boruta_max_iter: int = 100
    boruta_alpha: float = 0.05
    boruta_depth: int = 5
    # ---- models
    kan_hidden: int = 16
    kan_grid: int = 5
    kan_order: int = 3
    latent_ratio: float = 0.25        # hybrid: one (smallest) latent size = ceil(d/4)
    recon_weight: float = 1.0
    log_range_threshold: float = 100.0   # signed-log1p on columns with max(|min|,|max|) > 100
    # ---- training
    batch_size: int = 1024
    eval_batch_size: int = 8192
    lr: float = 4e-4
    weight_decay: float = 1e-3
    max_epochs: int = 40
    sched_factor: float = 0.5
    sched_patience: int = 3
    min_lr: float = 1e-6
    grad_clip: float = 0.5
    es_patience: int = 6              # early stopping on the validation detection loss (best weights restored)
    val_frac: float = 0.10            # validation hold-out of the TRAINING side (inner split / each client's split)
    focal_gamma: float = 1.5
    # ---- federated learning
    fl_clients: int = 10
    fl_alpha: float = 0.5
    fl_rounds: int = 120              # maximum rounds; also the length of the cosine local-lr schedule
    fl_local_epochs: int = 1
    fl_min_client_size: int = 100
    fl_es_patience: int = 25           # rounds without improvement of the federated validation loss
    fl_min_rounds: int = 10
    fs_vote_frac: float = 0.5
    fs_min_client_samples: int = 1000
    fs_min_features: int = 5
    fl_iid_baseline: bool = True      # reference: same FedAvg on stratified IID clients (all features)
    fl_central_fs_ablation: bool = True   # reference: same FedAvg with central Boruta features
    # ---- federated optimisation: local SGD with the same number of steps on every client (McMahan et al. 2017;
    #      Wang et al. 2020) and server momentum FedAvgM (Hsu et al. 2019). Previous behaviour:
    #      fl_local_opt="adamw", fl_local_steps=0, fl_server_opt="avg"
    fl_local_opt: str = "sgd"         # "sgd" | "adamw"
    fl_local_lr: float = 0.1          # local SGD learning rate (AdamW uses lr)
    fl_local_steps: int = 10          # local mini-batch steps per client per round (0 = fl_local_epochs epochs)
    fl_server_opt: str = "avgm"       # "avgm" (FedAvgM) | "avg" (FedAvg)
    fl_server_lr: float = 1.0
    fl_server_momentum: float = 0.9
    fl_prox_mu: float = 0.0          # FedProx proximal term (Li et al. 2020); 0 = plain FedAvg/FedAvgM local objective
    # ---- non-IID stabilisation: cosine decay of the local lr over fl_rounds (FedAvg on non-IID data needs a decaying
    #      lr, Li et al. 2020 "On the convergence of FedAvg on non-IID data") and a server-side exponential moving
    #      average of the global weights (Polyak averaging), used for validation, early stopping and the returned model.
    #      Fold 1, non-IID, all features: MCC 0.800 -> 0.933. Previous behaviour: fl_lr_schedule="const", fl_ema=0
    fl_lr_schedule: str = "cos"       # "cos" | "const"
    fl_lr_min_frac: float = 0.05      # local lr in the last round = fl_lr_min_frac x initial local lr
    fl_ema: float = 0.9               # EMA decay of the server model (0 = off)
    # ---- decision rule and deployment
    decision_threshold: float = 0.5   # standard rule for every model: attack if sigmoid score >= 0.5
    # Recall-oriented threshold, reported next to the 0.5 rule: "far" lowers the threshold to the value with the highest
    # detection rate whose false-alarm rate on out-of-fold development scores stays <= target_far (never above
    # decision_threshold, never chosen on the test set); "fixed" keeps decision_threshold only. Post-hoc: it only
    # re-reads stored scores, so changing it does not invalidate the cache of a run being resumed.
    threshold_mode: str = "far"       # "far" | "fixed"
    target_far: float = 0.001         # maximum false-alarm rate (FPR) accepted when lowering the threshold
    # "auto": Fed-KAN-AE is deployed when it is non-inferior to the best centralized model on mean cross-validation MCC
    # (the centralized model exceeds it by at most deploy_margin_mcc), otherwise the best centralized model is deployed;
    # or an explicit key: mlp / kan / kan_ae / fl_kan_ae. The test set never influences this decision.
    deploy_model: str = "fl_kan_ae"
    deploy_margin_mcc: float = 0.01
    proposed_setting: str = "fs"      # "fs" (federated voting), "all", or "auto" (higher mean CV MCC)
    multiclass: bool = True           # second-stage softmax attack-type classifier
    # ---- cross-dataset generalization: two-phase fine-tuning of the source model on the target dataset
    #      (stratified 70/10/20 train/validation/test split of the target; phase 1 trains the classification head with
    #      the encoder frozen, phase 2 fine-tunes every layer with one tenth of the phase-1 learning rate; early stopping
    #      on the target validation loss, phase 1 with the same patience as the centralized models)
    transfer: bool = True
    transfer_val_frac: float = 0.10
    transfer_test_frac: float = 0.20
    transfer_lr_head: float = 1e-3
    transfer_epochs_head: int = 25
    transfer_patience_head: int = 6
    transfer_lr_full: float = 1e-4
    transfer_epochs_full: int = 150
    transfer_patience_full: int = 10
    # ---- interface
    ui: str = "auto"                  # "auto", "rich", "ansi" or "plain"
    max_upload_mb: int = 50
    max_rows: int = 200_000

    def __post_init__(self):
        """Rejects invalid categorical settings before any work is done (a run takes hours)."""
        allowed = {"deploy_model": ("auto", "mlp", "kan", "kan_ae", "fl_kan_ae"),
                   "proposed_setting": ("fs", "all", "auto"), "fl_local_opt": ("sgd", "adamw"),
                   "fl_server_opt": ("avgm", "avg"), "fl_lr_schedule": ("cos", "const"),
                   "text_encoding": ("ordinal", "onehot"), "threshold_mode": ("far", "fixed"),
                   "ui": ("auto", "rich", "ansi", "plain")}
        for k, ok in allowed.items():
            if getattr(self, k) not in ok:
                raise ValueError(f"{k}={getattr(self, k)!r} is not one of {ok}")
        if not 0 < self.val_frac < 0.5:
            raise ValueError("val_frac must be in (0, 0.5)")
        if not 0 <= self.target_far < 1:
            raise ValueError("target_far must be in [0, 1)")

    def fl_protocol(self):
        """The federated protocol actually configured (console header and model card use this single description)."""
        server = (f"FedAvgM (server momentum {self.fl_server_momentum:g}, server lr {self.fl_server_lr:g})"
                  if self.fl_server_opt == "avgm" else "FedAvg")
        work = (f"{self.fl_local_steps} local mini-batch steps" if self.fl_local_steps
                else f"{self.fl_local_epochs} local epoch(s)")
        lr = self.fl_local_lr if self.fl_local_opt == "sgd" else self.lr
        lr_txt = (f"lr {lr:g}, cosine decay to {self.fl_lr_min_frac * lr:g} over {self.fl_rounds} rounds"
                  if self.fl_lr_schedule == "cos" else f"lr {lr:g}")
        ema = f"server EMA of the global weights (decay {self.fl_ema:g}); " if self.fl_ema > 0 else ""
        return (f"{server}; {self.fl_clients} clients, non-IID Dirichlet alpha={self.fl_alpha:g}; {work} of "
                f"{self.fl_local_opt.upper()} ({lr_txt}) per round; {ema}class weights from the aggregated label counts; "
                f"federated min/max scaling; number of rounds chosen in cross-validation by early stopping on the "
                f"clients' local validation loss (patience {self.fl_es_patience}), final model retrained for it")

    @property
    def dirs(self):
        return {k: os.path.join(self.out_dir, k) for k in ("results", "models", "figures", "logs", "cache",
                                                             "detections")}

    def apply_quick(self):
        """Reduced budget for smoke tests (--quick and the mock run)."""
        self.max_epochs, self.boruta_sample, self.boruta_max_iter, self.fl_rounds = 8, 30_000, 30, 8
        self.target_sample = min(self.target_sample or 30_000, 30_000)
        self.transfer_epochs_head, self.transfer_epochs_full = 4, 8

    @classmethod
    def from_dict(cls, d):
        """Tolerant constructor for saved bundles: unknown keys (older/newer versions) are ignored."""
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def set_seed(seed):
    """Seeds Python, NumPy and PyTorch (CPU + CUDA) and makes cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Settings applied after training to stored scores only; the cache snapshot ignores them (see reporting.Cache).
POST_HOC_SETTINGS = ("threshold_mode", "target_far")


# ---- model registry and display names
MODEL_NAMES = {"mlp": "MLP (baseline)", "kan": "KAN", "kan_ae": "KAN-AE (hybrid)", "fl_kan_ae": "Fed-KAN-AE",
               "fl_iid_kan_ae": "Fed-KAN-AE (IID)", "fl_kan_ae_cfs": "Fed-KAN-AE (central FS)"}
PROPOSED = "fl_kan_ae"
PROPOSED_NAME = "Federated KAN-AE (Fed-KAN-AE)"
ORDER = ["mlp", "kan", "kan_ae", "fl_kan_ae", "fl_iid_kan_ae", "fl_kan_ae_cfs"]
CENTRAL_MODELS = ["mlp", "kan", "kan_ae"]
SET_NAMES = {"all": "ALL FEATURES", "fs": "BORUTA-SELECTED FEATURES"}
# Model pairs compared statistically (CV corrected t-test and test-set McNemar): (model A, set A, model B, set B)
COMPARISONS = ([(m, s, "mlp", s) for s in ("all", "fs") for m in ("kan", "kan_ae", "fl_kan_ae")] +
               [(m, "fs", m, "all") for m in ("mlp", "kan", "kan_ae", "fl_kan_ae")] +
               [("kan_ae", s, "kan", s) for s in ("all", "fs")] +
               [("fl_kan_ae", s, "kan_ae", s) for s in ("all", "fs")] +
               [("fl_iid_kan_ae", "all", "kan_ae", "all"), ("fl_kan_ae", "all", "fl_iid_kan_ae", "all"),
                ("fl_kan_ae", "fs", "fl_kan_ae_cfs", "fs")])


def long_name(key):
    return PROPOSED_NAME if key == PROPOSED else MODEL_NAMES[key]


def arch_of(key):
    """Network architecture behind a model key (every federated variant trains a KAN-AE)."""
    return "kan_ae" if key.startswith("fl_") else key
