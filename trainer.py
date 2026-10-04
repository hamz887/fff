# -*- coding: utf-8 -*-
"""
trainer.py — losses, inference, the centralized trainer and the federated (FedAvg / FedAvgM) trainer.

Centralized protocol: AdamW + ReduceLROnPlateau + gradient clipping; binary focal loss with class weights ∝ 1/√n
(+ λ·MSE reconstruction for the KAN-AE, training objective only) or class-weighted cross-entropy for the softmax
attack-type head. Early stopping monitors the validation detection loss (the focal / cross-entropy term) on a hold-out
of the training side; the weights with the lowest validation loss are restored.
Federated protocol: every client splits off a local validation part; each round the clients run the same number of
local steps from the global model and send back only their weights; the server averages them weighted by client size
(FedAvg) and optionally applies server momentum (FedAvgM). Class weights come from the aggregated label counts, so the
federated objective Σ_i (n_i/N)·L_i equals the centralized one. On non-IID clients the local learning rate decays with a
cosine schedule over fl_rounds and the server keeps an exponential moving average (EMA) of the global weights, which
damps the round-to-round oscillation caused by single-class clients. Early stopping monitors the federated validation
loss of the EMA model, the size-weighted mean of the clients' local validation losses (validation data never leave a
client).
"""
import copy
import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import DEVICE, set_seed
from .data_loader import holdout_split
from .reporting import TrainProgress


# ---------------------------------------------------------------------------------------------------------------
# Losses and class weights
# ---------------------------------------------------------------------------------------------------------------
def class_weights(y, n_classes=2):
    """w_c ∝ 1/sqrt(n_c) over the classes present in y, normalised to mean 1 and clipped to [0.1, 10]. A class absent
    from y never enters the loss and gets weight 1, so it cannot distort the weights of the present classes."""
    c = np.bincount(y, minlength=n_classes).astype(np.float64)
    w, k = np.ones(n_classes), c > 0
    w[k] = np.clip(c[k] ** -0.5 / (c[k] ** -0.5).mean(), 0.1, 10.0)
    w[k] /= w[k].mean()
    return torch.tensor(w, dtype=torch.float32)


class BinaryFocalLoss(nn.Module):
    """FL(p_t) = -α_t (1-p_t)^γ log p_t (Lin et al. 2017) with α_t = class weight."""

    def __init__(self, gamma=1.5, w=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("w", w if w is not None else torch.ones(2))

    def forward(self, logit, y):
        y = y.float()
        bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        a = self.w[1] * y + self.w[0] * (1 - y)
        return (a * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def make_loss(cfg, y, n_classes=None):
    """n_classes=None -> binary focal loss; otherwise class-weighted cross-entropy. Weights from the labels y."""
    if n_classes:
        return nn.CrossEntropyLoss(weight=class_weights(y, n_classes).to(DEVICE))
    return BinaryFocalLoss(cfg.focal_gamma, class_weights(y).to(DEVICE)).to(DEVICE)


def to_tensor(X, dtype=None):
    """NumPy -> tensor on DEVICE, done once per fit instead of once per mini-batch."""
    return torch.as_tensor(np.ascontiguousarray(X), dtype=dtype, device=DEVICE)


# ---------------------------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_scores(model, X, bs=8192, multiclass=False):
    """Attack probability (sigmoid) or class probabilities (softmax) for a float32 matrix X."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        z, _ = model(torch.from_numpy(np.ascontiguousarray(X[i:i + bs], dtype=np.float32)).to(DEVICE))
        out.append((torch.softmax(z, 1) if multiclass else torch.sigmoid(z)).cpu().numpy())
    if out:
        return np.concatenate(out)
    return np.zeros((0, getattr(model, "n_out", 1))) if multiclass else np.zeros((0,), np.float32)


def _train_mode(model):
    """model.train(), but modules whose parameters are all frozen (e.g. a frozen feature extractor during head-only
    fine-tuning) stay in eval mode so BatchNorm statistics are not updated."""
    model.train()
    for m in model.modules():
        ps = list(m.parameters(recurse=False))
        if ps and not any(p.requires_grad for p in ps):
            m.eval()


# ---------------------------------------------------------------------------------------------------------------
# Centralized trainer
# ---------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, R=None):
        self.cfg, self.R = cfg, R

    def _epoch(self, model, opt, X, y, loss_fn, rng, on_batch=None):
        """One pass over (X, y) (tensors on DEVICE) in a seeded random order; returns the mean training loss."""
        cfg = self.cfg
        _train_mode(model)
        N = len(X)
        perm = torch.from_numpy(rng.permutation(N)).to(DEVICE)
        tot, n = 0.0, 0
        nb = int(np.ceil(N / cfg.batch_size))
        for k, i in enumerate(range(0, N, cfg.batch_size), 1):
            b = perm[i:i + cfg.batch_size]
            if len(b) < 2:                                   # BatchNorm needs >= 2 samples
                continue
            xb, yb = X[b], y[b]
            logit, rec = model(xb)
            loss = loss_fn(logit, yb)
            if rec is not None:
                loss = loss + cfg.recon_weight * F.mse_loss(rec, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
            if on_batch is not None:
                on_batch(k, nb, tot / n)
        return tot / max(n, 1)

    @torch.no_grad()
    def _val(self, model, parts, loss_fn, n_classes=None):
        """Validation detection loss and F1 over one or more (X, y) tensor pairs: a single hold-out, or the clients'
        local validation parts (the loss is then the size-weighted mean of the client losses and F1 follows from the
        summed confusion counts). F1 at the decision threshold (binary) or macro-F1 (attack types). The KAN-AE decoder
        is not run: the reconstruction term belongs to the training objective only."""
        model.eval()
        bs, tot, n, ys, ps = self.cfg.eval_batch_size, 0.0, 0, [], []
        for X, y in parts:
            for i in range(0, len(X), bs):
                xb, yb = X[i:i + bs], y[i:i + bs]
                logit, _ = model(xb)
                tot += loss_fn(logit, yb).item() * len(xb)
                ps.append(logit.argmax(1) if n_classes
                          else (torch.sigmoid(logit) >= self.cfg.decision_threshold).long())
                ys.append(yb)
            n += len(X)
        yv, pv = torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()
        return tot / max(n, 1), f1_score(yv, pv, average="macro" if n_classes else "binary", zero_division=0)

    def _optim(self, model, lr=None):
        cfg = self.cfg
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr or cfg.lr,
                                weight_decay=cfg.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=cfg.sched_factor,
                                                         patience=cfg.sched_patience, min_lr=cfg.min_lr)
        return opt, sch

    def fit(self, model, Xtr, ytr, Xva, yva, seed, label="", n_classes=None, lr=None, max_epochs=None):
        """Trains on (Xtr, ytr) with early stopping on the detection loss of the hold-out (Xva, yva); returns (model
        with the lowest-validation-loss weights, history, info). n_classes=None -> binary detector, else softmax head."""
        cfg = self.cfg
        set_seed(seed)
        model.to(DEVICE)
        rng = np.random.RandomState(seed)
        ytr, yva = ytr.astype(np.int64), yva.astype(np.int64)
        loss_fn = make_loss(cfg, ytr, n_classes)
        opt, sch = self._optim(model, lr)
        Xtr_t, ytr_t = to_tensor(Xtr, torch.float32), to_tensor(ytr)
        val = [(to_tensor(Xva, torch.float32), to_tensor(yva))]
        best_loss, best_ep, best, wait = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
        hist, stop, t0, ep = [], "max epochs reached", time.time(), 0
        E = max_epochs or cfg.max_epochs
        with TrainProgress(self.R, label, E, "epoch") as pb:
            for ep in range(1, E + 1):
                te, lr_now = time.time(), opt.param_groups[0]["lr"]
                pb.start(ep, int(np.ceil(len(Xtr) / cfg.batch_size)))
                tl = self._epoch(model, opt, Xtr_t, ytr_t, loss_fn, rng,
                                 on_batch=lambda k, nb, l: pb.step(k, f"loss {l:.5f}"))
                vl, vf1 = self._val(model, val, loss_fn, n_classes)
                sch.step(vl)
                imp = vl < best_loss - 1e-6
                if imp:
                    best_loss, best_ep, best, wait = vl, ep, copy.deepcopy(model.state_dict()), 0
                else:
                    wait += 1
                hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "val_f1": vf1, "lr": lr_now,
                             "time_s": time.time() - te, "best": imp})
                pb.end(f"train {tl:.5f} · val {vl:.5f} · F1 {vf1:.4f}{' ★' if imp else ''}")
                if wait >= cfg.es_patience:
                    stop = f"validation loss did not improve for {cfg.es_patience} epochs"
                    break
        model.load_state_dict(best)
        return model, hist, {"epochs_run": ep, "best_epoch": best_ep, "best_val_loss": best_loss, "stop_reason": stop,
                             "time_s": time.time() - t0}

    def fit_fixed(self, model, X, y, seed, epochs, label=""):
        """Final retraining on the full development set for the CV-selected number of epochs (no validation data
        exists at this point, so the LR scheduler follows the training loss)."""
        cfg = self.cfg
        set_seed(seed)
        model.to(DEVICE)
        rng = np.random.RandomState(seed)
        y = y.astype(np.int64)
        loss_fn = make_loss(cfg, y)
        opt, sch = self._optim(model)
        Xt, yt = to_tensor(X, torch.float32), to_tensor(y)
        hist, t0 = [], time.time()
        with TrainProgress(self.R, label, epochs, "epoch") as pb:
            for ep in range(1, epochs + 1):
                te, lr_now = time.time(), opt.param_groups[0]["lr"]
                pb.start(ep, int(np.ceil(len(X) / cfg.batch_size)))
                tl = self._epoch(model, opt, Xt, yt, loss_fn, rng, on_batch=lambda k, nb, l: pb.step(k, f"loss {l:.5f}"))
                sch.step(tl)
                hist.append({"epoch": ep, "train_loss": tl, "lr": lr_now, "time_s": time.time() - te})
                pb.end(f"train {tl:.5f}")
        return model, hist, {"epochs_run": epochs, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------------------------------------------
# Federated trainer (FedAvg / FedAvgM)
# ---------------------------------------------------------------------------------------------------------------
class FederatedTrainer(Trainer):
    """FedAvg (McMahan et al. 2017) with optional server momentum (FedAvgM, Hsu et al. 2019); protocol in the module
    docstring. Non-IID clients come from a Dirichlet(alpha) label split (Hsu et al. 2019), IID clients from a stratified
    split."""

    @staticmethod
    def dirichlet_partition(y, n, alpha, seed, min_size):
        rng, N = np.random.RandomState(seed), len(y)
        for _ in range(100):
            parts = [[] for _ in range(n)]
            for k in np.unique(y):
                idx = np.where(y == k)[0]
                rng.shuffle(idx)
                p = rng.dirichlet(np.repeat(alpha, n))
                p = np.array([pj * (len(parts[j]) < N / n) for j, pj in enumerate(p)])
                p = p / p.sum()
                for j, ch in enumerate(np.split(idx, (np.cumsum(p) * len(idx)).astype(int)[:-1])):
                    parts[j].extend(ch.tolist())
            if min(len(pj) for pj in parts) >= min_size:
                break
        else:
            warnings.warn("minimum client size not reached; last Dirichlet partition used")
        return [np.sort(np.array(pj, dtype=np.int64)) for pj in parts]

    @staticmethod
    def iid_partition(y, n, seed):
        return [np.sort(te) for _, te in StratifiedKFold(n, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y)]

    def _local_lr(self, r=None):
        """Local learning rate of round r: constant, or cosine decay from the initial value to fl_lr_min_frac of it
        over fl_rounds (the same schedule in cross-validation and in the final retraining)."""
        cfg = self.cfg
        lr = cfg.fl_local_lr if cfg.fl_local_opt == "sgd" else cfg.lr
        if cfg.fl_lr_schedule == "cos" and r is not None:
            lr *= cfg.fl_lr_min_frac + (1 - cfg.fl_lr_min_frac) * 0.5 * (1 + math.cos(math.pi * (r - 1) / cfg.fl_rounds))
        return lr

    def _reset_fl_state(self, model):
        """Server momentum buffer and EMA model, reset at the start of every federated training run."""
        self._mom = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self._ema = None

    def _ema_step(self, model):
        """Server-side EMA of the global weights after a round (Polyak averaging); returns the model to validate and
        keep (the EMA model, or the global model itself when fl_ema = 0)."""
        b = self.cfg.fl_ema
        if b <= 0:
            return model
        if self._ema is None:
            self._ema = copy.deepcopy(model)
        else:
            with torch.no_grad():
                for pe, p in zip(self._ema.parameters(), model.parameters()):
                    pe.mul_(b).add_(p.detach(), alpha=1 - b)
        return self._ema

    def _local_batches(self, n, rng):
        """Local work of one client: fl_local_steps mini-batches (same for every client, reshuffled cyclically), or
        fl_local_epochs full epochs when fl_local_steps = 0."""
        cfg = self.cfg
        if cfg.fl_local_steps > 0:
            reps = int(np.ceil(cfg.fl_local_steps * cfg.batch_size / n)) + 1
            order = np.concatenate([rng.permutation(n) for _ in range(reps)])
            return [order[i * cfg.batch_size:(i + 1) * cfg.batch_size] for i in range(cfg.fl_local_steps)]
        out = []
        for _ in range(cfg.fl_local_epochs):
            perm = rng.permutation(n)
            out += [perm[i:i + cfg.batch_size] for i in range(0, n, cfg.batch_size)]
        return out

    def _local_update(self, local, client, loss_fn, rng, on_batch=None, lr=None):
        """Local training of one client with SGD (or AdamW) at learning rate lr; returns the mean training loss."""
        cfg = self.cfg
        lr = lr or self._local_lr()
        Xc, yc = client
        params = [p for p in local.parameters() if p.requires_grad]
        anchor = [p.detach().clone() for p in params]       # global model received this round (FedProx anchor)
        opt = torch.optim.SGD(params, lr=lr) if cfg.fl_local_opt == "sgd" else \
            torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        _train_mode(local)
        batches = self._local_batches(len(yc), rng)
        tot, seen = 0.0, 0
        for k, b in enumerate(batches, 1):
            if len(b) < 2:                                   # BatchNorm needs >= 2 samples
                continue
            b = torch.from_numpy(b).to(DEVICE)
            xb, yb = Xc[b], yc[b]
            logit, rec = local(xb)
            loss = loss_fn(logit, yb)
            if rec is not None:
                loss = loss + cfg.recon_weight * F.mse_loss(rec, xb)
            if cfg.fl_prox_mu > 0:  # FedProx: keep the client close to the global model
                loss = loss + 0.5 * cfg.fl_prox_mu * sum(((p - a) ** 2).sum() for p, a in zip(params, anchor))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            tot, seen = tot + loss.item() * len(b), seen + len(b)
            if on_batch is not None:
                on_batch(k, len(batches), tot / seen)
        return tot / max(seen, 1)

    def _round(self, model, clients, loss_fn, rng, pb=None, lr=None):
        """One round over device-resident clients [(X, y)]: local training, size-weighted average of the client models
        (FedAvg), then the server step. FedAvgM applies momentum to the averaged update
        Δ = Σ_i p_i w_i − w:  v ← βv + Δ,  w ← w + η_s·v."""
        cfg = self.cfg
        nbs = [len(self._local_batches(len(c[1]), np.random.RandomState(0))) for c in clients]
        if pb is not None:
            pb.n_inner = max(sum(nbs), 1)
        frac = np.array([len(c[1]) for c in clients], float)
        frac = frac / frac.sum()
        g = copy.deepcopy(model.state_dict())
        avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in g.items()}
        loss, base = 0.0, 0
        for ci, (client, f) in enumerate(zip(clients, frac)):
            local = copy.deepcopy(model)
            local.load_state_dict(g)
            cb = (lambda k, nb, l, off=base, ci=ci: pb.step(off + k, f"client {ci + 1}/{len(clients)}")) \
                if pb is not None else None
            loss += f * self._local_update(local, client, loss_fn, rng, cb, lr)
            base += nbs[ci]
            for k, v in local.state_dict().items():
                avg[k] += f * v.float()
        new = {k: v.to(g[k].dtype) for k, v in avg.items()}
        if cfg.fl_server_opt == "avgm":
            for n in self._mom:
                self._mom[n] = cfg.fl_server_momentum * self._mom[n] + (avg[n] - g[n].float())
                new[n] = (g[n].float() + cfg.fl_server_lr * self._mom[n]).to(g[n].dtype)
        model.load_state_dict(new)
        return loss

    def fit(self, model, clients, seed, label="", n_classes=None):
        """Federated training with early stopping on the federated validation loss. ``clients`` = [(X, y)] NumPy
        arrays; each client holds out val_frac of its rows (stratified by label) for local validation; a client too
        small to hold out any trains on all its rows. n_classes=None -> binary; otherwise softmax over n_classes."""
        cfg = self.cfg
        set_seed(seed)
        model.to(DEVICE)
        rng = np.random.RandomState(seed)
        clients = [(Xc, yc.astype(np.int64)) for Xc, yc in clients if len(yc)]
        splits = [holdout_split(yc, cfg.val_frac, seed + i) for i, (_, yc) in enumerate(clients)]
        train = [(to_tensor(Xc[f], torch.float32), to_tensor(yc[f])) for (Xc, yc), (f, _) in zip(clients, splits)]
        val = [(to_tensor(Xc[v], torch.float32), to_tensor(yc[v]))
               for (Xc, yc), (_, v) in zip(clients, splits) if len(v)]
        if not val:
            raise RuntimeError("no client is large enough to hold out a local validation part")
        # class weights from the label counts summed over the clients (a sum obtainable by secure aggregation)
        loss_fn = make_loss(cfg, np.concatenate([yc[f] for (_, yc), (f, _) in zip(clients, splits)]), n_classes)
        self._reset_fl_state(model)
        best_loss, best, wait, best_r = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
        hist, stop, t0, r = [], "max rounds reached", time.time(), 0
        with TrainProgress(self.R, label, cfg.fl_rounds, "round", "client batches") as pb:
            for r in range(1, cfg.fl_rounds + 1):
                tr0 = time.time()
                pb.start(r, 1)
                tl = self._round(model, train, loss_fn, rng, pb, self._local_lr(r))
                ev = self._ema_step(model)                   # EMA model (or the global model when fl_ema = 0)
                vl, vf1 = self._val(ev, val, loss_fn, n_classes)
                # warm-up: the server momentum builds up over ~1/(1-β) rounds, so rounds before fl_min_rounds are
                # neither eligible as the best round nor counted towards the patience
                eligible = r >= min(cfg.fl_min_rounds, cfg.fl_rounds)
                imp = eligible and vl < best_loss - 1e-6
                if imp:
                    best_loss, best, wait, best_r = vl, copy.deepcopy(ev.state_dict()), 0, r
                elif eligible:
                    wait += 1
                hist.append({"epoch": r, "train_loss": tl, "val_loss": vl, "val_f1": vf1, "lr": self._local_lr(r),
                             "time_s": time.time() - tr0, "best": imp})
                pb.end(f"local {tl:.5f} · val {vl:.5f} · F1 {vf1:.4f}{' ★' if imp else ''}")
                if wait >= cfg.fl_es_patience:
                    stop = f"federated validation loss did not improve for {cfg.fl_es_patience} rounds"
                    break
        model.load_state_dict(best)
        return model, hist, {"epochs_run": r, "best_epoch": best_r, "best_val_loss": best_loss, "stop_reason": stop,
                             "time_s": time.time() - t0}

    def fit_fixed(self, model, clients, seed, rounds, label=""):
        """Final federated retraining on all client rows for the CV-selected number of rounds."""
        set_seed(seed)
        model.to(DEVICE)
        rng, hist, t0 = np.random.RandomState(seed), [], time.time()
        clients = [(Xc, yc.astype(np.int64)) for Xc, yc in clients if len(yc)]
        loss_fn = make_loss(self.cfg, np.concatenate([yc for _, yc in clients]))
        train = [(to_tensor(Xc, torch.float32), to_tensor(yc)) for Xc, yc in clients]
        self._reset_fl_state(model)
        ev = model
        with TrainProgress(self.R, label, rounds, "round", "client batches") as pb:
            for r in range(1, rounds + 1):
                tr0 = time.time()
                pb.start(r, 1)
                tl = self._round(model, train, loss_fn, rng, pb, self._local_lr(r))
                ev = self._ema_step(model)
                hist.append({"epoch": r, "train_loss": tl, "lr": self._local_lr(r), "time_s": time.time() - tr0})
                pb.end(f"local {tl:.5f}")
        if ev is not model:                                  # the EMA model is the trained model (as in CV)
            model.load_state_dict(ev.state_dict())
        return model, hist, {"epochs_run": rounds, "time_s": time.time() - t0}
