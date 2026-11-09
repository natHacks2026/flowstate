"""Hybrid physics + LSTM detector, and the training / scoring loops.

Inputs per step (6): FIT101, LIT101, MV101 open, P101 on, P102 on, and the
mass-balance residual (how far the level moved from what physics expected).

Two heads on top of the LSTM:
  * forecast head: next-step FIT101 and LIT101. LIT101 = physics prediction + learned
    correction (the "hybrid" part). Trained only on normal steps.
  * attack head: probability that a stage-1 attack is happening now. This is the score.

Fine-tuning freezes the 2-layer LSTM backbone and trains only the adapter + heads.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Detector(nn.Module):
    def __init__(self, n_features: int = 6, hidden: int = 64, adapter: int = 32, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.adapter = nn.LSTM(hidden, adapter, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(adapter)
        self.forecast_head = nn.Linear(adapter, 2)  # [next FIT101, correction to physics LIT101]
        self.attack_head = nn.Sequential(nn.Linear(adapter, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, phys_next: torch.Tensor):
        h, _ = self.backbone(x)
        h, _ = self.adapter(h)
        z = self.norm(h[:, -1])
        f = self.forecast_head(z)
        fit_next = f[:, 0]
        lit_next = phys_next + f[:, 1]
        logit = self.attack_head(z).squeeze(-1)
        return fit_next, lit_next, logit

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tensors(w: dict, dev: torch.device) -> list[torch.Tensor]:
    return [torch.as_tensor(w[k], device=dev) for k in ("x", "phys_next", "y_next", "label", "masked", "forecast_ok")]


def _loss(model, batch, use_cls: bool, pos_weight: float):
    x, phys, y, lab, msk, fok = batch
    fit_p, lit_p, logit = model(x, phys)
    lf = F.huber_loss(fit_p, y[:, 0], reduction="none") + F.huber_loss(lit_p, y[:, 1], reduction="none")
    lf = (lf * fok).sum() / fok.sum().clamp(min=1.0)
    if not use_cls:
        return lf
    keep = 1.0 - msk
    lc = F.binary_cross_entropy_with_logits(
        logit, lab, reduction="none", pos_weight=torch.tensor(pos_weight, device=logit.device))
    lc = (lc * keep).sum() / keep.sum().clamp(min=1.0)
    return lf + lc


def train(model: Detector, train_w: dict, val_w: dict, *, use_cls: bool, lr: float,
          max_epochs: int, patience: int = 3, batch_size: int = 256, seed: int = 0,
          tag: str = "") -> list[dict]:
    """Train with early stopping on validation loss. Returns per-epoch history."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    tr, va = _tensors(train_w, dev), _tensors(val_w, dev)
    n_pos = float(train_w["label"][train_w["masked"] == 0].sum())
    n_neg = float(((train_w["label"] == 0) & (train_w["masked"] == 0)).sum())
    use_cls = use_cls and n_pos > 0
    pos_weight = float(np.clip(n_neg / max(n_pos, 1.0), 1.0, 50.0))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best, best_state, bad, hist = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    n = len(tr[0])
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n, generator=g).to(dev)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            loss = _loss(model, [t[idx] for t in tr], use_cls, pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vl, m = 0.0, len(va[0])
            for i in range(0, m, 2048):
                vl += float(_loss(model, [t[i:i + 2048] for t in va], use_cls, pos_weight)) * min(2048, m - i)
        rec = dict(epoch=epoch, train_loss=tot / n, val_loss=vl / max(m, 1), seconds=round(time.time() - t0, 1))
        hist.append(rec)
        print(f"  {tag} epoch {epoch:2d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  ({rec['seconds']}s)", flush=True)
        if rec["val_loss"] < best - 1e-5:
            best, best_state, bad = rec["val_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def score(model: Detector, w: dict, batch_size: int = 2048) -> dict:
    """Attack probability per window, plus the LIT101 forecast error (z units)."""
    dev = next(model.parameters()).device
    model.eval()
    probs, lit_err = [], []
    for i in range(0, len(w["x"]), batch_size):
        x = torch.as_tensor(w["x"][i:i + batch_size], device=dev)
        ph = torch.as_tensor(w["phys_next"][i:i + batch_size], device=dev)
        _, lit_p, logit = model(x, ph)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        lit_err.append((lit_p.cpu().numpy() - w["y_next"][i:i + batch_size, 1]))
    return dict(prob=np.concatenate(probs), lit_err=np.concatenate(lit_err))
