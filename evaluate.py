"""Scoring: honest, attack-level numbers first; point-wise and point-adjusted F1 second.

All functions take per-step arrays from ONE continuous test stretch:
    score   anomaly score (higher = more suspicious)
    label   1 inside a stage-1 attack
    masked  1 inside an attack on another stage (ignored everywhere)
    attack_id  which stage-1 attack the step belongs to (0 = none)
"""
from __future__ import annotations

import numpy as np

GRACE_STEPS = 180  # alerts up to 15 min (180 x 5 s) after an attack ends still count as that attack


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.r_[0, mask.astype(int), 0]
    e = np.flatnonzero(np.diff(m))
    return list(zip(e[0::2], e[1::2]))


def pointwise(pred: np.ndarray, label: np.ndarray, masked: np.ndarray) -> dict:
    keep = masked == 0
    p, y = pred[keep].astype(bool), label[keep].astype(bool)
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = float((p == y).mean()) if len(y) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc)


def point_adjusted_f1(pred: np.ndarray, label: np.ndarray, masked: np.ndarray) -> float:
    """Common in SWaT papers but inflates results: one alert credits the whole attack.
    Reported for comparison only."""
    adj = pred.copy().astype(bool)
    for s, e in _runs(label == 1):
        if adj[s:e].any():
            adj[s:e] = True
    return pointwise(adj, label, masked)["f1"]


def events(pred: np.ndarray, label: np.ndarray, masked: np.ndarray, attack_id: np.ndarray,
           sample_seconds: int) -> dict:
    """Attack-level results: which attacks were caught, how fast, and false alarms per day."""
    pred = pred.astype(bool) & (masked == 0)
    per_attack = []
    covered = np.zeros(len(pred), dtype=bool)        # attack + grace period
    for aid in sorted(set(int(a) for a in np.unique(attack_id) if a > 0)):
        idx = np.flatnonzero(attack_id == aid)
        s, e = idx[0], idx[-1] + 1
        covered[s:min(len(pred), e + GRACE_STEPS)] = True
        hits = np.flatnonzero(pred[s:e])
        per_attack.append(dict(attack_id=aid, detected=bool(len(hits)),
                               delay_s=float(hits[0] * sample_seconds) if len(hits) else None,
                               duration_s=float((e - s) * sample_seconds)))
    false_alarms = sum(1 for s, e in _runs(pred) if not covered[s:e].any())
    normal_days = ((label == 0) & (masked == 0)).sum() * sample_seconds / 86400
    caught = [a for a in per_attack if a["detected"]]
    return dict(
        attacks_total=len(per_attack),
        attacks_detected=len(caught),
        mean_delay_s=float(np.mean([a["delay_s"] for a in caught])) if caught else None,
        false_alarms=false_alarms,
        false_alarms_per_day=float(false_alarms / normal_days) if normal_days > 0 else None,
        per_attack=per_attack,
    )


def pick_threshold(score: np.ndarray, label: np.ndarray, masked: np.ndarray,
                   attack_id: np.ndarray | None = None, sample_seconds: int = 5,
                   rule: str = "false_alarms", max_fa_per_day: float = 2.0) -> float:
    """Choose the alert threshold on VALIDATION data only (never on test).

    rule="false_alarms" (default): the lowest threshold whose false alarms on validation
        stay at or below max_fa_per_day. This is how an operator would set it.
    rule="f1": the threshold with the best point-wise F1 (common in papers, but often
        produces far too many false alarms to be usable).
    """
    keep = masked == 0
    cands = np.unique(np.quantile(score[keep], np.linspace(0.5, 0.99999, 500)))
    if rule == "f1":
        best_t, best_f1 = float(cands[-1]), -1.0
        for t in cands:
            f1 = pointwise(score >= t, label, masked)["f1"]
            if f1 > best_f1 + 1e-9:
                best_t, best_f1 = float(t), f1
        return best_t
    aid = attack_id if attack_id is not None else label.astype(int)
    for t in cands:  # ascending: the first that meets the budget catches the most
        fa = events(score >= t, label, masked, aid, sample_seconds)["false_alarms_per_day"]
        if fa is not None and fa <= max_fa_per_day:
            return float(t)
    return float(cands[-1])


def full_report(score, label, masked, attack_id, threshold, sample_seconds) -> dict:
    pred = score >= threshold
    pw = pointwise(pred, label, masked)
    ev = events(pred, label, masked, attack_id, sample_seconds)
    return dict(threshold=float(threshold), **ev,
                pointwise_precision=pw["precision"], pointwise_recall=pw["recall"],
                pointwise_f1=pw["f1"], pointwise_accuracy=pw["accuracy"], point_adjusted_f1=point_adjusted_f1(pred, label, masked))