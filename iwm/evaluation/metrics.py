"""Systema-style metrics.

Pearson on delta is the primary — Pearson on raw is forbidden (systematic
variation leaks into it; see Systema, Nat Biotech 2025).
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def pearson_delta(delta_pred: np.ndarray, delta_true: np.ndarray) -> float:
    """Pearson correlation computed over the flattened (N*d) deltas.

    Returns 0.0 if either vector is constant (undefined correlation).
    """
    p = delta_pred.reshape(-1).astype(np.float64)
    t = delta_true.reshape(-1).astype(np.float64)
    if p.std() < 1e-8 or t.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def centroid_accuracy(
    preds: np.ndarray,                     # (N, d)
    true_by_perturbation: Dict[int, np.ndarray],   # p -> (d,)
    perturbation_ids: np.ndarray,          # (N,) true id per row
) -> float:
    """For each row, which perturbation centroid is closest?"""
    ids = sorted(true_by_perturbation.keys())
    centroids = np.stack([true_by_perturbation[p] for p in ids], axis=0)   # (P, d)
    dists = np.linalg.norm(preds[:, None, :] - centroids[None, :, :], axis=-1)   # (N, P)
    pred_ids = np.array(ids)[dists.argmin(axis=1)]
    correct = (pred_ids == perturbation_ids).mean()
    return float(correct)


def deg_recall_at_k(
    delta_pred: np.ndarray,           # (N, d)
    delta_true: np.ndarray,           # (N, d)
    k: int = 30,
) -> float:
    """Mean-level differential-'expression' recall. In synthetic d-space
    the 'genes' are latent dims; in real runs swap delta_pred/true for
    reconstructed gene-space deltas."""
    mean_pred = delta_pred.mean(axis=0)
    mean_true = delta_true.mean(axis=0)
    k = min(k, mean_pred.size)
    top_pred = set(np.argsort(-np.abs(mean_pred))[:k].tolist())
    top_true = set(np.argsort(-np.abs(mean_true))[:k].tolist())
    return float(len(top_pred & top_true) / k)


def rollout_pearson(
    traj_pred: np.ndarray,          # (N, T, d)
    traj_true: np.ndarray,
    skip_step0: bool = True,
) -> float:
    if skip_step0:
        traj_pred = traj_pred[:, 1:]
        traj_true = traj_true[:, 1:]
    if traj_pred.size == 0:
        return 0.0
    return pearson_delta(traj_pred, traj_true)
