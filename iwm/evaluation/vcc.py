"""Virtual Cell Challenge 2026 — evaluation hook.

The VCC (Arc Institute) asks: given a starting transcriptome and a
perturbation, predict the resulting expression shifts. The inaugural
challenge (2025) was scored on held-out cell types × single-gene CRISPRi.

Our hook wraps the task in three pieces:

    1. `VccBundle` — the standard VCC input, covering
       (control_state, perturbation_id, held_out_cell_type, observed_post_state).
    2. `score_vcc` — computes the three VCC primary metrics:
       * pearson_delta (the flagship)
       * deg_recall_at_30 (top-30 DE genes)
       * centroid_accuracy (Systema-style)
    3. `iwm_predict_vcc` — uses a trained IWM world model to emit the
       predicted post-state vector in the original gene space.

Data format (from the 2025 release — verify for 2026):
    AnnData `vcc_train.h5ad`, `vcc_val.h5ad`
      obs: perturbation (gene name), cell_type, is_control
      X:   normalised counts (log1p) OR raw counts depending on split

Once the 2026 dataset drops, plug the h5ad path into `load_vcc_bundle`
and call `score_vcc(preds, bundle)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .metrics import centroid_accuracy, deg_recall_at_k, pearson_delta


@dataclass
class VccBundle:
    """Standardised VCC eval inputs."""

    control_state: np.ndarray           # (N, G) mean-pooled control per cell_type
    perturbed_obs: np.ndarray           # (N, G) observed post-perturbation state
    perturbation_id: np.ndarray         # (N,) index into unique perturbation list
    cell_type_id: np.ndarray            # (N,)
    perturbation_names: list[str]
    cell_type_names: list[str]


def load_vcc_bundle(
    train_h5ad: str | Path,
    val_h5ad: str | Path,
    perturbation_key: str = "perturbation",
    cell_type_key: str = "cell_type",
    control_value: str = "control",
) -> VccBundle:
    import anndata as ad

    train = ad.read_h5ad(train_h5ad)
    val = ad.read_h5ad(val_h5ad)

    pert_col = val.obs[perturbation_key].astype(str).values
    ct_col = val.obs[cell_type_key].astype(str).values

    perts = sorted(set(pert_col) - {control_value})
    cts = sorted(set(ct_col))
    pert_to_id = {p: i for i, p in enumerate(perts)}
    ct_to_id = {c: i for i, c in enumerate(cts)}

    # Control state per cell type, computed from TRAIN.
    import numpy as np
    import scipy.sparse as sp

    def _to_dense(X):
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    ctrl_state = {}
    for c in cts:
        mask = (train.obs[perturbation_key] == control_value) & (train.obs[cell_type_key] == c)
        if mask.sum() == 0:
            # Fallback to global control mean
            mask = train.obs[perturbation_key] == control_value
        ctrl_state[c] = _to_dense(train.X[mask]).mean(axis=0)

    N = val.n_obs
    G = val.n_vars
    control_state = np.stack([ctrl_state[c] for c in ct_col], axis=0).astype(np.float32)
    perturbed_obs = _to_dense(val.X).astype(np.float32)
    perturbation_id = np.array([pert_to_id[p] for p in pert_col], dtype=np.int64)
    cell_type_id = np.array([ct_to_id[c] for c in ct_col], dtype=np.int64)

    return VccBundle(
        control_state=control_state,
        perturbed_obs=perturbed_obs,
        perturbation_id=perturbation_id,
        cell_type_id=cell_type_id,
        perturbation_names=perts,
        cell_type_names=cts,
    )


def score_vcc(preds: np.ndarray, bundle: VccBundle) -> Dict[str, float]:
    """Compute VCC metrics for a prediction matrix (N, G)."""

    delta_pred = preds - bundle.control_state
    delta_true = bundle.perturbed_obs - bundle.control_state

    # per-perturbation centroids for centroid_accuracy
    cent_true = {}
    for pid in np.unique(bundle.perturbation_id):
        mask = bundle.perturbation_id == pid
        cent_true[int(pid)] = bundle.perturbed_obs[mask].mean(axis=0)

    return {
        "pearson_delta": pearson_delta(delta_pred, delta_true),
        "deg_recall_at_30": deg_recall_at_k(delta_pred, delta_true, k=30),
        "centroid_accuracy": centroid_accuracy(preds, cent_true, bundle.perturbation_id),
    }
