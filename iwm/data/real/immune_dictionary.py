"""Immune Dictionary — Cui et al., Nature 2024.

17 immune cell types × 86 cytokines × multiple donors in vivo. The only
cytokine-conditioned scRNA atlas at this scale; essential for IWM's
cytokine-action pillar.

Broad Single Cell Portal accession: SCP2554.
DOI: 10.1038/s41586-023-06816-9.
License: Single Cell Portal data use agreement (non-commercial research).

Download (manual — requires free SCP account):
    https://singlecell.broadinstitute.org/single_cell/study/SCP2554/
    → place the expression + metadata files into data_cache/immune_dict/

The file layout downloaded from SCP:
    expression_matrix.mtx + features.tsv + barcodes.tsv
    metadata.tsv

The loader stitches these into an AnnData and exposes:
  * adata.obs["cytokine"]      — one of 86 cytokine names or "untreated"
  * adata.obs["cell_type"]     — one of 17 immune cell types
  * adata.obs["donor"]         — donor ID

For IWM, `cytokine` becomes the cytokine dose vector (one-hot over 86
channels, times dose — Cui et al. used fixed doses so one-hot is exact).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


CYTOKINE_COUNT = 86                   # matches Cui et al. 2024 design


def load_immune_dictionary(
    data_dir: str | Path = "data_cache/immune_dict",
    matrix_name: str = "expression_matrix.mtx",
    features_name: str = "features.tsv",
    barcodes_name: str = "barcodes.tsv",
    metadata_name: str = "metadata.tsv",
) -> "object":
    data_dir = Path(data_dir)
    import scanpy as sc
    import pandas as pd
    import scipy.io as sio
    import anndata as ad

    mtx_path = data_dir / matrix_name
    feat_path = data_dir / features_name
    bc_path = data_dir / barcodes_name
    meta_path = data_dir / metadata_name
    for p in (mtx_path, feat_path, bc_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. See module docstring for SCP2554 download hint."
            )

    X = sio.mmread(str(mtx_path)).T.tocsr()
    features = pd.read_csv(feat_path, sep="\t", header=None)
    barcodes = pd.read_csv(bc_path, sep="\t", header=None)
    metadata = pd.read_csv(meta_path, sep="\t", index_col=0)

    adata = ad.AnnData(X=X)
    adata.var_names = features.iloc[:, 0].astype(str).values
    adata.obs_names = barcodes.iloc[:, 0].astype(str).values
    adata.obs = metadata.loc[adata.obs_names]
    logger.info("Immune Dictionary: %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata


def cytokine_onehot(adata: "object", column: str = "cytokine") -> np.ndarray:
    """Build (N, 86) one-hot over the 86 cytokines + 'untreated'.
    Returns a float32 array; untreated rows are all-zero.
    """
    cyto = adata.obs[column].astype(str).values
    names = sorted(set(c for c in cyto if c.lower() not in ("untreated", "pbs", "control", "")))
    if len(names) > CYTOKINE_COUNT:
        raise ValueError(f"Found {len(names)} cytokines, expected ≤ {CYTOKINE_COUNT}.")
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros((adata.n_obs, CYTOKINE_COUNT), dtype=np.float32)
    for i, c in enumerate(cyto):
        if c in idx:
            out[i, idx[c]] = 1.0
    return out
