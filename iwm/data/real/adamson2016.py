"""Adamson et al. 2016 — CRISPRi Perturb-seq in K562 cells.

Citation: Adamson et al., Cell 2016, doi:10.1016/j.cell.2016.11.048.
License: CC-BY (scPerturb collection, Zenodo 7041849).
Download: ~32 MB from scPerturb Zenodo.

CRISPRi knockdowns in K562 cells (UPR pathway screen). Effect sizes
are large (~1-5 std units in latent space vs ~0.0001 for disease labels),
making this ideal for IWM OOD-gene validation.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_ZENODO_URL = (
    "https://zenodo.org/api/records/7041849/files/"
    "AdamsonWeissman2016_GSM2406675_10X001.h5ad/content"
)
_CTRL_LABEL = "control"


def load_adamson2016(
    cache_dir: str | Path = "data_cache",
    max_cells: Optional[int] = None,
    seed: int = 0,
    force: bool = False,
) -> "object":
    """Return Adamson 2016 AnnData with `_perturbation` and `_cell_type` keys.

    Downloads ~32 MB on first call to `{cache_dir}/adamson2016.h5ad`.
    """
    import anndata as ad

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = cache_dir / "adamson2016.h5ad"

    if not h5ad_path.exists() or force:
        logger.info("Downloading Adamson 2016 (~32 MB) from Zenodo…")
        req = urllib.request.Request(
            _ZENODO_URL,
            headers={"User-Agent": "Mozilla/5.0 iwm-pipeline/1.0"},
        )
        with urllib.request.urlopen(req) as resp, open(h5ad_path, "wb") as out:
            chunk = 1 << 20
            total = 0
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                total += len(buf)
        logger.info("Saved %.1f MB to %s", total / 1e6, h5ad_path)

    adata = ad.read_h5ad(h5ad_path)

    # scPerturb stores perturbation in obs["perturbation"]
    pert_col = "perturbation"
    if pert_col not in adata.obs.columns:
        candidates = [c for c in adata.obs.columns if "pert" in c.lower()]
        pert_col = candidates[0] if candidates else adata.obs.columns[0]
        logger.warning("Perturbation column not found; using %s", pert_col)

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        logger.info("Subsampled to %d cells", adata.n_obs)

    # Normalise label: strip whitespace, lowercase
    adata.obs["_perturbation"] = adata.obs[pert_col].str.strip().str.lower()
    adata.obs["_cell_type"] = "K562"

    n_perts = adata.obs["_perturbation"].nunique()
    n_ctrl  = (adata.obs["_perturbation"] == _CTRL_LABEL).sum()
    logger.info("Adamson 2016: %d cells × %d genes  |  %d perturbations  |  %d ctrl",
                adata.n_obs, adata.n_vars, n_perts, n_ctrl)
    return adata
