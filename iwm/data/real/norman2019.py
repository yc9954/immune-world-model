"""Norman et al. 2019 — CRISPRa combinatorial Perturb-seq in K562 cells.

Citation: Norman et al., Science 2019, doi:10.1126/science.aax4438.
License: CC-BY (via pertpy built-in datasets).

Programmatic download via pertpy (~100 MB cache):
    import pertpy
    adata = pertpy.data.norman()

K562 is a CML cell line with hematopoietic origin — not primary T cells but
ideal for benchmarking because single-gene CRISPRa perturbations produce
large, well-characterised transcriptional shifts (~1-5 normalised units vs
~0.0001 for disease labels).

We use single-perturbation cells only (drop double combos) so the IWM
gene encoder has a clean 1-perturbation-per-cell setup, matching the
architecture assumption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_PERTURBATION_KEY = "perturbation"
_CTRL_LABEL = "control"


def load_norman2019(
    cache_dir: str | Path = "data_cache",
    single_only: bool = True,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> "object":
    """Return a Norman 2019 AnnData with `_perturbation` and `_cell_type` keys.

    Parameters
    ----------
    cache_dir:
        Directory where pertpy caches the download.
    single_only:
        Drop cells that received two simultaneous perturbations; keep single
        and control cells only.
    max_cells:
        If set, subsample down to this many cells after filtering.
    seed:
        Random state for subsampling.
    """
    import anndata as ad

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = cache_dir / "norman_2019.h5ad"

    if not h5ad_path.exists():
        import urllib.request
        url = "https://exampledata.scverse.org/pertpy/norman_2019.h5ad"
        logger.info("Downloading Norman 2019 (~100 MB) from %s …", url)
        urllib.request.urlretrieve(url, h5ad_path)
        logger.info("Saved to %s", h5ad_path)

    logger.info("Loading %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)

    # pertpy sets obs["perturbation"] as the gene target label.
    pert_col = _PERTURBATION_KEY
    if pert_col not in adata.obs.columns:
        # Fallback: scan for likely column names
        candidates = [c for c in adata.obs.columns if "pert" in c.lower()]
        if candidates:
            pert_col = candidates[0]
            logger.warning("Using %s as perturbation column (expected %s)", pert_col, _PERTURBATION_KEY)
        else:
            raise KeyError(f"No perturbation column found in {list(adata.obs.columns)}")

    # Single-perturbation filter (drop "+"-joined double combos).
    if single_only:
        is_double = adata.obs[pert_col].str.contains(r"\+", regex=True, na=False)
        n_before = adata.n_obs
        adata = adata[~is_double].copy()
        logger.info("Dropped %d double-perturbation cells (%d → %d)",
                    is_double.sum(), n_before, adata.n_obs)

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
        idx.sort()
        adata = adata[idx].copy()
        logger.info("Subsampled to %d cells", adata.n_obs)

    # Normalise perturbation label: strip whitespace, lowercase
    adata.obs["_perturbation"] = adata.obs[pert_col].str.strip().str.lower()
    adata.obs["_cell_type"] = "K562"       # single cell-type dataset

    n_perts = adata.obs["_perturbation"].nunique()
    n_ctrl = (adata.obs["_perturbation"] == _CTRL_LABEL).sum()
    logger.info("Norman 2019: %d cells × %d genes  |  %d perturbations  |  %d control cells",
                adata.n_obs, adata.n_vars, n_perts, n_ctrl)
    return adata
