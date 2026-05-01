"""Genome-scale CD4+ T-cell Perturb-seq (bioRxiv Dec 2025).

Citation: bioRxiv 2025.12.23.696273 — "Genome-scale perturb-seq in
primary human CD4+ T cells maps context-specific regulators of T cell
programs and human immune traits."
License: bioRxiv preprint (typically CC-BY).

This dataset extends Schmidt 2022 to genome-wide scale. The authors
distribute processed data via a public S3 bucket (URL announced with the
final paper release) — until then users must obtain the raw counts from
the lab's Figshare or Zenodo deposit listed in the preprint.

Once an AnnData is in hand at `data_cache/cd4_perturb_2025.h5ad`, this
loader returns it with IWM-compatible metadata keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_cd4_perturb_seq_2025(
    h5ad_path: str | Path = "data_cache/cd4_perturb_2025.h5ad",
    perturbation_key: str = "gene_target",
    cell_type_key: Optional[str] = "cd4_subtype",
) -> "object":
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1 "
            "for data distribution links as they come online."
        )
    import anndata as ad
    adata = ad.read_h5ad(path)
    logger.info("CD4+ Perturb-seq 2025: %d cells × %d genes", adata.n_obs, adata.n_vars)
    if perturbation_key in adata.obs:
        adata.obs["_perturbation"] = adata.obs[perturbation_key]
    if cell_type_key and cell_type_key in adata.obs:
        adata.obs["_cell_type"] = adata.obs[cell_type_key]
    return adata
