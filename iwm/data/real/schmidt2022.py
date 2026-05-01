"""Schmidt et al. 2022 — CRISPRa/i Perturb-seq in primary human T cells.

Citation: Schmidt et al., Science 2022, doi:10.1126/science.abj4008.
GEO accession: GSE190604.
License: GEO data is typically CC-BY.

This loader assumes the user has already downloaded the raw counts to
`data_cache/schmidt2022/` as either 10x Genomics files or a pre-processed
`.h5ad`. The function returns an AnnData; an optional `perturbation_key`
maps the sgRNA annotation column to gene names.

Download hint (≈6 GB):
    mkdir -p data_cache/schmidt2022 && cd data_cache/schmidt2022
    curl -L -o GSE190604_RAW.tar \\
        'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE190604&format=file'
    tar xvf GSE190604_RAW.tar

The file naming within GSE190604 varies by sample (the paper contains
multiple CRISPRa and CRISPRi arms). Once extracted, point
`load_schmidt2022(h5ad_path=...)` at the merged AnnData of your choice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_schmidt2022(
    h5ad_path: str | Path = "data_cache/schmidt2022/schmidt_merged.h5ad",
    perturbation_key: str = "gene_target",
    cell_type_key: Optional[str] = "cell_type",
) -> "object":
    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See module docstring for GSE190604 download hint."
        )
    import anndata as ad
    adata = ad.read_h5ad(path)
    logger.info("Schmidt2022: %d cells × %d genes; perturbations=%d",
                adata.n_obs, adata.n_vars,
                adata.obs[perturbation_key].nunique() if perturbation_key in adata.obs else 0)
    # Add convenience keys to match IWM loader expectations
    if perturbation_key and perturbation_key in adata.obs:
        adata.obs["_perturbation"] = adata.obs[perturbation_key]
    if cell_type_key and cell_type_key in adata.obs:
        adata.obs["_cell_type"] = adata.obs[cell_type_key]
    return adata
