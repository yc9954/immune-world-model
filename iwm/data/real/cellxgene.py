"""CellxGene Census loader — live data via cellxgene-census.

Docs: https://chanzuckerberg.github.io/cellxgene-census/
License: CC-BY 4.0 (commercial-clean).

Usage:
    from iwm.data.real.cellxgene import stream_immune_subset
    adata = stream_immune_subset(n_cells=50_000, cache_dir="data_cache")

The loader filters to immune tissues + specific cell types, then streams
the obs + X matrix into an AnnData. This is the main on-demand real-data
source used by the scVI pretrain and by benchmarks that do not require
perturbation annotations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


DEFAULT_IMMUNE_CELL_TYPES: tuple[str, ...] = (
    "CD4-positive, alpha-beta T cell",
    "CD8-positive, alpha-beta T cell",
    "naive thymus-derived CD4-positive, alpha-beta T cell",
    "effector memory CD4-positive, alpha-beta T cell",
    "regulatory T cell",
    "natural killer cell",
    "B cell",
    "memory B cell",
    "plasmablast",
    "plasmacytoid dendritic cell",
    "conventional dendritic cell",
    "classical monocyte",
    "non-classical monocyte",
    "macrophage",
)


def stream_immune_subset(
    n_cells: int = 50_000,
    cell_types: Optional[Sequence[str]] = None,
    cache_dir: str | Path = "data_cache",
    tissue_general: str = "blood",
    organism: str = "homo_sapiens",
    census_version: str = "stable",
    force: bool = False,
) -> "object":
    """Pull an immune-cell subset from CellxGene Census and return an AnnData.

    Cached to `{cache_dir}/cx_immune_{n}_{tissue}_{types_hash}.h5ad`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import hashlib

    types = tuple(cell_types or DEFAULT_IMMUNE_CELL_TYPES)
    key = hashlib.md5(("|".join(types) + f"|{tissue_general}|{n_cells}").encode()).hexdigest()[:8]
    out_path = cache_dir / f"cx_immune_{n_cells}_{tissue_general}_{key}.h5ad"

    if out_path.exists() and not force:
        logger.info("Cache hit: %s", out_path)
        import anndata as ad
        return ad.read_h5ad(out_path)

    import cellxgene_census
    import anndata as ad
    import numpy as np

    obs_col_names = ["cell_type", "disease", "tissue", "tissue_general",
                     "assay", "suspension_type", "donor_id"]
    obs_filter = (
        f"tissue_general == '{tissue_general}'"
        f" and cell_type in {list(types)}"
        " and is_primary_data == True"
    )

    logger.info("Fetching cell index from CellxGene Census (%s)…", census_version)
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Step 1: pull obs metadata only (no X) to get soma_joinids cheaply.
        obs_df = (
            census["census_data"][organism]
            .obs.read(
                value_filter=obs_filter,
                column_names=["soma_joinid"] + obs_col_names,
            )
            .concat()
            .to_pandas()
        )
        logger.info("Found %d matching cells; sampling %d", len(obs_df), min(n_cells, len(obs_df)))
        if len(obs_df) > n_cells:
            obs_df = obs_df.sample(n_cells, random_state=0)
        obs_coords = obs_df["soma_joinid"].values

        # Step 2: download X only for the sampled cells.
        logger.info("Downloading expression matrix for %d cells…", len(obs_coords))
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=organism,
            obs_coords=obs_coords,
            obs_column_names=obs_col_names,
        )

    adata.var_names_make_unique()
    logger.info("Writing %d × %d to %s", adata.n_obs, adata.n_vars, out_path)
    adata.write_h5ad(out_path)
    return adata
