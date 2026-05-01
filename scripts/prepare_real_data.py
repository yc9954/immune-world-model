"""Prepare a real-data subset for end-to-end testing.

Streams an immune-cell subset from CellxGene Census (commercial-clean,
CC-BY) and writes to `data_cache/`. This is the only data source IWM
can pull automatically — other real datasets (Schmidt 2022, Immune
Dictionary, CD4+ Dec 2025) require manual download per their licenses
and the module docstrings in iwm/data/real/.

Usage:
    python scripts/prepare_real_data.py --n-cells 20000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from iwm.data.real.cellxgene import stream_immune_subset


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=20_000)
    ap.add_argument("--tissue", default="blood",
                    help="CellxGene `tissue_general` filter (blood, lung, lymph node, ...)")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    adata = stream_immune_subset(
        n_cells=args.n_cells,
        cache_dir=args.cache_dir,
        tissue_general=args.tissue,
        force=args.force,
    )
    print(f"Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes from CellxGene Census")
    print(f"Cell types: {adata.obs['cell_type'].value_counts().head(10).to_dict()}")
    print(f"Diseases:   {adata.obs['disease'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
