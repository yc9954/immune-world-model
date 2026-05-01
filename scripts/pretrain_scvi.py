"""Pretrain an scVI cell-state encoder on an AnnData.

Inputs:
  --adata   path to a .h5ad file (produced by prepare_real_data.py or
            a downloaded real dataset)
  --out     directory to save the scVI model
  --latent  latent dim (default 64)
  --epochs  training epochs (default 20; scale with dataset size)

Outputs:
  out/scvi_model/          scvi-tools model directory (reusable via
                           scvi.model.SCVI.load)
  out/latent.npy           (N, latent_dim) — latent matrix for the input
  out/iwm_sample.npz       IWM-compatible sample dict, ready to drop into
                           the eval or training pipeline

This is the commercial-clean L1 path (scVI is BSD-3). scvi-tools supports
Apple MPS as of 1.1+; --accelerator=auto picks the best available.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np


def _install_hint_if_missing() -> None:
    try:
        import scvi  # noqa: F401
        import scanpy  # noqa: F401
        import anndata  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scvi-tools / scanpy / anndata are optional deps. Install with "
            "`pip install scvi-tools scanpy anndata`."
        ) from e


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--out", default="results/scvi_run")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-key", default=None,
                    help="obs column for batch correction (e.g. donor_id)")
    ap.add_argument("--perturbation-key", default=None,
                    help="obs column to use as perturbation label (e.g. disease)")
    ap.add_argument("--accelerator", default="auto")
    args = ap.parse_args()

    _install_hint_if_missing()
    import anndata as ad
    import scanpy as sc
    import scvi

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading %s", args.adata)
    adata = ad.read_h5ad(args.adata)
    sc.pp.filter_genes(adata, min_cells=10)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=args.batch_key,
    )
    model = scvi.model.SCVI(
        adata,
        n_latent=args.latent,
        n_layers=2,
        gene_likelihood="nb",
    )
    logging.info("Training scVI for %d epochs", args.epochs)
    model.train(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        enable_progress_bar=True,
    )

    model_dir = out_dir / "scvi_model"
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    Z = model.get_latent_representation(adata)
    np.save(out_dir / "latent.npy", Z)

    # Build an IWM sample dict — single-step "control → observed" trajectory.
    from iwm.data.real.base import adata_to_iwm_sample
    adata.obsm["X_scVI"] = Z
    bundle = adata_to_iwm_sample(
        adata,
        latent_key="X_scVI",
        perturbation_key=args.perturbation_key,
        cell_type_key="cell_type",
        scvi_model=model,
    )
    np.savez_compressed(out_dir / "iwm_sample.npz", **bundle.as_sample_dict())
    logging.info("Saved scVI model to %s, latent to latent.npy, sample to iwm_sample.npz",
                 model_dir)


if __name__ == "__main__":
    main()
