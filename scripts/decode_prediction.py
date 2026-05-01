"""Decode IWM latent predictions back to gene expression space.

Usage:
    python scripts/decode_prediction.py \
        --adata data_cache/cx_immune_500_blood_3fee4a03.h5ad \
        --scvi-model results/scvi_disease/scvi_model \
        --iwm-ckpt results/real_disease_run/wm.pt \
        --iwm-sample results/scvi_disease/iwm_sample.npz \
        --out results/decoded

Outputs:
    decoded_normal.npy       (N_ctrl, n_genes) — control cell gene expression
    decoded_pred_lupus.npy   (N_lupus, n_genes) — IWM predicted lupus expression
    decoded_true_lupus.npy   (N_lupus, n_genes) — actual lupus expression
    top_deg.tsv              top differentially expressed genes IWM found
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata",      required=True)
    ap.add_argument("--scvi-model", required=True)
    ap.add_argument("--iwm-ckpt",   required=True)
    ap.add_argument("--iwm-sample", required=True)
    ap.add_argument("--out",        default="results/decoded")
    ap.add_argument("--top-k",      type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad
    import scvi
    from iwm.embeddings.scvi_wrapper import ScviEncoder
    from iwm.training.trainer import IWMLightningModule

    # Load scVI model
    logging.info("Loading scVI model from %s", args.scvi_model)
    import scanpy as sc
    adata = ad.read_h5ad(args.adata)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=10)   # must match pretrain_scvi.py
    enc = ScviEncoder.from_trained(args.scvi_model, adata=adata)

    # Load IWM
    logging.info("Loading IWM checkpoint from %s", args.iwm_ckpt)
    ckpt = torch.load(args.iwm_ckpt, map_location="cpu", weights_only=False)
    module = IWMLightningModule(ckpt["train_cfg"])
    module.load_state_dict(ckpt["state_dict"])
    wm = module.model.eval()

    # Load sample
    s = np.load(args.iwm_sample, allow_pickle=True)
    z_seq    = s["z_seq"].astype(np.float32)
    gene_feat = s["gene_feat"].astype(np.float32)
    drug     = s["drug_onehot"].astype(np.float32)
    cyto     = s["cyto_ext_seq"].astype(np.float32)
    gene_id  = s["gene_id"].astype(np.int64)
    neighbors = s["neighbors"].astype(np.int64)
    N = z_seq.shape[0]

    z_t    = z_seq[:, 0]
    z_true = z_seq[:, 1]

    # lupus = disease index 12
    mask_lupus  = gene_id == 12
    mask_normal = gene_id == 7
    logging.info("Normal: %d  Lupus: %d", mask_normal.sum(), mask_lupus.sum())

    # IWM prediction for lupus cells
    with torch.no_grad():
        zt   = torch.from_numpy(z_t[mask_lupus])
        gf   = torch.from_numpy(gene_feat[mask_lupus])
        dr   = torch.from_numpy(drug[mask_lupus])
        cy   = torch.from_numpy(cyto[mask_lupus, 0])
        neigh_idx = neighbors[mask_lupus] % N
        znbr = torch.from_numpy(z_t[neigh_idx])
        pred_z = wm(zt, gf, dr, cy, znbr)["mu"].numpy()

    # Decode to gene space via kNN retrieval (realistic LFC magnitudes)
    logging.info("Decoding latents to gene expression via kNN retrieval…")
    import scanpy as sc
    from scipy.sparse import issparse
    adata_knn = adata.copy()
    sc.pp.normalize_total(adata_knn, target_sum=1e4)
    sc.pp.log1p(adata_knn)
    X_log1p = adata_knn.X
    if issparse(X_log1p): X_log1p = X_log1p.toarray()
    Z_ref = enc.encode(adata)   # (M, d) latents for all reference cells

    expr_pred  = enc.knn_decode(pred_z,             Z_ref, X_log1p)
    expr_true  = enc.knn_decode(z_true[mask_lupus], Z_ref, X_log1p)
    expr_ctrl  = enc.knn_decode(z_t[mask_lupus],    Z_ref, X_log1p)

    np.save(out_dir / "decoded_ctrl.npy",       expr_ctrl.astype(np.float32))
    np.save(out_dir / "decoded_pred_lupus.npy", expr_pred.astype(np.float32))
    np.save(out_dir / "decoded_true_lupus.npy", expr_true.astype(np.float32))

    # Top DEGs: genes where |pred - ctrl| is largest
    gene_names = enc.gene_names
    delta_pred = (expr_pred - expr_ctrl).mean(axis=0)
    delta_true = (expr_true - expr_ctrl).mean(axis=0)

    top_idx = np.argsort(np.abs(delta_pred))[::-1][:args.top_k]

    logging.info("\nTop %d predicted DEGs (IWM, lupus vs control):", args.top_k)
    lines = ["gene\tdelta_pred\tdelta_true\tdirection"]
    for i in top_idx:
        direction = "UP" if delta_pred[i] > 0 else "DOWN"
        lines.append(f"{gene_names[i]}\t{delta_pred[i]:.4f}\t{delta_true[i]:.4f}\t{direction}")
        logging.info("  %-12s  pred=%+.3f  true=%+.3f  %s",
                     gene_names[i], delta_pred[i], delta_true[i], direction)

    (out_dir / "top_deg.tsv").write_text("\n".join(lines))

    # Pearson on gene-space delta
    from iwm.evaluation.metrics import pearson_delta
    pr = pearson_delta(delta_pred[None], delta_true[None])
    logging.info("\nGene-space Pearson(pred_delta, true_delta): %.3f", pr)
    logging.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
