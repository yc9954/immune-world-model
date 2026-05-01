"""End-to-end Perturb-seq validation pipeline for IWM.

Downloads Norman 2019, trains scVI, trains IWM, evaluates OOD-gene prediction
vs Ridge baseline.

Usage (single command, no manual downloads):
    python scripts/run_perturb_seq_pipeline.py

Options:
    --out          results/perturb_seq_run   output directory
    --n-cells      5000                      how many cells to use (set larger if S3 is stable)
    --latent       32                        scVI latent dim
    --scvi-epochs  15                        scVI training epochs
    --iwm-epochs   30                        IWM training epochs
    --ood-frac     0.15                      fraction of perturbations held out for OOD test
    --seed         0

Outputs:
    out/scvi_model/         trained scVI model
    out/latent.npy          (N, latent_dim)
    out/iwm_sample.npz      IWM-compatible sample bundle
    out/wm.pt               trained IWM checkpoint
    out/eval_results.json   OOD-gene Pearson (IWM vs Ridge vs Mean-shift)
    out/top_ood_degs.tsv    top predicted DEGs for OOD perturbations
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _check_deps() -> None:
    missing = []
    for pkg in ["scvi", "scanpy", "anndata"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing dependencies: {missing}\n"
            "Install with: pip install scvi-tools scanpy anndata"
        )


def _train_scvi(adata, out_dir: Path, latent: int, epochs: int, accelerator: str):
    import scvi
    import scanpy as sc

    sc.pp.filter_genes(adata, min_cells=10)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    model = scvi.model.SCVI(adata, n_latent=latent, n_layers=2, gene_likelihood="nb")
    log.info("Training scVI for %d epochs (latent=%d)…", epochs, latent)
    model.train(max_epochs=epochs, accelerator=accelerator, enable_progress_bar=True)

    model_dir = out_dir / "scvi_model"
    model.save(str(model_dir), overwrite=True, save_anndata=False)
    Z = model.get_latent_representation(adata)
    np.save(out_dir / "latent.npy", Z)
    log.info("scVI done — latent shape: %s", Z.shape)
    return Z, model


def _build_sample(adata, Z: np.ndarray, scvi_model, out_dir: Path) -> dict:
    from iwm.data.real.base import adata_to_iwm_sample

    adata.obsm["X_scVI"] = Z
    bundle = adata_to_iwm_sample(
        adata,
        latent_key="X_scVI",
        perturbation_key="_perturbation",
        cell_type_key="_cell_type",
        scvi_model=scvi_model,
        gene_feature_dim=32,
    )
    sample = bundle.as_sample_dict()
    np.savez_compressed(out_dir / "iwm_sample.npz", **sample)
    log.info("IWM sample saved: %d cells, %d perturbations, gene_feat_dim=%d",
             Z.shape[0], bundle.meta["n_perturbations"], sample["gene_feat"].shape[1])
    return sample


def _train_iwm(sample: dict, out_dir: Path, epochs: int, accelerator: str) -> "object":
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from iwm.data.datamodule import StepDataset
    from iwm.models.world_model import WorldModelCfg
    from iwm.training.losses import LossConfig
    from iwm.training.trainer import IWMLightningModule, TrainCfg

    N = sample["z_seq"].shape[0]
    traj_len = sample["z_seq"].shape[1] - 1          # 1 for snapshot data
    latent_dim = sample["z_seq"].shape[2]
    gene_input_dim = int(sample["gene_feat"].shape[1])
    n_drugs = int(sample["drug_onehot"].shape[1])
    n_cytokines = int(sample["cyto_ext_seq"].shape[2])

    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    n_val = max(1, int(N * 0.1))
    train_s = {k: v[idx[:N - n_val]] for k, v in sample.items()
               if hasattr(v, "__len__") and len(v) == N}
    val_s   = {k: v[idx[N - n_val:]] for k, v in sample.items()
               if hasattr(v, "__len__") and len(v) == N}
    for s in (train_s, val_s):
        n = s["z_seq"].shape[0]
        s["neighbors"] = (s["neighbors"] % n).astype(np.int64)

    train_ds = StepDataset(train_s, step_lo=0, step_hi=traj_len, rollout_steps=1)
    val_ds   = StepDataset(val_s,   step_lo=0, step_hi=traj_len, rollout_steps=1)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=0)

    wm_cfg = WorldModelCfg(
        latent_dim=latent_dim,
        gene_input_dim=gene_input_dim,
        n_drugs=n_drugs,
        n_cytokines=n_cytokines,
        gene_embed_dim=16,
        drug_embed_dim=16,
        cyto_embed_dim=16,
        env_embed_dim=32,
        n_layers=4,
        n_heads=4,
        ffn_dim=128,
        dropout=0.1,
        d_tok=64,
    )
    loss_cfg = LossConfig(
        latent_mse=1.0, latent_nll=0.3, vicreg=0.1,
        cyto_mse=0.0,  fate_ce=0.0,    jepa=0.2,
        rollout=0.0,   flow=0.0,
    )
    train_cfg = TrainCfg(model=wm_cfg, loss=loss_cfg, lr=3e-4, weight_decay=1e-4,
                         rollout_steps=1, grad_clip=1.0)

    module = IWMLightningModule(train_cfg)
    from iwm.utils import lightning_accelerator
    accel = lightning_accelerator(accelerator)
    trainer = pl.Trainer(
        max_epochs=epochs, gradient_clip_val=1.0,
        default_root_dir=str(out_dir), enable_checkpointing=False,
        log_every_n_steps=1, accelerator=accel, devices=1, precision=32, logger=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ckpt = {"state_dict": module.state_dict(), "train_cfg": train_cfg}
    torch.save(ckpt, out_dir / "wm.pt")
    log.info("IWM checkpoint saved to %s", out_dir / "wm.pt")
    return module.model.eval()


def _evaluate(sample: dict, wm, adata, scvi_model_dir: Path, out_dir: Path,
              top_k: int = 20, ood_frac: float = 0.30):
    """OOD-gene eval: hold out `ood_frac` fraction of perturbations."""
    import scanpy as sc
    from iwm.embeddings.scvi_wrapper import ScviEncoder
    from iwm.evaluation.metrics import pearson_delta

    N = sample["z_seq"].shape[0]
    gene_id = sample["gene_id"]
    unique_gids = np.unique(gene_id)
    rng = np.random.default_rng(42)
    n_ood = max(2, int(len(unique_gids) * ood_frac))
    ood_gids = rng.choice(unique_gids, size=n_ood, replace=False)
    iid_mask = ~np.isin(gene_id, ood_gids)
    ood_mask =  np.isin(gene_id, ood_gids)
    log.info("Eval: %d IID  |  %d OOD cells (%d perturbations held out)",
             iid_mask.sum(), ood_mask.sum(), n_ood)

    z_seq   = sample["z_seq"].astype(np.float32)
    gf      = sample["gene_feat"].astype(np.float32)
    drug    = sample["drug_onehot"].astype(np.float32)
    cyto    = sample["cyto_ext_seq"].astype(np.float32)
    nbrs    = sample["neighbors"].astype(np.int64)

    z0 = z_seq[:, 0]
    z1 = z_seq[:, 1]            # "true" perturbed latent

    def predict(mask):
        neigh_idx = nbrs[mask] % N
        with torch.no_grad():
            out = wm(
                z_t        = torch.from_numpy(z0[mask]),
                gene_feat  = torch.from_numpy(gf[mask]),
                drug_onehot = torch.from_numpy(drug[mask]),
                cyto_dose  = torch.from_numpy(cyto[mask, 0]),
                z_nbr      = torch.from_numpy(z0[neigh_idx]),
            )
        return out["mu"].numpy()

    def ridge_predict(mask, train_mask):
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0)
        reg.fit(z0[train_mask], z1[train_mask])
        return reg.predict(z0[mask])

    log.info("Running IWM prediction on OOD cells…")
    pred_ood_iwm   = predict(ood_mask)
    pred_ood_ridge = ridge_predict(ood_mask, iid_mask)
    pred_ood_mean  = z1[iid_mask].mean(axis=0, keepdims=True) * np.ones((ood_mask.sum(), 1))

    delta_true  = (z1[ood_mask]    - z0[ood_mask]).mean(axis=0, keepdims=True)
    delta_iwm   = (pred_ood_iwm    - z0[ood_mask]).mean(axis=0, keepdims=True)
    delta_ridge = (pred_ood_ridge  - z0[ood_mask]).mean(axis=0, keepdims=True)
    delta_mean  = (pred_ood_mean   - z0[ood_mask]).mean(axis=0, keepdims=True)

    pr_iwm   = pearson_delta(delta_iwm,   delta_true)
    pr_ridge = pearson_delta(delta_ridge, delta_true)
    pr_mean  = pearson_delta(delta_mean,  delta_true)

    log.info("OOD latent-space Pearson(delta_pred, delta_true):")
    log.info("  IWM   = %.3f", pr_iwm)
    log.info("  Ridge = %.3f", pr_ridge)
    log.info("  Mean  = %.3f", pr_mean)

    results = {"iwm": float(pr_iwm), "ridge": float(pr_ridge), "mean_shift": float(pr_mean),
               "n_ood_cells": int(ood_mask.sum()), "n_ood_perts": int(n_ood)}
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))

    # Gene-space decoding via kNN retrieval (realistic LFC magnitudes)
    try:
        import scanpy as sc
        from scipy.sparse import issparse
        from iwm.embeddings.scvi_wrapper import ScviEncoder as _Enc

        m = torch.load(str(scvi_model_dir / "model.pt"), map_location="cpu", weights_only=False)
        saved_vars = list(m["var_names"])
        adata_knn = adata[:, saved_vars].copy() if hasattr(adata, "var") else adata.copy()
        if "counts" not in adata_knn.layers:
            adata_knn.layers["counts"] = adata_knn.X.copy()
        enc = _Enc.from_trained(str(scvi_model_dir), adata=adata_knn)

        adata_log = adata_knn.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e4)
        sc.pp.log1p(adata_log)
        X_log1p = adata_log.X
        if issparse(X_log1p): X_log1p = X_log1p.toarray()
        Z_ref = enc.encode(adata_knn)

        log.info("kNN-decoding %d OOD cells to gene space…", ood_mask.sum())
        expr_pred = enc.knn_decode(pred_ood_iwm,    Z_ref, X_log1p)
        expr_true = enc.knn_decode(z1[ood_mask],    Z_ref, X_log1p)
        expr_ctrl = enc.knn_decode(z0[ood_mask],    Z_ref, X_log1p)

        delta_gene_pred = (expr_pred - expr_ctrl).mean(axis=0)
        delta_gene_true = (expr_true - expr_ctrl).mean(axis=0)

        gene_names = enc.gene_names
        top_idx = np.argsort(np.abs(delta_gene_pred))[::-1][:top_k]
        lines = ["gene\tdelta_pred\tdelta_true\tdirection"]
        for i in top_idx:
            d = "UP" if delta_gene_pred[i] > 0 else "DOWN"
            lines.append(f"{gene_names[i]}\t{delta_gene_pred[i]:.4f}\t{delta_gene_true[i]:.4f}\t{d}")

        (out_dir / "top_ood_degs.tsv").write_text("\n".join(lines))
        log.info("Top %d OOD DEGs (kNN) written to top_ood_degs.tsv", top_k)

        from iwm.evaluation.metrics import pearson_delta as pd2
        pr_gene = pd2(delta_gene_pred[None], delta_gene_true[None])
        pr_gene_ridge = pd2(
            (enc.knn_decode(pred_ood_ridge, Z_ref, X_log1p) - expr_ctrl).mean(0, keepdims=True),
            delta_gene_true[None],
        )
        log.info("Gene-space Pearson  IWM=%.3f  Ridge=%.3f", pr_gene, pr_gene_ridge)
        results["gene_pearson_iwm"]   = float(pr_gene)
        results["gene_pearson_ridge"] = float(pr_gene_ridge)
        (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    except Exception as exc:
        log.warning("Gene-space decoding skipped: %s", exc)

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default="results/perturb_seq_run")
    ap.add_argument("--n-cells",      type=int, default=5000)
    ap.add_argument("--latent",       type=int, default=32)
    ap.add_argument("--scvi-epochs",  type=int, default=15)
    ap.add_argument("--iwm-epochs",   type=int, default=30)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--accelerator",  default="auto")
    ap.add_argument("--top-k",        type=int, default=20)
    ap.add_argument("--ood-frac",     type=float, default=0.30,
                    help="Fraction of perturbations held out for OOD test")
    ap.add_argument("--dataset",      default="adamson2016",
                    choices=["adamson2016", "norman2019"],
                    help="Which Perturb-seq dataset to use")
    args = ap.parse_args()

    _check_deps()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load data
    if args.dataset == "norman2019":
        from iwm.data.real.norman2019 import load_norman2019
        adata = load_norman2019(cache_dir="data_cache", single_only=True,
                                max_cells=args.n_cells, seed=args.seed)
    else:
        from iwm.data.real.adamson2016 import load_adamson2016
        adata = load_adamson2016(cache_dir="data_cache",
                                 max_cells=args.n_cells, seed=args.seed)

    # Step 2: Train scVI
    Z, _scvi = _train_scvi(adata, out_dir, args.latent, args.scvi_epochs, args.accelerator)

    # Step 3: Build IWM sample (with scVI-derived gene features)
    sample = _build_sample(adata, Z, _scvi, out_dir)

    # Step 4: Train IWM
    wm = _train_iwm(sample, out_dir, args.iwm_epochs, args.accelerator)

    # Step 5: Eval
    results = _evaluate(sample, wm, adata, out_dir / "scvi_model", out_dir,
                        top_k=args.top_k, ood_frac=args.ood_frac)

    log.info("\n=== Perturb-seq Validation Summary ===")
    log.info("IWM   OOD Pearson: %.3f", results["iwm"])
    log.info("Ridge OOD Pearson: %.3f", results["ridge"])
    log.info("Mean  OOD Pearson: %.3f", results["mean_shift"])
    beat = results["iwm"] > results["ridge"]
    log.info("IWM beats Ridge: %s", "YES ✓" if beat else "NO — needs more work")


if __name__ == "__main__":
    main()
