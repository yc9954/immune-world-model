"""Train IWM on the synthetic three-pillar toy.

Usage:
    python scripts/train.py                  # uses toy.yaml defaults
    python scripts/train.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from iwm.data import IWMDataModule
from iwm.data.datamodule import DataModuleCfg, StepDataset
from iwm.data.synthetic import SynthConfig
from iwm.models.world_model import WorldModelCfg
from iwm.training.losses import LossConfig
from iwm.training.trainer import IWMLightningModule, TrainCfg
from iwm.utils import lightning_accelerator, set_seed, tune_threads


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _build_real_loaders(npz_path: str, conf: dict) -> tuple:
    """Load a real iwm_sample.npz and return (train_loader, val_loaders, TrainCfg)."""
    from torch.utils.data import DataLoader, random_split

    sample = dict(np.load(npz_path, allow_pickle=True))
    N = sample["z_seq"].shape[0]
    traj_len = sample["z_seq"].shape[1] - 1       # T=1 for snapshot real data
    latent_dim = sample["z_seq"].shape[2]
    gene_input_dim = int(sample["gene_feat"].shape[1])
    n_drugs = int(sample["drug_onehot"].shape[1])
    n_cytokines = int(sample["cyto_ext_seq"].shape[2])

    tconf = conf["training"]
    rollout_steps = min(tconf.get("rollout_steps", 1), traj_len)

    # 80/10/10 split by cell index
    rng = np.random.default_rng(conf.get("seed", 0))
    idx = rng.permutation(N)
    n_val = max(1, int(N * 0.1))
    train_sample = {k: v[idx[: N - n_val]] for k, v in sample.items()
                    if hasattr(v, "__len__") and len(v) == N}
    val_sample   = {k: v[idx[N - n_val :]] for k, v in sample.items()
                    if hasattr(v, "__len__") and len(v) == N}
    # fix neighbors to stay within each split
    for s in (train_sample, val_sample):
        n = s["z_seq"].shape[0]
        s["neighbors"] = (s["neighbors"] % n).astype(np.int64)

    train_ds = StepDataset(train_sample, step_lo=0, step_hi=traj_len, rollout_steps=rollout_steps)
    val_ds   = StepDataset(val_sample,   step_lo=0, step_hi=traj_len, rollout_steps=1)

    loader_kw = dict(batch_size=tconf.get("batch_size", 64), num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kw)
    val_loaders  = [DataLoader(val_ds,  shuffle=False, **loader_kw)]

    m = conf["model"]
    wm_cfg = WorldModelCfg(
        latent_dim=latent_dim,
        gene_input_dim=gene_input_dim,
        n_drugs=n_drugs,
        n_cytokines=n_cytokines,
        gene_embed_dim=m.get("gene_embed_dim", 16),
        drug_embed_dim=m.get("drug_embed_dim", 16),
        cyto_embed_dim=m.get("cyto_embed_dim", 16),
        env_embed_dim=m.get("env_embed_dim", 32),
        n_layers=m.get("n_layers", 4),
        n_heads=m.get("n_heads", 4),
        ffn_dim=m.get("ffn_dim", 128),
        dropout=m.get("dropout", 0.1),
        d_tok=m.get("d_tok", 64),
    )
    lw = tconf.get("loss_weights", {})
    loss = LossConfig(
        latent_mse=lw.get("latent_mse", 1.0), latent_nll=lw.get("latent_nll", 0.3),
        vicreg=lw.get("vicreg", 0.1),         cyto_mse=lw.get("cyto_mse", 0.3),
        fate_ce=lw.get("fate_ce", 0.0),       jepa=lw.get("jepa", 0.2),
        rollout=lw.get("rollout", 0.0),        flow=lw.get("flow", 0.0),
    )
    train_cfg = TrainCfg(
        model=wm_cfg, loss=loss,
        lr=tconf.get("lr", 3e-4),
        weight_decay=tconf.get("weight_decay", 1e-4),
        rollout_steps=rollout_steps,
        grad_clip=tconf.get("grad_clip", 1.0),
    )
    return train_loader, val_loaders, train_cfg


def _build_cfgs(conf: dict) -> tuple[DataModuleCfg, TrainCfg]:
    d = conf["data"]
    synth = SynthConfig(
        n_cells=d["n_cells"],
        n_cell_types=d["n_cell_types"],
        n_genes_pert=d["n_genes_pert"],
        n_drugs=d["n_drugs"],
        n_cytokines=d["n_cytokines"],
        latent_dim=d["latent_dim"],
        gene_feature_dim=d.get("gene_feature_dim", 8),
        trajectory_len=d["trajectory_len"],
        dt=d["dt"],
        neighbor_k=d["neighbor_k"],
        noise_std=d["noise_std"],
        coupling_strength=d["coupling_strength"],
        spectral_radius=d.get("spectral_radius", 0.9),
        init_scale=d.get("init_scale", 1.5),
        action_scale=d.get("action_scale", 1.0),
        nonlinearity=d.get("nonlinearity", "tanh"),
        nonlinearity_scale=d.get("nonlinearity_scale", 3.0),
        seed=conf["seed"],
    )
    tconf = conf["training"]
    dm_cfg = DataModuleCfg(
        synth=synth,
        batch_size=tconf["batch_size"],
        rollout_steps=tconf["rollout_steps"],
        ood_temporal_split_at=conf["splits"]["ood_temporal_split_at"],
        num_workers=tconf.get("num_workers", 0),
        persistent_workers=tconf.get("persistent_workers", False),
        pin_memory=False,       # MPS does not benefit from pinned host memory
    )

    m = conf["model"]
    wm_cfg = WorldModelCfg(
        latent_dim=d["latent_dim"],
        gene_input_dim=d.get("gene_feature_dim", 8) or d["n_genes_pert"],
        n_drugs=d["n_drugs"],
        n_cytokines=d["n_cytokines"],
        gene_embed_dim=m["gene_embed_dim"],
        drug_embed_dim=m["drug_embed_dim"],
        cyto_embed_dim=m["cyto_embed_dim"],
        env_embed_dim=m["env_embed_dim"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        ffn_dim=m["ffn_dim"],
        dropout=m["dropout"],
        d_tok=m.get("d_tok", 64),
    )
    lw = conf["training"]["loss_weights"]
    loss = LossConfig(
        latent_mse=lw["latent_mse"], latent_nll=lw["latent_nll"],
        vicreg=lw["vicreg"], cyto_mse=lw["cyto_mse"], fate_ce=lw["fate_ce"],
        jepa=lw["jepa"], rollout=lw["rollout"], flow=lw["flow"],
    )
    train_cfg = TrainCfg(
        model=wm_cfg, loss=loss,
        lr=conf["training"]["lr"],
        weight_decay=conf["training"]["weight_decay"],
        rollout_steps=conf["training"]["rollout_steps"],
        grad_clip=conf["training"]["grad_clip"],
    )
    return dm_cfg, train_cfg


def _validate_multi(trainer: pl.Trainer, module: IWMLightningModule, dm: IWMDataModule) -> dict:
    """Run validation on each split separately so per-split metrics are logged."""
    loaders = dm.val_dataloader()
    out = {}
    for name, loader in loaders.items():
        results = trainer.validate(module, dataloaders=loader, verbose=False)
        if results:
            r = results[0]
            for k, v in r.items():
                if name in k or "loss" in k:
                    out[f"{name}/{k}"] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="iwm/configs/toy.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None,
                    help="real:path/to/iwm_sample.npz — skips synthetic generation")
    ap.add_argument("--accelerator", default=None,
                    help="Override training.accelerator (auto/mps/cuda/cpu)")
    ap.add_argument("--precision", default=None,
                    help="Override training.precision (32, 16-mixed, bf16-mixed)")
    args = ap.parse_args()

    conf = _load_yaml(Path(args.config))
    if args.epochs is not None:
        conf["training"]["max_epochs"] = args.epochs
    if args.accelerator is not None:
        conf["training"]["accelerator"] = args.accelerator
    if args.precision is not None:
        conf["training"]["precision"] = args.precision
    out_dir = Path(args.out or conf["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(conf["seed"])
    threads = conf["training"].get("threads") or None
    applied_threads = tune_threads(threads)

    real_npz = None
    if args.data and args.data.startswith("real:"):
        real_npz = args.data[len("real:"):]

    ablate_neighbor = conf.get("ablate_neighbor", False)

    if real_npz:
        train_loader, val_loaders, train_cfg = _build_real_loaders(real_npz, conf)
        num_workers_display = 0
        print(f"[train] real-data mode: {real_npz}")
    else:
        dm_cfg, train_cfg = _build_cfgs(conf)
        dm = IWMDataModule(dm_cfg)
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loaders = list(dm.val_dataloader().values())
        num_workers_display = dm_cfg.num_workers

    module = IWMLightningModule(train_cfg, ablate_neighbor=ablate_neighbor)

    accelerator = lightning_accelerator(conf["training"].get("accelerator", "auto"))
    precision = conf["training"].get("precision", 32)
    print(f"[train] accelerator={accelerator} precision={precision} "
          f"num_workers={num_workers_display} threads={applied_threads}")

    trainer = pl.Trainer(
        max_epochs=conf["training"]["max_epochs"],
        gradient_clip_val=train_cfg.grad_clip,
        default_root_dir=str(out_dir),
        enable_checkpointing=False,
        log_every_n_steps=1,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loaders)

    # Persist the trained model + a small manifest.
    ckpt = {"state_dict": module.state_dict(), "train_cfg": train_cfg}
    if not real_npz:
        ckpt["synth_cfg"] = dm_cfg.synth
    torch.save(ckpt, out_dir / "wm.pt")

    manifest = {
        "config": args.config,
        "epochs": conf["training"]["max_epochs"],
        "n_params": sum(p.numel() for p in module.parameters()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[train] wrote {out_dir / 'wm.pt'} (params={manifest['n_params']})")


if __name__ == "__main__":
    main()
