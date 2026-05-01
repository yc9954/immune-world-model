"""Demonstrate the counterfactual rollout primitive:

    env.rollout()
    env.with_intervention(...).rollout()

This is the user-facing capability that endpoint-only models
(STATE/Stack/scFMs/VCWorld) cannot provide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from iwm.data.synthetic import SynthConfig, ThreePillarSynth
from iwm.models.agent_simulator import Intervention, TissueEnv
from iwm.models.world_model import WorldModel, WorldModelCfg
from iwm.training.losses import LossConfig
from iwm.training.trainer import IWMLightningModule, TrainCfg


def _load_wm(ckpt: str, conf: dict) -> WorldModel:
    m = conf["model"]; d = conf["data"]
    wm_cfg = WorldModelCfg(
        latent_dim=d["latent_dim"],
        gene_input_dim=d.get("gene_feature_dim", 8),
        n_drugs=d["n_drugs"], n_cytokines=d["n_cytokines"],
        gene_embed_dim=m["gene_embed_dim"], drug_embed_dim=m["drug_embed_dim"],
        cyto_embed_dim=m["cyto_embed_dim"], env_embed_dim=m["env_embed_dim"],
        n_layers=m["n_layers"], n_heads=m["n_heads"],
        ffn_dim=m["ffn_dim"], dropout=m["dropout"],
        d_tok=m.get("d_tok", 64),
    )
    module = IWMLightningModule(TrainCfg(model=wm_cfg, loss=LossConfig()))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])
    return module.model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="iwm/configs/toy.yaml")
    ap.add_argument("--ckpt", default="results/toy_run/wm.pt")
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text())
    d = conf["data"]

    gen_cfg = SynthConfig(
        n_cells=128, n_cell_types=d["n_cell_types"],
        n_genes_pert=d["n_genes_pert"], n_drugs=d["n_drugs"],
        n_cytokines=d["n_cytokines"], latent_dim=d["latent_dim"],
        gene_feature_dim=d.get("gene_feature_dim", 8),
        trajectory_len=args.horizon, dt=d["dt"],
        neighbor_k=d["neighbor_k"], noise_std=d["noise_std"],
        coupling_strength=d["coupling_strength"],
        nonlinearity=d.get("nonlinearity", "tanh"),
        nonlinearity_scale=d.get("nonlinearity_scale", 3.0),
        seed=conf["seed"] + 1,
    )
    gen = ThreePillarSynth(gen_cfg)
    s = gen.sample("iid")

    wm = _load_wm(args.ckpt, conf)

    env = TissueEnv(
        wm=wm,
        z0=torch.from_numpy(s["z_seq"][:, 0]),
        gene_feat=torch.from_numpy(s["gene_feat"]),
        drug_onehot=torch.from_numpy(s["drug_onehot"]),
        neighbors=torch.from_numpy(s["neighbors"]),
        horizon=args.horizon,
    )

    baseline = env.rollout()

    # counterfactual — at step 4, add IL-2-like cytokine dose (dim 0) to all cells
    dose = torch.zeros(gen_cfg.n_cytokines)
    dose[0] = 1.0
    mask = torch.ones(s["z_seq"].shape[0], dtype=torch.bool)
    cf_env = env.with_intervention(Intervention(step=4, target_mask=mask, delta_cyto=dose))
    cf = cf_env.rollout()

    divergence = (cf - baseline).norm(dim=-1).mean(dim=0).numpy()
    print("step-wise divergence between baseline and counterfactual:")
    print(divergence.round(3))
    print(
        f"Peak divergence step: {int(np.argmax(divergence))}; "
        f"peak value: {float(divergence.max()):.3f}"
    )


if __name__ == "__main__":
    main()
