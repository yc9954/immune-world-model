"""Evaluate WorldModel + all baselines on the four splits with
Systema-style metrics. Prints and saves a markdown table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from iwm.data import IWMDataModule
from iwm.data.datamodule import DataModuleCfg
from iwm.data.synthetic import SynthConfig, ThreePillarSynth
from iwm.evaluation.metrics import pearson_delta, deg_recall_at_k
from iwm.models.baselines import (
    Identity, PerturbedMean, RidgeOneHot, MotifBilinear, ScviPlusRidge,
)
from iwm.models.world_model import WorldModel, WorldModelCfg
from iwm.training.trainer import IWMLightningModule, TrainCfg
from iwm.training.losses import LossConfig
from iwm.utils import pick_device, set_seed, tune_threads


# ---------------------------------------------------------------- utils

def _build_synth(conf: dict) -> SynthConfig:
    d = conf["data"]
    return SynthConfig(
        n_cells=d["n_cells"], n_cell_types=d["n_cell_types"],
        n_genes_pert=d["n_genes_pert"], n_drugs=d["n_drugs"],
        n_cytokines=d["n_cytokines"], latent_dim=d["latent_dim"],
        gene_feature_dim=d.get("gene_feature_dim", 8),
        trajectory_len=d["trajectory_len"], dt=d["dt"],
        neighbor_k=d["neighbor_k"], noise_std=d["noise_std"],
        coupling_strength=d["coupling_strength"],
        spectral_radius=d.get("spectral_radius", 0.9),
        init_scale=d.get("init_scale", 1.5),
        action_scale=d.get("action_scale", 1.0),
        nonlinearity=d.get("nonlinearity", "tanh"),
        nonlinearity_scale=d.get("nonlinearity_scale", 3.0),
        seed=conf["seed"],
    )


def _predict_baseline_on_sample(baseline, sample: dict, step: int) -> np.ndarray:
    return baseline.predict_next(sample, step)


def _predict_wm_on_sample(
    model: WorldModel, sample: dict, step: int, device: torch.device,
    ablate_neighbor: bool = False,
) -> np.ndarray:
    z_t = torch.from_numpy(sample["z_seq"][:, step]).to(device)
    gene_feat = torch.from_numpy(sample["gene_feat"]).to(device)
    drug = torch.from_numpy(sample["drug_onehot"]).to(device)
    cyto = torch.from_numpy(sample["cyto_ext_seq"][:, step]).to(device)
    neigh = torch.from_numpy(sample["neighbors"]).to(device)
    z_nbr = z_t[neigh]
    if ablate_neighbor:
        z_nbr = torch.zeros_like(z_nbr)
    with torch.no_grad():
        out = model(z_t, gene_feat, drug, cyto, z_nbr)
    return out["mu"].cpu().numpy()


def _score_one_step(pred_next: np.ndarray, sample: dict, step: int) -> Dict[str, float]:
    z_t = sample["z_seq"][:, step]
    z_true = sample["z_seq"][:, step + 1]
    delta_pred = pred_next - z_t
    delta_true = z_true - z_t
    return {
        "pearson_delta": pearson_delta(delta_pred, delta_true),
        "deg_recall_30": deg_recall_at_k(delta_pred, delta_true, k=30),
    }


def _score_rollout(
    pred_fn, sample: dict, start_step: int, end_step: int,
) -> Dict[str, float]:
    """Closed-loop rollout: feed prediction back in as the next input."""
    z = sample["z_seq"][:, start_step].copy()
    true_traj = [sample["z_seq"][:, start_step]]
    pred_traj = [z.copy()]
    mock_sample = {k: v for k, v in sample.items()}
    for step in range(start_step, end_step):
        mock_sample = dict(sample)
        mock_sample["z_seq"] = sample["z_seq"].copy()
        mock_sample["z_seq"][:, step] = z        # overwrite for baseline lookup
        pred = pred_fn(mock_sample, step)
        z = pred
        pred_traj.append(z.copy())
        true_traj.append(sample["z_seq"][:, step + 1])
    pred_arr = np.stack(pred_traj, axis=1)
    true_arr = np.stack(true_traj, axis=1)
    delta_pred = pred_arr[:, 1:] - pred_arr[:, :-1]
    delta_true = true_arr[:, 1:] - true_arr[:, :-1]
    return {"rollout_pearson": pearson_delta(delta_pred, delta_true)}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="iwm/configs/toy.yaml")
    ap.add_argument("--ckpt", default="results/toy_run/wm.pt")
    ap.add_argument("--out", default="results/toy_run")
    ap.add_argument("--device", default="auto",
                    help="auto / mps / cuda / cpu")
    args = ap.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text())
    set_seed(conf["seed"])
    tune_threads()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_cfg = _build_synth(conf)
    gen = ThreePillarSynth(synth_cfg)

    samples = {
        "train": gen.sample("iid"),
        "iid":   gen.sample("iid"),
        "ood_gene": gen.sample("ood_gene"),
        "ood_temporal": None,        # filled in below
        "ood_multicell": gen.sample("ood_multicell"),
    }
    samples["ood_temporal"] = samples["iid"]   # same cells, but evaluated at later steps

    step_lo, step_hi = 0, conf["splits"]["ood_temporal_split_at"]
    later_lo = conf["splits"]["ood_temporal_split_at"]
    later_hi = synth_cfg.trajectory_len

    # ------------------------------------------------------------ baselines
    baselines = [Identity(), PerturbedMean(), RidgeOneHot(), MotifBilinear(), ScviPlusRidge()]
    for b in baselines:
        b.fit(samples["train"], step_lo=step_lo, step_hi=step_hi)

    device = pick_device(args.device)
    print(f"[eval] device={device}")

    # ------------------------------------------------------------ WM
    wm = None
    if Path(args.ckpt).exists():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m = conf["model"]; d = conf["data"]
        wm_cfg = WorldModelCfg(
            latent_dim=d["latent_dim"], gene_input_dim=d.get("gene_feature_dim", 8) or d["n_genes_pert"],
            n_drugs=d["n_drugs"], n_cytokines=d["n_cytokines"],
            gene_embed_dim=m["gene_embed_dim"], drug_embed_dim=m["drug_embed_dim"],
            cyto_embed_dim=m["cyto_embed_dim"], env_embed_dim=m["env_embed_dim"],
            n_layers=m["n_layers"], n_heads=m["n_heads"],
            ffn_dim=m["ffn_dim"], dropout=m["dropout"],
        )
        train_cfg = TrainCfg(model=wm_cfg, loss=LossConfig())
        module = IWMLightningModule(train_cfg)
        module.load_state_dict(ckpt["state_dict"])
        wm = module.model.to(device).eval()

    # ------------------------------------------------------------ score

    rows: List[dict] = []
    def _score_model(name: str, predict_fn) -> None:
        row = {"model": name}
        # IID / OOD_gene / OOD_multicell: one-step at step_lo+2 (mid-training window)
        for split in ("iid", "ood_gene", "ood_multicell"):
            sample = samples[split]
            step = (step_lo + step_hi) // 2
            pred = predict_fn(sample, step)
            s = _score_one_step(pred, sample, step)
            row[f"{split}_pearson_delta"] = round(s["pearson_delta"], 3)
            row[f"{split}_deg_recall_30"] = round(s["deg_recall_30"], 3)
        # OOD_temporal: rollout from later_lo to later_hi - 1
        roll = _score_rollout(predict_fn, samples["ood_temporal"], later_lo, later_hi - 1)
        row["ood_temporal_rollout_pearson"] = round(roll["rollout_pearson"], 3)
        rows.append(row)

    for b in baselines:
        _score_model(b.name, lambda s, st, bb=b: _predict_baseline_on_sample(bb, s, st))

    ablate_neighbor = conf.get("ablate_neighbor", False)
    if wm is not None:
        _score_model(
            "IWM_WorldModel",
            lambda s, st: _predict_wm_on_sample(wm, s, st, device, ablate_neighbor),
        )
    else:
        rows.append({"model": "IWM_WorldModel", "note": "no checkpoint found at " + args.ckpt})

    # ------------------------------------------------------------ write

    results_json = out_dir / "eval_results.json"
    results_json.write_text(json.dumps(rows, indent=2))

    # markdown table
    headers = [
        "model",
        "iid_pearson_delta", "iid_deg_recall_30",
        "ood_gene_pearson_delta", "ood_gene_deg_recall_30",
        "ood_multicell_pearson_delta", "ood_multicell_deg_recall_30",
        "ood_temporal_rollout_pearson",
    ]
    md_lines = ["| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md_lines.append(
            "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
        )
    md_path = out_dir / "eval_results.md"
    md_path.write_text("\n".join(md_lines))
    print("\n".join(md_lines))
    print(f"\n[eval] wrote {results_json} and {md_path}")


if __name__ == "__main__":
    main()
