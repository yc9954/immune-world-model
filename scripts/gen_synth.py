"""Inspect the synthetic generator without training anything.

Prints summary statistics so you can confirm the three pillars are
actually present: non-trivial temporal evolution, neighbor-coupled
dynamics, and compositional action decomposition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from iwm.data.synthetic import SynthConfig, ThreePillarSynth
from iwm.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="iwm/configs/toy.yaml")
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    set_seed(conf["seed"])

    d = conf["data"]
    cfg = SynthConfig(
        n_cells=d["n_cells"], n_cell_types=d["n_cell_types"],
        n_genes_pert=d["n_genes_pert"], n_drugs=d["n_drugs"],
        n_cytokines=d["n_cytokines"], latent_dim=d["latent_dim"],
        trajectory_len=d["trajectory_len"], dt=d["dt"],
        neighbor_k=d["neighbor_k"], noise_std=d["noise_std"],
        coupling_strength=d["coupling_strength"], seed=conf["seed"],
    )
    gen = ThreePillarSynth(cfg)
    s = gen.sample("iid")

    z = s["z_seq"]
    print("z_seq shape:", z.shape)
    print("step-wise latent norm (mean):",
          np.linalg.norm(z, axis=-1).mean(axis=0).round(3))
    print("step-wise delta norm (mean):",
          np.linalg.norm(np.diff(z, axis=1), axis=-1).mean(axis=0).round(3))
    # Compositional check — is gene effect separable from drug?
    # Compare mean delta grouped by gene vs. by drug independently.
    deltas = (z[:, 1:] - z[:, :-1]).mean(axis=1)     # (N, d)
    by_gene = {g: deltas[s["gene_id"] == g].mean(axis=0)
               for g in range(cfg.n_genes_pert)}
    by_drug = {d_: deltas[s["drug_id"] == d_].mean(axis=0)
               for d_ in range(cfg.n_drugs)}
    print("gene-group delta variance across genes:",
          np.var(np.stack(list(by_gene.values())), axis=0).mean().round(4))
    print("drug-group delta variance across drugs:",
          np.var(np.stack(list(by_drug.values())), axis=0).mean().round(4))


if __name__ == "__main__":
    main()
