# Architecture — Immune World Model (IWM)

## Overview

Three layers, each independently testable.

```
┌───────────────────────────────────────────────────────────────────┐
│ L3: Agent Simulator                                               │
│  N cells in a tissue; shared cytokine field c(t).                 │
│  At step t: for each cell i                                       │
│    a_env_i = {c(t),  neighbor_pool({z_j : j in N(i)})}            │
│    z_i(t+1), secretion_i = WM(z_i(t), a_gene, a_drug, a_cyto,     │
│                                 a_env_i)                          │
│  c(t+1) = decay * c(t) + sum_i secretion_i   (mean-field)         │
└───────────────────────────────────────────────────────────────────┘
                               ▲
┌───────────────────────────────────────────────────────────────────┐
│ L2: World Model Core — transformer, action-conditioned            │
│  Inputs:  z_t  ∈ R^D                                              │
│           a_gene ∈ R^{E_g}    (gene KO embedding)                 │
│           a_drug ∈ R^{E_d}    (SMILES / MolFormer embed)          │
│           a_cyto ∈ R^{C}      (cytokine dose vector, C≈86)        │
│           a_env  ∈ R^{D}      (pooled neighbor state + field)     │
│           Δt     ∈ R^+        (time-step size)                    │
│  Arch:                                                            │
│    Tokens = [CLS_z, z_t, a_gene, a_drug, a_cyto, a_env, Δt]       │
│    6 × TransformerBlock with FiLM on Δt                           │
│    Heads:  z_mu, z_log_sigma                 (heteroscedastic)    │
│            cytokine_secretion_hat           (R^{C})               │
│            fate_logits                      (proliferate/die/∅)   │
│  Output: z_{t+Δt} ~ N(μ, σ²)                                      │
└───────────────────────────────────────────────────────────────────┘
                               ▲
┌───────────────────────────────────────────────────────────────────┐
│ L1: Cell Embedding                                                │
│  Primary:    scVI (BSD-3, commercial OK) → z ∈ R^64               │
│  Zero-shot:  UCE (MIT) → z ∈ R^1280                               │
│  Research:   STATE-SE (non-commercial) → z ∈ R^768  [optional]    │
│                                                                   │
│  Action encoders:                                                 │
│    gene:  learned embedding or ESM-2/Geneformer[research]         │
│    drug:  MolFormer (MIT) or ChemBERTa over SMILES                │
│    cyto:  86-d dose vector, log-scaled, learned projection        │
│    env:   mean-pool + attention-pool over neighbor latents        │
└───────────────────────────────────────────────────────────────────┘
```

## Commercial-clean vs. research tracks

Two parallel configs. Same interface, different L1/action encoders.

| Component | Commercial (default) | Research |
|---|---|---|
| Cell embedding | scVI (BSD-3) | STATE-SE (Arc, NC) |
| Gene encoder | Learned + gene2vec | Geneformer (Broad, NC) |
| Drug encoder | MolFormer (MIT) | same |
| Cytokine encoder | Learned projection | Immune Dictionary-conditioned |
| Training data (pretrain) | CellxGene Census (CC-BY) | + Tahoe-100M (CC-BY-NC) |

Runtime switch via `configs/commercial.yaml` vs `configs/research.yaml`.

## Losses (L2)

```
L_total = λ₁ · L_latent_mse            # main state-prediction loss
       + λ₂ · L_latent_nll             # Gaussian NLL (uncertainty)
       + λ₃ · L_vicreg                 # variance/covariance regularizer
                                       # prevents collapse, VICReg Bardes 2022
       + λ₄ · L_cyto_mse               # cytokine secretion head
       + λ₅ · L_fate_ce                # cell-fate classification
       + λ₆ · L_jepa                   # JEPA-style masked latent prediction
       + λ₇ · L_rollout                # 2+ step rollout consistency
       + λ₈ · L_flow                   # optional flow-matching velocity
```

Defaults: `λ = (1.0, 0.5, 0.1, 0.3, 0.2, 0.3, 0.5, 0.0)`.
`L_flow` off by default; enable when trajectory density is high.

## Evaluation splits (critical)

All four must be reported.

1. **IID** — random 80/20 holdout within (cell, perturbation, time) triples.
2. **OOD-gene** — 20% of perturbation genes never seen in training. (Systema-style; the hard test.)
3. **OOD-temporal** — training uses early time-points only; eval predicts late time-points given early-state input. (Our pillar 1.)
4. **OOD-multicell** — held-out neighborhood compositions (novel mixtures of cell types/states). (Our pillar 2.)

## Metrics (Systema-style, not Pearson-on-raw)

Report all four, not just the best one.

- `pearson_delta`: Pearson on (predicted − control_mean) vs. (observed − control_mean).
- `centroid_accuracy`: is the predicted post-perturbation centroid closer to the true perturbation centroid than to any other?
- `deg_recall@k`: recall of top-k DEGs (k=30 default).
- `rollout_pearson`: Pearson over T≥5 step rollout vs. observed trajectory.

## Baselines (must include in every run)

| ID | Description | Expected to win on |
|---|---|---|
| B0 | Identity (predict control) | — |
| B1 | Perturbed mean (per-perturbation global mean shift) | IID, often OOD-gene (Systema) |
| B2 | Ridge on action one-hot | IID |
| B3 | Motif-bilinear (gene embedding × context) | OOD-gene |
| B4 | scVI latent + ridge delta | IID |

IWM must beat all four on OOD-temporal and OOD-multicell; must match or beat B3 on OOD-gene.

## Rollout / counterfactual API (L3 deliverable)

```python
env = TissueEnv(
    initial_cells=pop_df,              # N cells × L1 latent
    interventions=[
        {"step": 0, "target": "T_cells", "action": a_cyto_IL2_10ng},
        {"step": 4, "target": "all",     "action": a_drug_CX4945},
    ],
    dt=6.0,                            # hours
    horizon=48,
)
trajectory = env.rollout()             # returns (N, horizon, D)
counterfactual = env.fork().rollout()  # replay with modified interventions
```

This `fork + modify + rollout` is the core user-facing primitive — the thing neither VCWorld nor STATE/Stack can do.
