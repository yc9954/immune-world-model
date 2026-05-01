# Evaluation protocol

Goal: ensure every change is measured against the same reference baselines, on the same four splits, with the same four metrics. Follow Systema (Nat Biotech 2025) — Pearson on raw expression is a leaky metric; use Pearson on delta + centroid accuracy as primary.

## The four splits

| Split | Construction | What it tests |
|---|---|---|
| IID | Random 80/20 over (cell_id, perturbation, t) | Fit |
| OOD-gene | Hold out 20% of perturbation genes | Compositional gene generalization |
| OOD-temporal | Train on t ∈ {0..T_tr}; eval on t ∈ {T_tr+1..T_end} given z_0 | Pillar 1 — trajectory |
| OOD-multicell | Train on neighborhood compositions ∈ set A; eval on novel compositions ∈ set B | Pillar 2 — multi-cell |

## The four metrics (report all, per split, per baseline)

1. **pearson_delta** — `pearsonr(pred - ctrl_mean, obs - ctrl_mean)` over genes.
2. **centroid_accuracy** — `1 if argmin_p ||pred_centroid - true_centroid_p||₂ == p* else 0`, averaged over perturbations.
3. **deg_recall@30** — `|top30_pred ∩ top30_obs| / 30`.
4. **rollout_pearson** — only for OOD-temporal and L3 runs; Pearson of whole predicted trajectory vs. observed.

## Reporting template

| Model | IID delta | OOD-gene delta | OOD-temporal pearson | OOD-multi delta |
|---|---|---|---|---|
| B0 Identity | — | — | — | — |
| B1 Perturbed-mean | … | … | … | … |
| B2 Ridge-onehot | … | … | … | … |
| B3 Motif-bilinear | … | … | … | … |
| B4 scVI+ridge | … | … | … | … |
| **IWM v1** | … | … | … | … |

## Win conditions (pre-registered)

- **Must**: IWM ≥ B3 on OOD-gene (within 0.02 Pearson).
- **Must**: IWM ≥ best-baseline + 0.05 Pearson on OOD-temporal.
- **Must**: IWM ≥ best-baseline + 0.05 Pearson on OOD-multicell.
- **Nice**: IWM rollout_pearson ≥ 0.80 at T = 5.

If any "must" fails, do not claim the architecture works — debug before scaling.
