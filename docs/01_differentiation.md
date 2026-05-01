# Differentiation: Immune World Model (IWM) vs. 2025–2026 Landscape

## TL;DR — three pillars nobody else has all of

1. **Temporal-native**: predicts **cell state trajectories** across multiple time steps, not a single perturbed endpoint.
2. **Multi-cell coupling**: each cell's next state depends on the **aggregated state of its neighbors** via a learned cytokine/ligand mean-field.
3. **Compositional actions**: action is a **vector** over `{gene KO, drug SMILES embedding, cytokine concentration, neighbor-state summary}` — continuous and composable — not a discrete "one of N perturbations" index.

Plus, a **license-clean** stack (scVI BSD-3, UCE MIT, MolFormer MIT, our own weights) for the commercial path.

---

## The field (April 2026) and why none of them cover the 3 pillars

### VCWorld (GENTEL-lab, arXiv 2512.00306, Nov 2025 → Feb 2026)
- **Paradigm**: LLM (Llama 3.1 8B) + curated pathway knowledge graph + chain-of-thought reasoning. CLI verbs: `de prepare / retrieve / prompt / infer`.
- **Data**: GeneTAK = 5 cell lines × 348 drugs, derived from Tahoe-100M.
- **Input action**: drug identifier (discrete).
- **Output**: differential expression (DE) endpoint + mechanistic explanation.
- **NOT** a world model in the Genie sense — no learned latent dynamics, no trajectory rollout, no multi-cell agents, single-perturbation endpoint only.
- **What we borrow**: evaluation framing on DE; pathway KG as soft prior (regularizer).
- **Where we diverge on all 3 pillars**: temporal (no), multi-cell (no), compositional (no — drug ID only).

### STATE (Arc Institute, 2025)
- SE + ST: set-transformer that maps `(control set, perturbation) → perturbed set`. Best-in-class for **unseen-perturbation endpoint prediction**.
- Still **cell-iid**, **endpoint-only**, **discrete gene-ID action**.
- License: non-commercial research. Blocker for sweetspot.co.kr product.

### Stack (Arc Institute, Jan 2026)
- Tabular attention + in-context learning over 149M cells. Simulate new conditions without fine-tuning.
- Huge usability win — but still endpoint-oriented single-cell; no temporal rollout, no multi-cell coupling, no compositional actions.
- License TBD (verify Stack-Large weights).

### C2S-Scale (Google/Yale, Oct 2025, 27B)
- Gemma-2 27B fine-tuned on "cell sentences". Genuine discovery (CX-4945 as IFN-conditional amplifier, experimentally validated).
- Apache-2.0 — **commercial-clean**. Useful as a *gene-relation prior*, not as a dynamics simulator.
- Single-cell LLM; no trajectories, no multi-cell, not explicitly action-compositional.

### Flow matching / TIGON / CellOT
- Model the cell-population *distribution* over time via optimal transport or flow matching. Closest to our "trajectory" pillar.
- But they are **population-level** (distribution → distribution) — they do not expose an action-conditioned, per-cell state-transition operator that can be composed with multi-cell agent rollouts.
- We borrow their training signal (flow matching loss) on top of a transformer WM.

### scFMs (scGPT, Geneformer, scFoundation, UCE)
- Cell embeddings. Linear baseline debate (Kedzierska 2024, Wenkel 2025, Systema 2025) shows they barely beat ridge on perturbation. We use them **only as optional L1 encoders**, never as the core claim.

### Classical ABMs (PhysiCell, CompuCell3D, PhysiGym)
- Multi-cell, spatial, trajectory-native — everything the above miss — but **hand-coded mechanics**, not learned.
- We take the **agent-simulator shell** (how to step many cells in a tissue) from this tradition and drop the learned WM into each cell as its internal dynamics.

---

## The competitive matrix

| Capability | VCWorld | STATE / Stack | scFMs | TIGON / CellOT | PhysiCell | **IWM (ours)** |
|---|---|---|---|---|---|---|
| Latent dynamics learned from data | ✗ (LLM+KG) | ✓ (endpoint) | ✓ (no dynamics) | ✓ (dist→dist) | ✗ (ODE+rules) | **✓** |
| Multi-timestep trajectory | ✗ | ✗ | ✗ | ✓ (distribution) | ✓ | **✓ (per-cell)** |
| Multi-cell coupling | ✗ | ✗ | ✗ | ✗ | ✓ | **✓ (learned field)** |
| Compositional actions | ✗ (drug ID) | ✗ (gene ID) | ✗ | limited | N/A | **✓ (vector)** |
| Interpretability | high | low | low | mid | high | mid (per-step attn) |
| Commercial license path | ? (Llama) | ✗ | mixed | ✓ | ✓ | **✓** |

**We are the first to combine learned latent dynamics + multi-cell coupling + compositional actions + commercial-clean.**

---

## What we explicitly do NOT claim

- Not a foundation model — we don't pretrain on 100M cells; we train a dynamics model on time-course data.
- Not mechanistic — we don't output pathway traces. We expose attention maps over actions/neighbors as the interpretability surface.
- Not tissue-scale — v1 is ~10³ cells in well-mixed-plus-neighborhood regime; true spatial tissue simulation is v2.
- Not a scFM competitor — on endpoint DE benchmarks we may lose to Stack/STATE. Our win condition is **trajectory faithfulness** and **multi-cell counterfactual rollout**, metrics those models don't even emit.

---

## Win conditions (what makes us publishable / shippable)

1. On time-course Perturb-seq (e.g., genome-scale CD4+ T-cell or T-cell exhaustion time-series), beat motif-ridge by ≥ 0.05 Pearson on delta and ≥ 10% DEG recall at **later timepoints** (pillar 1).
2. Rollout stability: 10-step trajectory Pearson ≥ 0.8 vs. observed, where endpoint-only models by definition score 0 after step 1 (pillar 1).
3. On held-out multi-cell compositions (e.g., novel ratios of helper/exhausted T-cells in a co-culture dataset), beat iid cell models by ≥ 0.1 Pearson on delta (pillar 2).
4. Compositional generalization: train on {gene × drug} with a subset, predict unseen combinations, beat motif-ridge by ≥ 0.05 Pearson (pillar 3).
