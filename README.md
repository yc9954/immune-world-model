# IWM — Immune World Model

Action-conditioned, multi-cell latent dynamics simulator for cell-level
immunology. Inspired by Genie 3 / V-JEPA 2. **Not** a single-cell
foundation model or an LLM-over-pathway reasoner — a learned simulator
that runs trajectory and counterfactual rollouts over populations of cells.

## Why this exists

The 2025–2026 cell-AI landscape is crowded but lopsided:

| | Endpoint DE | Trajectory | Multi-cell | Compositional action |
|---|---|---|---|---|
| VCWorld (arXiv 2512.00306) | ✓ | ✗ | ✗ | ✗ |
| STATE / Stack (Arc) | ✓ | ✗ | ✗ | ✗ |
| scFMs (scGPT, Geneformer, UCE, C2S-Scale) | partial | ✗ | ✗ | ✗ |
| TIGON, CellOT, flow matching | population only | ✓ distribution | ✗ | limited |
| PhysiCell / CompuCell3D | N/A | ✓ | ✓ | hand-coded |
| **IWM (this repo)** | ✓ | ✓ | ✓ | ✓ (vector) |

The three pillars no existing model covers together:
1. **Temporal** — predicts cell-state *trajectories*, not endpoints.
2. **Multi-cell** — each cell's next state depends on learned pooling of
   its neighbours and the shared cytokine field.
3. **Compositional actions** — action is a vector over
   `{gene KO, drug SMILES, cytokine dose, neighbour summary}`.

Additionally, the default stack (scVI BSD-3 + MolFormer MIT + UCE MIT)
is **commercial-clean**, unlike STATE / Tahoe-100M / Geneformer.

Full competitor analysis: [`docs/01_differentiation.md`](docs/01_differentiation.md).

## Layout

```
iwm/
  data/         ThreePillarSynth generator + Lightning DataModule
  embeddings/   Factorised action encoders + multi-cell neighbour pool +
                scVI wrapper (optional, for real-data runs)
  models/       L2 WorldModel (transformer), B0–B4 baselines, L3 TissueEnv
  training/     Losses (VICReg + NLL + JEPA-masked + rollout) + Lightning module
  evaluation/   Systema-style metrics, split definitions
  configs/      toy / commercial / research
scripts/        gen_synth · train · eval · rollout_demo
tests/          11 unit tests (pipeline + shape + baseline sanity)
docs/           01_differentiation · 02_architecture · 03_eval_protocol
results/        eval_results.md, RESULTS.md, wm.pt
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional real-data extras: pip install -e .[real] cellxgene-census

# 1. synthetic sanity check
python scripts/gen_synth.py

# 2. small toy (2k cells, 40 ep, ~5 min on M-series MPS)
python scripts/train.py --epochs 40
python scripts/eval.py

# 3. LARGE experiment (8k cells, 50 genes, 64d latent, 6-layer WM, ~20 min MPS)
python scripts/train.py --config iwm/configs/large.yaml --epochs 30
python scripts/eval.py --config iwm/configs/large.yaml \
                      --ckpt results/large_run/wm.pt --out results/large_run

# 4. counterfactual rollout demo
python scripts/rollout_demo.py

# 5. real-data pull (live from CellxGene Census, CC-BY)
python scripts/prepare_real_data.py --n-cells 50000

# 6. tests
pytest
```

## Hardware acceleration

The default `accelerator: auto` in `iwm/configs/toy.yaml` picks MPS on
Apple Silicon and CUDA on Nvidia. CPU threading is also raised to the
physical-core count (`torch.set_num_threads(os.cpu_count())`).

Measured on an Apple M5 (10-core), 5 epochs of the toy:

| Accelerator | it/s | Per-epoch | Notes |
|---|---|---|---|
| CPU (10 threads) | 6.83 | ~7.0s | torch default threads=4 was 1.8× slower |
| **MPS** | **12.04** | **~4.0s** | 1.76× per-step speed-up |

For runs under ~30 seconds total, MPS initialization tax (first-call
kernel JIT) can eat the wins; the gap widens with longer schedules and
larger models. `num_workers=4` + `persistent_workers=true` are set in
`iwm/configs/toy.yaml` so DataLoader prefetch overlaps compute.

**Tuning knobs**: `training.accelerator`, `training.precision`
(`32 | bf16-mixed` — bf16 only on M3+), `training.num_workers`,
`training.threads`.

## Current status — 2026-04-20

Engineering complete. Pipeline runs end-to-end on the synthetic toy.
Scientific validation of pillar-1/2/3 advantages over motif-ridge requires
either a stronger synthetic (shared gene features + non-linear dynamics)
or real time-course Perturb-seq data. The toy results write-up is in
[`results/toy_run/RESULTS.md`](results/toy_run/RESULTS.md).

## Evaluation rules (do not violate)

1. **Always report all four splits.** IID + OOD-gene + OOD-temporal +
   OOD-multicell. A single-split win is not a claim.
2. **Pearson on delta, not raw.** Systema (Nat Biotech 2025) showed raw
   Pearson is leaky. `iwm/evaluation/metrics.py` enforces delta.
3. **Motif-ridge is the opponent.** Ridge is the minimum viable baseline
   and typically beats scFMs on perturbation (Wenkel, Nat Methods 2025).
4. **Rollout ≥ 5 steps** for the OOD-temporal verdict.

## Licensing

- This code: MIT.
- Dependencies used by default: torch (BSD-3), pytorch-lightning
  (Apache-2), scikit-learn (BSD-3), numpy (BSD-3). All commercial-clean.
- Optional: scvi-tools (BSD-3). Clean.
- **Do NOT check in or train against** Tahoe-100M (CC-BY-NC), STATE
  weights (NC research), Geneformer weights (NC), scFoundation weights
  (NC) without a proper license audit — see [`docs/project_license_constraints`](docs/02_architecture.md).

## Real-data integration

Full details: [`docs/04_real_data.md`](docs/04_real_data.md).

| Source | Loader | Status |
|---|---|---|
| **CellxGene Census** (immune) | `iwm/data/real/cellxgene.py` | **Live** — `scripts/prepare_real_data.py` streams without full download |
| Schmidt 2022 (GSE190604) | `iwm/data/real/schmidt2022.py` | Ready, needs manual GEO download |
| Immune Dictionary (SCP2554) | `iwm/data/real/immune_dictionary.py` | Ready, needs manual SCP download |
| CD4+ Perturb-seq (Dec 2025) | `iwm/data/real/cd4_perturb_seq_2025.py` | Ready, awaiting data release |
| Virtual Cell Challenge 2026 | `iwm/evaluation/vcc.py` | Metric harness ready |

```bash
# Stream live from CellxGene (CC-BY, commercial-clean)
python scripts/prepare_real_data.py --n-cells 50000

# Pretrain scVI on the immune subset (BSD-3 L1 encoder)
python scripts/pretrain_scvi.py \
    --adata data_cache/cx_immune_*.h5ad --out results/scvi_run \
    --latent 64 --epochs 20 --accelerator auto
```

## Remaining work

1. Wire `train.py --data real:results/scvi_run/iwm_sample.npz` so the
   main pipeline reads scVI-latent real data directly.
2. Trajectory reconstruction from snapshot Perturb-seq (TIGON / flow
   matching) — needed until true time-course datasets dominate.
3. Gene-feature prior swap — currently a random (G, F) matrix; plug in
   gene2vec / ESM-2 for real transfer.
4. Ablations: drop multi-cell pool; drop gene encoder; drop rollout
   loss. Each ablation should cost pearson_delta on the matching split.
5. Benchmark against STATE / Stack on Virtual Cell Challenge 2026 when
   data drops (research track only — NC license).

## Related work (April 2026 snapshot)

See [`docs/01_differentiation.md`](docs/01_differentiation.md) for the
full table with URLs. Highlights:

- **Genie 3** (DeepMind Aug 2025) — autoregressive transformer for
  real-time world generation. IWM borrows the action-conditioning idea.
- **V-JEPA 2 / V-JEPA 2-AC** (Meta Jun 2025) — joint embedding predictive
  architecture + MPC. IWM's latent-prediction head is JEPA-ish.
- **STATE / Stack** (Arc Institute) — SOTA endpoint perturbation models.
- **VCWorld** (arXiv 2512.00306) — LLM + pathway KG for DE prediction.
  IWM's main structural differentiator is *not being* that.
- **Systema** (Nat Biotech 2025) — mandatory reading on baselines.
