# Real-data integration — 2026-04-20

This document maps each real-data source to its loader, license, and
readiness. Updates should keep these four columns honest.

## Loaders

| Source | Loader | License | Ready? |
|---|---|---|---|
| CellxGene Census (immune subset) | `iwm/data/real/cellxgene.py` → `stream_immune_subset` | CC-BY 4.0 | **Works** — `scripts/prepare_real_data.py` streams live. |
| Schmidt et al. 2022 (GSE190604) | `iwm/data/real/schmidt2022.py` | CC-BY (GEO standard) | Requires manual GEO download (~6 GB). Loader ready. |
| Immune Dictionary (SCP2554) | `iwm/data/real/immune_dictionary.py` | SCP DUA (NC research) | Requires manual SCP download. Loader + cytokine one-hot helper ready. |
| CD4+ Perturb-seq (bioRxiv Dec 2025) | `iwm/data/real/cd4_perturb_seq_2025.py` | CC-BY (preprint) | Requires manual download; link in module docstring. Loader ready. |
| Tahoe-100M | — | CC-BY-NC | **NOT wired** — 429 GB, non-commercial. Deferred to research track. |

## Pipeline

```
raw real data (AnnData)
        │
        ▼
scripts/pretrain_scvi.py         # BSD-3 encoder, commercial-clean L1
        │
        ▼
{ scVI model dir, latent.npy, iwm_sample.npz }
        │
        ▼
iwm.data.real.base.adata_to_iwm_sample   # AnnData → IWM sample dict
        │
        ▼
scripts/train.py --data-source real   # (to be wired — see TODO below)
scripts/eval.py
```

## Commands for a clean end-to-end real run

```bash
# 1. pull a 50k-cell immune subset from CellxGene Census (commercial-clean)
python scripts/prepare_real_data.py --n-cells 50000 --tissue blood

# 2. train scVI for the L1 encoder (MPS on M-series)
python scripts/pretrain_scvi.py \
    --adata data_cache/cx_immune_50000_blood_*.h5ad \
    --out results/scvi_run \
    --latent 64 --epochs 20 --batch-key donor_id

# 3. downstream IWM training on the scVI latents (TODO: wire --iwm-sample flag)
# 4. VCC eval when dataset is available
#    python scripts/eval_vcc.py --train vcc_train.h5ad --val vcc_val.h5ad
```

## What's still open

1. **DataModule real-mode switch** — train.py currently defaults to
   synthetic; a `--data npz:results/scvi_run/iwm_sample.npz` CLI flag
   needs to read IWM sample dicts saved by `pretrain_scvi.py`.
2. **Trajectory assembly from time-course data** — synthetic generates
   T-step trajectories natively; real Perturb-seq is mostly endpoint.
   Pseudotime / TIGON / flow-matching can synthesise T>1 trajectories
   from snapshot data — see the "flow matching for cells"
   literature referenced in the memory landscape file.
3. **Gene-feature prior choice** — current default is learned per
   corpus. For transfer across datasets, use gene2vec (commercial-clean,
   Du et al. 2019) or ESM-2 gene-product embeddings.
4. **VCC 2026 dataset** — pending release. The hook in
   `iwm/evaluation/vcc.py` matches the 2025 data spec; will likely
   work with 2026 data with minor key renames.

## License sanity check

Default stack (all commercial-clean):

| Component | License |
|---|---|
| CellxGene Census data | CC-BY 4.0 |
| scVI (scvi-tools) | BSD-3-Clause |
| MolFormer (drug encoder) | MIT |
| gene2vec | MIT |
| torch / Lightning / numpy / sklearn | BSD / Apache |
| This repo (iwm) | MIT |

Research-track extras (blocked for product shipping):
- Geneformer (non-commercial)
- scFoundation (non-commercial)
- STATE / Stack (Arc research license)
- Tahoe-100M (CC-BY-NC)
- Immune Dictionary (SCP DUA)
