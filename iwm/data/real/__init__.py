"""Real-data connectors for IWM.

Each loader returns a standardised `RealDataBundle` so downstream
components (scVI pretrain, DataModule, eval) treat it the same way. The
synthetic ThreePillarSynth and real loaders share the same output keys
where applicable: `z_seq`, `gene_feat`, `drug_onehot`, `neighbors`, etc.

Status (Apr 2026):
  cellxgene  — actually pulls live data via the Census API (works in-session)
  schmidt2022 — documented loader; requires user to download GSE190604 first
  immune_dictionary — documented loader; SCP2554 on Single Cell Portal
  tahoe100m — documented loader; 429 GB HuggingFace dataset
"""

from .base import RealDataBundle, adata_to_iwm_sample

__all__ = ["RealDataBundle", "adata_to_iwm_sample"]
