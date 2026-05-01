"""Factorised action encoders.

Each sub-action (gene, drug, cytokine) has its own embedding so the
world model can compose them. Gene input is a **feature vector** — not a
one-hot — which is what gives the model OOD-gene generalisation. For
synthetic runs `gene_feat` comes from a fixed (G, F) matrix; for real
runs it comes from a gene-feature prior (gene2vec, Geneformer tokens,
ESM-2 embedding of the protein product, GO term indicator, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionEncoderCfg:
    gene_input_dim: int = 8       # dim of gene_feat input (F); set to n_genes for 1-hot
    n_drugs: int = 10
    n_cytokines: int = 8
    gene_embed_dim: int = 16
    drug_embed_dim: int = 16
    cyto_embed_dim: int = 16


class ActionEncoder(nn.Module):
    def __init__(self, cfg: ActionEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gene_proj = nn.Linear(cfg.gene_input_dim, cfg.gene_embed_dim, bias=False)
        self.drug_proj = nn.Linear(cfg.n_drugs, cfg.drug_embed_dim, bias=False)
        self.cyto_proj = nn.Sequential(
            nn.Linear(cfg.n_cytokines, cfg.cyto_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.cyto_embed_dim, cfg.cyto_embed_dim),
        )
        nn.init.normal_(self.gene_proj.weight, std=0.05)
        nn.init.normal_(self.drug_proj.weight, std=0.05)

    def forward(
        self,
        gene_feat: torch.Tensor,       # (B, F)
        drug_onehot: torch.Tensor,     # (B, n_drugs)
        cyto_dose: torch.Tensor,       # (B, n_cytokines)
    ) -> dict[str, torch.Tensor]:
        return {
            "a_gene": self.gene_proj(gene_feat),
            "a_drug": self.drug_proj(drug_onehot),
            "a_cyto": self.cyto_proj(cyto_dose),
        }
