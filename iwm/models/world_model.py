"""L2 — action-conditioned multi-cell latent dynamics transformer.

Input token sequence (each token is a d_tok vector):

    [CLS]  z_t  a_gene  a_drug  a_cyto  a_env  dt_scalar

All action tokens are projected to d_tok so they can attend to each other.
Δt enters as FiLM-modulated scale/shift on the CLS output; this way one
model can handle variable time-step training data without retraining.

Heads off the CLS:
    * z_mu, z_log_sigma  — heteroscedastic next-state Gaussian
    * cyto_hat           — predicted cytokine secretion (auxiliary)
    * fate_logits        — proliferation / death / quiescent (optional)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..embeddings import ActionEncoder, ActionEncoderCfg, NeighborPool


@dataclass
class WorldModelCfg:
    latent_dim: int = 32
    gene_input_dim: int = 8          # F — dim of gene feature vector
    n_drugs: int = 10
    n_cytokines: int = 8
    gene_embed_dim: int = 16
    drug_embed_dim: int = 16
    cyto_embed_dim: int = 16
    env_embed_dim: int = 32
    d_tok: int = 64
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    n_fates: int = 3


class _TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, ffn: int, p: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ffn), nn.GELU(), nn.Dropout(p), nn.Linear(ffn, d)
        )
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_enc = ActionEncoder(
            ActionEncoderCfg(
                gene_input_dim=cfg.gene_input_dim,
                n_drugs=cfg.n_drugs,
                n_cytokines=cfg.n_cytokines,
                gene_embed_dim=cfg.gene_embed_dim,
                drug_embed_dim=cfg.drug_embed_dim,
                cyto_embed_dim=cfg.cyto_embed_dim,
            )
        )
        self.nbr_pool = NeighborPool(cfg.latent_dim, cfg.env_embed_dim)

        d = cfg.d_tok
        self.z_to_tok = nn.Linear(cfg.latent_dim, d)
        self.gene_to_tok = nn.Linear(cfg.gene_embed_dim, d)
        self.drug_to_tok = nn.Linear(cfg.drug_embed_dim, d)
        self.cyto_to_tok = nn.Linear(cfg.cyto_embed_dim, d)
        self.env_to_tok = nn.Linear(cfg.env_embed_dim, d)

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 6, d))          # 6 tokens incl. CLS
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList(
            _TransformerBlock(d, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.ln_f = nn.LayerNorm(d)

        # FiLM on dt (scalar -> (gamma, beta) on CLS)
        self.dt_film = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, 2 * d))

        self.head_mu = nn.Linear(d, cfg.latent_dim)
        self.head_logsigma = nn.Linear(d, cfg.latent_dim)
        self.head_cyto = nn.Linear(d, cfg.n_cytokines)
        self.head_fate = nn.Linear(d, cfg.n_fates)

        # Residual skip — we predict z_{t+1} = z_t + delta so dynamics are
        # easier to learn on near-identity regions.
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    # ------------------------------------------------------------------ fwd

    def forward(
        self,
        z_t: torch.Tensor,               # (B, d_latent)
        gene_feat: torch.Tensor,         # (B, F) — feature vector, not one-hot
        drug_onehot: torch.Tensor,
        cyto_dose: torch.Tensor,
        z_nbr: torch.Tensor,             # (B, K, d_latent)
        dt: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        actions = self.action_enc(gene_feat, drug_onehot, cyto_dose)
        a_env = self.nbr_pool(z_t, z_nbr)

        B = z_t.size(0)
        cls = self.cls.expand(B, -1, -1)
        z_tok = self.z_to_tok(z_t).unsqueeze(1)
        g_tok = self.gene_to_tok(actions["a_gene"]).unsqueeze(1)
        d_tok = self.drug_to_tok(actions["a_drug"]).unsqueeze(1)
        c_tok = self.cyto_to_tok(actions["a_cyto"]).unsqueeze(1)
        e_tok = self.env_to_tok(a_env).unsqueeze(1)
        tokens = torch.cat([cls, z_tok, g_tok, d_tok, c_tok, e_tok], dim=1) + self.pos

        x = tokens
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        cls_out = x[:, 0]

        if isinstance(dt, (int, float)):
            dt_tensor = torch.full((B, 1), float(dt), device=z_t.device)
        else:
            dt_tensor = dt.view(B, 1)
        film = self.dt_film(dt_tensor)
        gamma, beta = film.chunk(2, dim=-1)
        cls_out = cls_out * (1 + gamma) + beta

        delta = self.head_mu(cls_out) * self.delta_scale
        mu = z_t + delta
        log_sigma = self.head_logsigma(cls_out).clamp(-6.0, 2.0)
        cyto_hat = self.head_cyto(cls_out)
        fate_logits = self.head_fate(cls_out)
        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "cyto_hat": cyto_hat,
            "fate_logits": fate_logits,
        }

    # ---------------------------------------------------------------- roll

    @torch.no_grad()
    def rollout(
        self,
        z0: torch.Tensor,
        gene_feat: torch.Tensor,
        drug_onehot: torch.Tensor,
        cyto_dose_seq: torch.Tensor,         # (B, T, C)
        z_nbr_seq: torch.Tensor | None = None,   # (B, T, K, d) — if None, hold constant
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive closed-loop rollout.  Returns (B, T+1, d)."""
        B, T, _ = cyto_dose_seq.shape
        z = z0
        out = [z]
        for t in range(T):
            cyto_t = cyto_dose_seq[:, t]
            if z_nbr_seq is None:
                z_nbr = z.unsqueeze(1).expand(-1, 1, -1)
            else:
                z_nbr = z_nbr_seq[:, t]
            pred = self.forward(z, gene_feat, drug_onehot, cyto_t, z_nbr, dt)
            z = pred["mu"]
            out.append(z)
        return torch.stack(out, dim=1)
