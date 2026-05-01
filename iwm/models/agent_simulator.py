"""L3 — TissueEnv.

Wraps a trained L2 world model into a multi-cell environment with a shared
cytokine field. Exposes rollout + fork + counterfactual primitives. This
is the interface the user-facing product operates on.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from .world_model import WorldModel


@dataclass
class Intervention:
    step: int                                       # when to apply
    target_mask: torch.Tensor                       # (N,) bool over cells
    delta_cyto: Optional[torch.Tensor] = None       # (C,) cytokine dose delta
    gene_feat_override: Optional[torch.Tensor] = None     # (N, F) or (1, F)
    drug_onehot_override: Optional[torch.Tensor] = None


@dataclass
class TissueEnv:
    wm: WorldModel
    z0: torch.Tensor                 # (N, d)
    gene_feat: torch.Tensor          # (N, F) — gene feature vector
    drug_onehot: torch.Tensor        # (N, n_drugs)
    neighbors: torch.Tensor          # (N, K) long
    horizon: int = 48
    dt: float = 1.0
    cyto_base: Optional[torch.Tensor] = None   # (N, C)
    cyto_decay: float = 0.8
    cyto_secretion_gain: float = 0.5
    interventions: List[Intervention] = field(default_factory=list)

    # ------------------------------------------------------------------ api

    def fork(self) -> "TissueEnv":
        return copy.deepcopy(self)

    def with_intervention(self, iv: Intervention) -> "TissueEnv":
        env = self.fork()
        env.interventions = list(env.interventions) + [iv]
        return env

    # ----------------------------------------------------------- internal

    def _cytokine_for(self, t: int, z: torch.Tensor) -> torch.Tensor:
        N = z.size(0)
        if self.cyto_base is None:
            c = torch.zeros(N, self.wm.cfg.n_cytokines, device=z.device)
        else:
            c = self.cyto_base.clone()
        for iv in self.interventions:
            if iv.step == t and iv.delta_cyto is not None:
                delta = iv.delta_cyto.to(z.device).unsqueeze(0)        # (1, C)
                c = c + iv.target_mask[:, None].to(z.device) * delta
        return c

    def _resolve_actions(
        self, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gene = self.gene_feat.clone()
        drug = self.drug_onehot.clone()
        for iv in self.interventions:
            if iv.step == t:
                if iv.gene_feat_override is not None:
                    mask = iv.target_mask.to(gene.device)[:, None]
                    gene = torch.where(mask, iv.gene_feat_override.expand_as(gene), gene)
                if iv.drug_onehot_override is not None:
                    mask = iv.target_mask.to(drug.device)[:, None]
                    drug = torch.where(mask, iv.drug_onehot_override.expand_as(drug), drug)
        return gene, drug

    # ---------------------------------------------------------------- run

    @torch.no_grad()
    def rollout(self) -> torch.Tensor:
        """Returns trajectory (N, horizon+1, d)."""
        self.wm.eval()
        z = self.z0.clone()
        traj = [z]
        for t in range(self.horizon):
            gene_t, drug_t = self._resolve_actions(t)
            cyto_t = self._cytokine_for(t, z)
            z_nbr = z[self.neighbors]                         # (N, K, d)
            pred = self.wm(z, gene_feat=gene_t, drug_onehot=drug_t,
                           cyto_dose=cyto_t, z_nbr=z_nbr, dt=self.dt)
            z = pred["mu"]
            traj.append(z)
        return torch.stack(traj, dim=1)

    @torch.no_grad()
    def rollout_with_logger(self, hook: Callable[[int, torch.Tensor], None]) -> torch.Tensor:
        self.wm.eval()
        z = self.z0.clone()
        hook(0, z)
        traj = [z]
        for t in range(self.horizon):
            gene_t, drug_t = self._resolve_actions(t)
            cyto_t = self._cytokine_for(t, z)
            z_nbr = z[self.neighbors]
            pred = self.wm(z, gene_feat=gene_t, drug_onehot=drug_t,
                           cyto_dose=cyto_t, z_nbr=z_nbr, dt=self.dt)
            z = pred["mu"]
            hook(t + 1, z)
            traj.append(z)
        return torch.stack(traj, dim=1)
