"""Lightning module tying WorldModel + losses + metrics together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ..evaluation.metrics import pearson_delta
from ..models.world_model import WorldModel, WorldModelCfg
from .losses import LossConfig, world_model_losses


@dataclass
class TrainCfg:
    model: WorldModelCfg = field(default_factory=WorldModelCfg)
    loss: LossConfig = field(default_factory=LossConfig)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    rollout_steps: int = 3
    grad_clip: float = 1.0


class IWMLightningModule(pl.LightningModule):
    def __init__(self, cfg: TrainCfg, ablate_neighbor: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.ablate_neighbor = ablate_neighbor
        self.model = WorldModel(cfg.model)
        self._val_cache: Dict[str, list] = {}

    # ------------------------------------------------------------------ fwd

    def _step_forward(self, batch: dict) -> dict:
        z_nbr = batch["neighbor_z_t"]
        if self.ablate_neighbor:
            z_nbr = torch.zeros_like(z_nbr)
        return self.model(
            z_t=batch["z_t"],
            gene_feat=batch["gene_feat"],
            drug_onehot=batch["drug_onehot"],
            cyto_dose=batch["cyto_ext_seq"][:, 0],
            z_nbr=z_nbr,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        out = self._step_forward(batch)
        z_next_true = batch["z_next_seq"][:, 0]
        cyto_true = batch["cyto_total_t"]
        total, parts = world_model_losses(
            out, z_next_true, cyto_true, target_fate=None, cfg=self.cfg.loss,
        )

        # multi-step rollout consistency
        if self.cfg.loss.rollout > 0 and batch["z_next_seq"].size(1) > 1:
            roll = self._rollout_loss(batch)
            total = total + self.cfg.loss.rollout * roll
            parts["l_rl"] = roll.detach()

        self.log_dict({f"train/{k}": v for k, v in parts.items()}, on_step=False, on_epoch=True)
        self.log("train/loss", total, prog_bar=True, on_step=False, on_epoch=True)
        return total

    def _rollout_loss(self, batch: dict) -> torch.Tensor:
        z = batch["z_t"]
        gene = batch["gene_feat"]
        drug = batch["drug_onehot"]
        cyto_seq = batch["cyto_ext_seq"]
        z_next_seq = batch["z_next_seq"]
        neighbor_z_seq = batch["neighbor_z_seq"]    # (B, R, K, d) — ground-truth neighbor states
        R = z_next_seq.size(1)
        total = z.new_zeros(())
        for r in range(R):
            z_nbr = neighbor_z_seq[:, r]            # (B, K, d) at step t+r
            out = self.model(
                z_t=z, gene_feat=gene, drug_onehot=drug,
                cyto_dose=cyto_seq[:, r], z_nbr=z_nbr,
            )
            z = out["mu"]
            total = total + F.mse_loss(z, z_next_seq[:, r])
        return total / R

    # ---------------------------------------------------------------- eval

    def on_validation_epoch_start(self) -> None:
        self._val_cache = {}

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> None:  # noqa: ARG002
        out = self._step_forward(batch)
        z_next_true = batch["z_next_seq"][:, 0]
        split_name = self._split_from_idx(dataloader_idx)
        self._val_cache.setdefault(split_name, []).append(
            {
                "pred": out["mu"].detach().cpu(),
                "true": z_next_true.detach().cpu(),
                "z_t": batch["z_t"].detach().cpu(),
            }
        )

    def _split_from_idx(self, idx: int) -> str:
        return ["iid", "ood_gene", "ood_temporal", "ood_multicell"][idx]

    def on_validation_epoch_end(self) -> None:
        for split, entries in self._val_cache.items():
            pred = torch.cat([e["pred"] for e in entries], dim=0).numpy()
            true = torch.cat([e["true"] for e in entries], dim=0).numpy()
            z_t = torch.cat([e["z_t"] for e in entries], dim=0).numpy()
            delta_pred = pred - z_t
            delta_true = true - z_t
            pr = pearson_delta(delta_pred, delta_true)
            self.log(f"val/{split}_pearson_delta", float(pr), prog_bar=True)

    # ------------------------------------------------------------ optim

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
