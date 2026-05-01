"""PyTorch Lightning DataModule wrapping ThreePillarSynth.

Keeps the four splits (iid / ood_gene / ood_temporal / ood_multicell) alive
as independent DataLoaders so evaluation can report per-split metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .synthetic import SynthConfig, ThreePillarSynth


class StepDataset(Dataset):
    """Samples a random (cell, step) pair and returns
    (z_t, z_{t+1}, actions, neighbor_z_t, cyto_ext_t).

    For ood_temporal we constrain the accessible step range.
    """

    def __init__(
        self,
        sample: dict,
        step_lo: int,
        step_hi: int,
        rollout_steps: int = 1,
    ) -> None:
        self.sample = sample
        self.step_lo = step_lo
        self.step_hi = step_hi
        self.rollout_steps = rollout_steps
        self.n_cells = sample["z_seq"].shape[0]
        self.traj_len = sample["z_seq"].shape[1] - 1
        # list of (cell_idx, t_start) pairs such that t_start .. t_start+rollout_steps are valid
        self._pairs = [
            (i, t)
            for i in range(self.n_cells)
            for t in range(step_lo, min(step_hi, self.traj_len - rollout_steps + 1))
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict:
        i, t = self._pairs[idx]
        s = self.sample
        neigh = s["neighbors"][i]
        out = {
            "z_t":    torch.from_numpy(s["z_seq"][i, t]),
            "z_next_seq": torch.from_numpy(
                s["z_seq"][i, t + 1 : t + 1 + self.rollout_steps]
            ),                                                   # (R, d)
            "gene_onehot": torch.from_numpy(s["gene_onehot"][i]),
            "gene_feat": torch.from_numpy(s["gene_feat"][i]),    # (F,)
            "drug_onehot": torch.from_numpy(s["drug_onehot"][i]),
            "cyto_ext_seq": torch.from_numpy(
                s["cyto_ext_seq"][i, t : t + self.rollout_steps]
            ),                                                   # (R, C)
            "cyto_total_t": torch.from_numpy(s["cyto_total_seq"][i, t]),
            "secretion_t": torch.from_numpy(s["secretion_seq"][i, t]),
            "neighbor_z_t": torch.from_numpy(s["z_seq"][neigh, t]),   # (K, d)
            "neighbor_z_seq": torch.from_numpy(
                np.stack([s["z_seq"][neigh, t + r] for r in range(self.rollout_steps)], axis=0)
            ),                                                          # (R, K, d)
            "cell_type": int(s["cell_type"][i]),
            "gene_id": int(s["gene_id"][i]),
            "drug_id": int(s["drug_id"][i]),
        }
        return out


@dataclass
class DataModuleCfg:
    synth: SynthConfig = field(default_factory=SynthConfig)
    batch_size: int = 256
    rollout_steps: int = 3
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    ood_temporal_split_at: int = 6          # train uses t < this; eval uses t >= this


class IWMDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._samples: Dict[str, dict] = {}
        self._gen: Optional[ThreePillarSynth] = None

    # ------------------------------------------------------------------ setup

    def prepare_data(self) -> None:  # one-process; fine for in-memory synth
        self._gen = ThreePillarSynth(self.cfg.synth)
        self._samples["train"] = self._gen.sample("iid")
        self._samples["iid"] = self._gen.sample("iid")
        self._samples["ood_gene"] = self._gen.sample("ood_gene")
        self._samples["ood_temporal"] = self._samples["iid"]         # reuse cells
        self._samples["ood_multicell"] = self._gen.sample("ood_multicell")

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        if self._gen is None:
            self.prepare_data()

    # ------------------------------------------------------------- loaders

    def _dataset(self, split: str) -> StepDataset:
        tl = self.cfg.synth.trajectory_len
        split_at = self.cfg.ood_temporal_split_at
        if split == "train":
            return StepDataset(self._samples["train"], 0, split_at,
                               rollout_steps=self.cfg.rollout_steps)
        if split == "iid":
            return StepDataset(self._samples["iid"], 0, split_at,
                               rollout_steps=1)
        if split == "ood_gene":
            return StepDataset(self._samples["ood_gene"], 0, split_at,
                               rollout_steps=1)
        if split == "ood_temporal":
            return StepDataset(self._samples["ood_temporal"], split_at, tl,
                               rollout_steps=1)
        if split == "ood_multicell":
            return StepDataset(self._samples["ood_multicell"], 0, split_at,
                               rollout_steps=1)
        raise ValueError(split)

    def _loader_kwargs(self) -> dict:
        kwargs = dict(
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        if self.cfg.num_workers > 0:
            kwargs["persistent_workers"] = self.cfg.persistent_workers
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset("train"),
            shuffle=True, drop_last=True, **self._loader_kwargs(),
        )

    def val_dataloader(self) -> Dict[str, DataLoader]:
        return {
            name: DataLoader(
                self._dataset(name), shuffle=False, **self._loader_kwargs(),
            )
            for name in ("iid", "ood_gene", "ood_temporal", "ood_multicell")
        }

    # ---------------------------------------------------------------- api

    @property
    def sample(self) -> Dict[str, dict]:
        return self._samples

    @property
    def generator(self) -> ThreePillarSynth:
        assert self._gen is not None
        return self._gen
