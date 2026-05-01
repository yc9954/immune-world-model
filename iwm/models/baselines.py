"""Baselines B0-B4.  Pure numpy / sklearn implementations so that the
headline comparison is architectural, not infrastructure luck.

Every baseline exposes `.fit(train_sample)` and
`.predict(split_sample) -> np.ndarray` of shape (N, T, d) so that a
single evaluation routine can sweep across them.

B0  Identity               predict z_{t+1} = z_t      (no information)
B1  Perturbed mean         predict z_{t+1} = ctrl_mean[t+1] + mean_delta[gene,drug]
B2  Ridge on one-hot       predict Δz = W · [gene‖drug‖cyto]
B3  Motif-bilinear         predict Δz = (U·a_gene) ⊙ (V·z_t) + W·[drug‖cyto]
B4  scVI+ridge             alias for B2 when the latent IS scVI;
                            kept for name-stability with the prior toy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------- helpers

def _gather_step_xy(
    sample: dict,
    step_lo: int,
    step_hi: int,
) -> Dict[str, np.ndarray]:
    """Flatten the trajectory into (N*T, ...) arrays for baseline fitting."""
    z = sample["z_seq"]                    # (N, T+1, d)
    N, T1, d = z.shape
    steps = np.arange(step_lo, step_hi)
    z_t = z[:, steps]                      # (N, S, d)
    z_tp1 = z[:, steps + 1]
    cyto = sample["cyto_ext_seq"][:, steps]    # (N, S, C)
    gene_oh = sample["gene_onehot"][:, None, :].repeat(len(steps), axis=1)
    gene_ft = sample["gene_feat"][:, None, :].repeat(len(steps), axis=1)
    drug = sample["drug_onehot"][:, None, :].repeat(len(steps), axis=1)
    gene_id = sample["gene_id"]
    drug_id = sample["drug_id"]
    return {
        "z_t": z_t.reshape(-1, d),
        "z_tp1": z_tp1.reshape(-1, d),
        "delta": (z_tp1 - z_t).reshape(-1, d),
        "gene_onehot": gene_oh.reshape(-1, gene_oh.shape[-1]),
        "gene_feat": gene_ft.reshape(-1, gene_ft.shape[-1]),
        "drug_onehot": drug.reshape(-1, drug.shape[-1]),
        "cyto": cyto.reshape(-1, cyto.shape[-1]),
        "gene_id": np.repeat(gene_id, len(steps)),
        "drug_id": np.repeat(drug_id, len(steps)),
        "N": N, "S": len(steps), "d": d,
        "steps": steps,
    }


class _BaselineBase:
    name: str = "base"

    def fit(self, train_sample: dict, step_lo: int, step_hi: int) -> None:
        raise NotImplementedError

    def predict_next(self, sample: dict, step: int) -> np.ndarray:
        """Predict z_{step+1} from z_{step}. Returns (N, d)."""
        raise NotImplementedError


# ---------------------------------------------------------------- B0

class Identity(_BaselineBase):
    name = "B0_identity"

    def fit(self, *args, **kwargs) -> None:  # noqa: D401, ANN002
        pass

    def predict_next(self, sample: dict, step: int) -> np.ndarray:
        return sample["z_seq"][:, step].copy()


# ---------------------------------------------------------------- B1

class PerturbedMean(_BaselineBase):
    name = "B1_perturbed_mean"

    def __init__(self) -> None:
        self._delta_by_pair: Dict[tuple, np.ndarray] = {}
        self._global_delta: np.ndarray | None = None
        self._ctrl_delta_by_step: np.ndarray | None = None

    def fit(self, train_sample: dict, step_lo: int, step_hi: int) -> None:
        packed = _gather_step_xy(train_sample, step_lo, step_hi)
        # per (gene_id, drug_id) mean delta
        keys = list(zip(packed["gene_id"].tolist(), packed["drug_id"].tolist()))
        unique = set(keys)
        self._global_delta = packed["delta"].mean(axis=0)
        for k in unique:
            mask = np.array([kk == k for kk in keys])
            self._delta_by_pair[k] = packed["delta"][mask].mean(axis=0)

    def predict_next(self, sample: dict, step: int) -> np.ndarray:
        z_t = sample["z_seq"][:, step]
        N = z_t.shape[0]
        out = np.empty_like(z_t)
        for i in range(N):
            k = (int(sample["gene_id"][i]), int(sample["drug_id"][i]))
            delta = self._delta_by_pair.get(k, self._global_delta)
            out[i] = z_t[i] + delta
        return out


# ---------------------------------------------------------------- B2

class RidgeOneHot(_BaselineBase):
    name = "B2_ridge_onehot"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model: Ridge | None = None
        self._input_fn = None

    def _feature(self, packed: dict) -> np.ndarray:
        return np.concatenate(
            [packed["z_t"], packed["gene_onehot"], packed["drug_onehot"], packed["cyto"]],
            axis=1,
        )

    def fit(self, train_sample: dict, step_lo: int, step_hi: int) -> None:
        packed = _gather_step_xy(train_sample, step_lo, step_hi)
        X = self._feature(packed)
        y = packed["delta"]
        self._model = Ridge(alpha=self.alpha).fit(X, y)

    def predict_next(self, sample: dict, step: int) -> np.ndarray:
        z_t = sample["z_seq"][:, step]
        X = np.concatenate(
            [z_t, sample["gene_onehot"], sample["drug_onehot"], sample["cyto_ext_seq"][:, step]],
            axis=1,
        )
        delta = self._model.predict(X)
        return z_t + delta


# ---------------------------------------------------------------- B3

@dataclass
class MotifBilinearCfg:
    gene_embed_dim: int = 16
    alpha: float = 1.0


class MotifBilinear(_BaselineBase):
    """Bilinear predictor Δz = ((gene_onehot @ U) ⊙ (z_t @ V)) @ W
                             + [drug ‖ cyto] @ W_aux.

    Gene and state are each projected to a K-dim motif space; their
    elementwise product is projected back to the latent space. This rank-K
    factorisation is what gave the prior toy its OOD-gene generalisation
    and is the baseline IWM must beat on temporal and multi-cell splits.

    Fitted with Adam over a small number of steps — not a stand-in for
    scale; simpler ALS variants failed the reshape and added no extra
    expressive power.
    """

    name = "B3_motif_bilinear"

    def __init__(self, cfg: MotifBilinearCfg | None = None) -> None:
        self.cfg = cfg or MotifBilinearCfg()
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.W_aux: np.ndarray | None = None

    def fit(self, train_sample: dict, step_lo: int, step_hi: int) -> None:
        import torch
        import torch.nn.functional as F

        from ..utils.device import pick_device

        device = pick_device("auto")

        packed = _gather_step_xy(train_sample, step_lo, step_hi)
        # Feature-based — enables OOD-gene. Ridge still uses one-hot (its
        # handicap is intentional; see docs/01_differentiation.md).
        gene = torch.from_numpy(packed["gene_feat"]).float().to(device)
        z_t = torch.from_numpy(packed["z_t"]).float().to(device)
        delta = torch.from_numpy(packed["delta"]).float().to(device)
        drug = torch.from_numpy(packed["drug_onehot"]).float().to(device)
        cyto = torch.from_numpy(packed["cyto"]).float().to(device)
        aux = torch.cat([drug, cyto], dim=1)

        K = self.cfg.gene_embed_dim
        F_in = gene.shape[1]
        d = z_t.shape[1]
        A = aux.shape[1]
        gen = torch.Generator().manual_seed(0)
        U = torch.nn.Parameter((torch.randn(F_in, K, generator=gen) * 0.1).to(device))
        V = torch.nn.Parameter((torch.randn(d, K, generator=gen) * 0.1).to(device))
        W = torch.nn.Parameter((torch.randn(K, d, generator=gen) * 0.1).to(device))
        W_aux = torch.nn.Parameter(torch.zeros(A, d, device=device))

        opt = torch.optim.Adam([U, V, W, W_aux], lr=5e-2, weight_decay=1e-4)
        for _ in range(800):
            motif = ((gene @ U) * (z_t @ V)) @ W
            pred = motif + aux @ W_aux
            loss = F.mse_loss(pred, delta)
            opt.zero_grad(); loss.backward(); opt.step()

        self.U = U.detach().cpu().numpy()
        self.V = V.detach().cpu().numpy()
        self.W = W.detach().cpu().numpy()
        self.W_aux = W_aux.detach().cpu().numpy()

    def predict_next(self, sample: dict, step: int) -> np.ndarray:
        z_t = sample["z_seq"][:, step]
        gene = sample["gene_feat"]
        aux = np.concatenate([sample["drug_onehot"], sample["cyto_ext_seq"][:, step]], axis=1)
        motif = ((gene @ self.U) * (z_t @ self.V)) @ self.W
        delta = motif + aux @ self.W_aux
        return z_t + delta


# ---------------------------------------------------------------- B4 alias

class ScviPlusRidge(RidgeOneHot):
    name = "B4_scvi_plus_ridge"
    # Same math as B2; the distinction only matters when L1 is scVI in a real run.
