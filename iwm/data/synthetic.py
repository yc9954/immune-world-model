"""Synthetic ground-truth generator that embeds the three IWM pillars.

The generator produces multi-cell trajectories under action sequences so
that models claiming to exploit (1) temporal structure, (2) multi-cell
coupling, and (3) compositional actions can be distinguished from
baselines that do not.

True dynamics (per cell i at step t):

    f = gene_onehot @ gene_features       # rank-F factorisation of gene effect
    z_pre = A z_i(t) + f @ B_f + drug @ C + cyto_total @ D
           + coupling * (neighbor_mean @ E)
    z_i(t+1) = nonlinearity(z_pre) + noise

    a_cyto_total(i,t) = a_cyto_ext(i,t) + k * mean_{j in N(i)} s_j(t)
    s_j(t) = tanh(W_s z_j(t))                         # secretion

Pillars:
  * (1) temporal — the recurrence A^t makes long-horizon prediction
    non-trivial; a single-step endpoint model cannot recover z(T) given z(0).
  * (2) multi-cell — the E term couples neighbors; an iid model
    cannot represent the interaction.
  * (3) compositional — gene effects are mediated through `gene_features`
    (R^{G x F}), so feature-aware models generalise to unseen genes by
    decomposing unseen one-hots into known feature directions, while a
    one-hot-only encoder cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SynthConfig:
    n_cells: int = 2048
    n_cell_types: int = 5
    n_genes_pert: int = 20
    n_drugs: int = 10
    n_cytokines: int = 8
    latent_dim: int = 32
    gene_feature_dim: int = 8               # shared motif space; 0 disables features
    trajectory_len: int = 10
    dt: float = 1.0
    neighbor_k: int = 8
    noise_std: float = 0.05
    coupling_strength: float = 0.6          # weight on the neighbor mean term
    spectral_radius: float = 0.9            # A matrix spectral radius cap
    init_scale: float = 1.5                 # centroid magnitude; larger → z starts in nonlinear regime
    action_scale: float = 1.0              # multiplier on B_f, C, D, E action matrices
    cyto_neighbor_k: float = 0.5            # fraction of cytokine coming from neighbors
    nonlinearity: str = "tanh"              # "none" | "tanh"
    nonlinearity_scale: float = 3.0         # z -> scale * tanh(z / scale)
    seed: int = 0


def _orthogonal(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    """d x k matrix with orthonormal columns when k<=d."""
    g = rng.standard_normal((d, d))
    q, _ = np.linalg.qr(g)
    return q[:, :k]


class ThreePillarSynth:
    """Generates ground-truth trajectories and hands out tensors for training.

    Split semantics are enforced here to keep metric pipelines honest:
      * iid           — random split over (cell, step)
      * ood_gene      — a held-out set of perturbation genes
      * ood_temporal  — later time-points are hidden from the training set
      * ood_multicell — held-out neighbor compositions
                        (realised by swapping in unseen cell-type mixes)
    """

    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        d = cfg.latent_dim
        F = cfg.gene_feature_dim

        # Stable dynamics: spectral radius clipped so trajectories don't diverge.
        raw_A = self.rng.standard_normal((d, d)) / np.sqrt(d)
        u, s, vt = np.linalg.svd(raw_A)
        self.A = (u * np.minimum(s, cfg.spectral_radius)) @ vt

        # Gene features: (G, F) sampled once and held fixed across all cells.
        # Ground-truth gene direction = gene_features[g] @ B_f.
        # Models that see gene_features can generalise to unseen genes;
        # models that see only gene_onehot cannot.
        if F > 0:
            self.gene_features = self.rng.standard_normal((cfg.n_genes_pert, F)) / np.sqrt(F)
            self.B_f = _orthogonal(self.rng, d, F) * 0.9 * cfg.action_scale
        else:
            self.gene_features = np.eye(cfg.n_genes_pert).astype(np.float32)
            self.B_f = _orthogonal(self.rng, d, cfg.n_genes_pert) * 0.6 * cfg.action_scale

        self.C = _orthogonal(self.rng, d, cfg.n_drugs) * 0.5 * cfg.action_scale
        self.D = _orthogonal(self.rng, d, cfg.n_cytokines) * 0.4 * cfg.action_scale
        self.E = self.rng.standard_normal((d, d)) / np.sqrt(d) * 0.3    # neighbor coupling
        self.W_s = self.rng.standard_normal((cfg.n_cytokines, d)) / np.sqrt(d) * 0.5

        # Cell-type priors on initial state.
        self.type_centroids = _orthogonal(self.rng, d, cfg.n_cell_types) * cfg.init_scale

        # Perturbation assignment. Each cell gets one gene KO and one drug
        # (drug can be "none" encoded as a zero vector); we vary cytokines by step.
        self.n_train_genes = int(cfg.n_genes_pert * 0.75)
        self.train_gene_ids = np.arange(self.n_train_genes)
        self.ood_gene_ids = np.arange(self.n_train_genes, cfg.n_genes_pert)

    # ------------------------------------------------------------------ sample

    def _initial_state(self, cell_type: np.ndarray) -> np.ndarray:
        centroids = self.type_centroids[:, cell_type].T    # (N, d)
        noise = self.rng.standard_normal(centroids.shape) * 0.1
        return centroids + noise

    def _secretion(self, z: np.ndarray) -> np.ndarray:
        # z: (N, d); returns (N, cyto)
        return np.tanh(z @ self.W_s.T)

    def _nonlinearity(self, z: np.ndarray) -> np.ndarray:
        if self.cfg.nonlinearity == "none":
            return z
        if self.cfg.nonlinearity == "tanh":
            s = self.cfg.nonlinearity_scale
            return s * np.tanh(z / s)
        raise ValueError(f"unknown nonlinearity: {self.cfg.nonlinearity}")

    def _step(
        self,
        z: np.ndarray,
        gene_onehot: np.ndarray,
        drug_onehot: np.ndarray,
        cyto_ext: np.ndarray,
        neighbors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One synchronous multi-cell step. Returns (z_next, cyto_total)."""

        cyto_from_nbr = self._secretion(z)[neighbors].mean(axis=1)         # (N, C)
        cyto_total = cyto_ext + self.cfg.cyto_neighbor_k * cyto_from_nbr

        neighbor_mean = z[neighbors].mean(axis=1)                           # (N, d)

        # Gene effect through feature space, which gives compositional generalisation.
        gene_feat = gene_onehot @ self.gene_features                        # (N, F)

        z_pre = (
            z @ self.A.T
            + gene_feat @ self.B_f.T
            + drug_onehot @ self.C.T
            + cyto_total @ self.D.T
            + self.cfg.coupling_strength * (neighbor_mean @ self.E.T)
        )
        z_pre = z_pre + self.rng.standard_normal(z_pre.shape) * self.cfg.noise_std
        z_next = self._nonlinearity(z_pre)
        return z_next, cyto_total

    # --------------------------------------------------------------- dataset

    def sample(self, split: str = "iid") -> dict:
        """Produce a dict of numpy arrays for a given split.

        split ∈ {iid, ood_gene, ood_temporal, ood_multicell}.
        """
        cfg = self.cfg
        N = cfg.n_cells

        if split == "ood_multicell":
            # Use only two of the rarer cell-types to build unseen mixes.
            allowed = np.array([cfg.n_cell_types - 2, cfg.n_cell_types - 1])
            cell_type = allowed[self.rng.integers(0, len(allowed), size=N)]
        else:
            cell_type = self.rng.integers(0, cfg.n_cell_types, size=N)

        z0 = self._initial_state(cell_type)

        # Gene KO assignments.
        if split == "ood_gene":
            gene_pool = self.ood_gene_ids
        else:
            gene_pool = self.train_gene_ids
        gene_id = gene_pool[self.rng.integers(0, len(gene_pool), size=N)]
        gene_onehot = np.eye(cfg.n_genes_pert)[gene_id]
        gene_feat_per_cell = self.gene_features[gene_id]         # (N, F)

        # Drug assignments (uniform over all drugs; composition test works
        # via train/test cross of gene × drug).
        drug_id = self.rng.integers(0, cfg.n_drugs, size=N)
        drug_onehot = np.eye(cfg.n_drugs)[drug_id]

        # Neighbors — choose fixed per-cell adjacency sampled from the whole
        # population.  For ood_multicell this mechanically realises a
        # different neighborhood distribution.
        neighbors = np.stack(
            [self.rng.choice(N, size=cfg.neighbor_k, replace=False) for _ in range(N)]
        )

        # Time-varying external cytokine — sinusoidal schedule per cell to
        # ensure the action sequence is non-trivial.
        t_grid = np.arange(cfg.trajectory_len)
        freqs = self.rng.uniform(0.1, 0.5, size=(N, cfg.n_cytokines))
        phases = self.rng.uniform(0, 2 * np.pi, size=(N, cfg.n_cytokines))
        cyto_ext_seq = np.sin(t_grid[None, :, None] * freqs[:, None, :] + phases[:, None, :]) * 0.3
        # shape: (N, T, C)

        z_seq = np.zeros((N, cfg.trajectory_len + 1, cfg.latent_dim))
        cyto_total_seq = np.zeros((N, cfg.trajectory_len, cfg.n_cytokines))
        secretion_seq = np.zeros_like(cyto_total_seq)

        z = z0
        z_seq[:, 0] = z
        for t in range(cfg.trajectory_len):
            cyto_ext_t = cyto_ext_seq[:, t, :]
            secretion_seq[:, t] = self._secretion(z)
            z, cyto_total = self._step(z, gene_onehot, drug_onehot, cyto_ext_t, neighbors)
            z_seq[:, t + 1] = z
            cyto_total_seq[:, t] = cyto_total

        return {
            "z_seq": z_seq.astype(np.float32),                     # (N, T+1, d)
            "gene_onehot": gene_onehot.astype(np.float32),          # (N, n_genes)
            "gene_feat": gene_feat_per_cell.astype(np.float32),     # (N, F)
            "drug_onehot": drug_onehot.astype(np.float32),          # (N, n_drugs)
            "cyto_ext_seq": cyto_ext_seq.astype(np.float32),        # (N, T, C)
            "cyto_total_seq": cyto_total_seq.astype(np.float32),
            "secretion_seq": secretion_seq.astype(np.float32),
            "neighbors": neighbors.astype(np.int64),
            "cell_type": cell_type.astype(np.int64),
            "gene_id": gene_id.astype(np.int64),
            "drug_id": drug_id.astype(np.int64),
            "gene_features_matrix": self.gene_features.astype(np.float32),  # (G, F), shared
        }

    # -------------------------------------------------------------- control

    def control_mean(self, split_sample: dict) -> np.ndarray:
        """Per-step mean of untreated (gene=-1, drug=-1, cyto=0) trajectory.

        Used as the reference for delta metrics.
        """
        cfg = self.cfg
        N = min(512, split_sample["z_seq"].shape[0])
        z = split_sample["z_seq"][:N, 0]
        zero_g = np.zeros((N, cfg.n_genes_pert), dtype=np.float32)
        zero_d = np.zeros((N, cfg.n_drugs), dtype=np.float32)
        zero_c = np.zeros((N, cfg.n_cytokines), dtype=np.float32)
        neigh = split_sample["neighbors"][:N] % N

        # Re-use the sample's own neighbors but restricted to the subset.
        traj = np.zeros((N, cfg.trajectory_len + 1, cfg.latent_dim), dtype=np.float32)
        traj[:, 0] = z
        for t in range(cfg.trajectory_len):
            z_next, _ = self._step(z, zero_g, zero_d, zero_c, neigh)
            traj[:, t + 1] = z_next
            z = z_next
        return traj.mean(axis=0)  # (T+1, d)
