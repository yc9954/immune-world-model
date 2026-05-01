"""Shared shapes and helpers for real-data loaders.

The IWM code base treats synthetic and real data interchangeably — both
ultimately produce a flat `sample` dict with keys:

    z_seq (N, T+1, d)        # latent states at each step (from scVI / STATE-SE)
    gene_feat (N, F)         # gene-feature vector (gene2vec / ESM-2 / learned)
    drug_onehot (N, n_drugs)
    cyto_ext_seq (N, T, C)   # external cytokine dose schedule
    cyto_total_seq (N, T, C) # observed / reconstructed total cytokine
    neighbors (N, K)
    cell_type (N,)
    gene_id (N,)
    drug_id (N,)

For endpoint-only real datasets (most Perturb-seq) we set T=1 and treat
the pair (control_latent, perturbed_latent) as a single step. The
trajectory pillar only lights up on time-course data (Immune Dictionary
has minimal temporal info; CellxGene Census has mostly snapshot).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RealDataBundle:
    """Output of every real-data loader — minimal interface."""

    # Cell-level arrays
    z_seq: np.ndarray                    # (N, T+1, d) latent trajectory
    gene_feat: np.ndarray                # (N, F) feature vector per cell's perturbation
    drug_onehot: np.ndarray              # (N, n_drugs)
    cyto_ext_seq: np.ndarray             # (N, T, C)
    cyto_total_seq: np.ndarray           # (N, T, C) — fallback = cyto_ext_seq
    neighbors: np.ndarray                # (N, K) cell indices
    cell_type: np.ndarray                # (N,) categorical
    gene_id: np.ndarray                  # (N,)
    drug_id: np.ndarray                  # (N,)
    gene_features_matrix: np.ndarray     # (G, F)

    # Metadata
    source: str = ""
    license: str = ""
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def as_sample_dict(self) -> dict:
        return {
            "z_seq": self.z_seq.astype(np.float32),
            "gene_onehot": np.eye(self.gene_features_matrix.shape[0])[self.gene_id].astype(np.float32),
            "gene_feat": self.gene_feat.astype(np.float32),
            "drug_onehot": self.drug_onehot.astype(np.float32),
            "cyto_ext_seq": self.cyto_ext_seq.astype(np.float32),
            "cyto_total_seq": self.cyto_total_seq.astype(np.float32),
            "secretion_seq": np.zeros_like(self.cyto_total_seq, dtype=np.float32),
            "neighbors": self.neighbors.astype(np.int64),
            "cell_type": self.cell_type.astype(np.int64),
            "gene_id": self.gene_id.astype(np.int64),
            "drug_id": self.drug_id.astype(np.int64),
            "gene_features_matrix": self.gene_features_matrix.astype(np.float32),
        }


def _knn_graph(z: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Approximate k-NN by random projection + pairwise distances on sample.

    For real runs replace with scanpy.pp.neighbors or faiss. This is the
    cheap default so a loader always has a valid `neighbors` array.
    """
    N = z.shape[0]
    rng = np.random.default_rng(seed)
    if N <= k + 1:
        # Degenerate — cycle the whole population.
        idx = np.tile(np.arange(N), (N, 1))[:, 1:k + 1]
        return idx
    # random 256-cell subsample to estimate distances against
    probe = rng.choice(N, size=min(256, N), replace=False)
    probe_z = z[probe]
    # pairwise squared distances to the probe
    d2 = ((z[:, None, :] - probe_z[None, :, :]) ** 2).sum(axis=-1)
    nearest_probe = d2.argsort(axis=-1)[:, :k]
    return probe[nearest_probe]


def scvi_gene_features(
    adata: "object",
    scvi_model: "object",
    perturbation_key: Optional[str],
    n_components: int = 32,
) -> np.ndarray:
    """Build per-perturbation gene feature matrix from scVI decoder loadings.

    Uses the mean normalised expression of each perturbation group as its
    feature vector, projected onto the top PCs of the scVI gene loading space.
    This is far more informative than random features and enables OOD
    generalisation to unseen perturbations that share gene-expression structure.

    Returns (G, n_components) float32 array, one row per unique perturbation.
    """
    from sklearn.decomposition import PCA

    Z = np.asarray(adata.obsm["X_scVI"])
    if perturbation_key and perturbation_key in adata.obs:
        pert = adata.obs[perturbation_key].astype(str).values
    else:
        pert = np.array(["control"] * len(Z))
    unique_perts = sorted(set(pert))
    G = len(unique_perts)

    # Mean normalised expression per perturbation group  (G, n_genes)
    expr = np.array(scvi_model.get_normalized_expression(adata, return_numpy=True))
    group_expr = np.zeros((G, expr.shape[1]), dtype=np.float32)
    for i, p in enumerate(unique_perts):
        mask = pert == p
        group_expr[i] = expr[mask].mean(axis=0)

    # PCA to compress to n_components
    k = min(n_components, G - 1, group_expr.shape[1] - 1)
    if k < 1:
        return group_expr.astype(np.float32)
    pca = PCA(n_components=k, random_state=0)
    feats = pca.fit_transform(group_expr).astype(np.float32)
    # zero-pad if fewer components than requested
    if feats.shape[1] < n_components:
        feats = np.pad(feats, ((0, 0), (0, n_components - feats.shape[1])))
    return feats


def adata_to_iwm_sample(
    adata: "object",
    latent_key: str = "X_scVI",
    perturbation_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
    gene_features: Optional[np.ndarray] = None,
    scvi_model: Optional["object"] = None,
    gene_feature_dim: int = 32,
    n_cytokines: int = 86,
    neighbor_k: int = 8,
    seed: int = 0,
) -> RealDataBundle:
    """Convert an AnnData object into a RealDataBundle.

    Assumes the latent representation is present in `adata.obsm[latent_key]`.
    For single-timepoint data (most real Perturb-seq), T = 1 and the
    trajectory is (control_mean, perturbed_latent) pairs per perturbation.

    If `scvi_model` is provided, gene features are derived from mean
    normalised expression per perturbation group (PCA-compressed). Otherwise
    random features are used as a fallback.
    """
    import anndata  # noqa: F401   # soft dependency

    Z = np.asarray(adata.obsm[latent_key])
    N, d = Z.shape

    if perturbation_key and perturbation_key in adata.obs:
        pert = adata.obs[perturbation_key].astype(str).values
    else:
        pert = np.array(["control"] * N)
    unique_perts = sorted(set(pert))
    pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
    gene_id = np.array([pert_to_idx[p] for p in pert])
    G = len(unique_perts)

    # Gene features: scVI-derived > provided > random fallback
    if gene_features is None:
        if scvi_model is not None:
            gene_features = scvi_gene_features(
                adata, scvi_model, perturbation_key, n_components=gene_feature_dim
            )
        else:
            rng = np.random.default_rng(seed)
            F = gene_feature_dim
            gene_features = rng.standard_normal((G, F)).astype(np.float32) / np.sqrt(F)
    gene_feat_per_cell = gene_features[gene_id]

    # Cell type — either present or fold to one bucket.
    if cell_type_key and cell_type_key in adata.obs:
        ct = adata.obs[cell_type_key].astype(str).values
        ct_map = {c: i for i, c in enumerate(sorted(set(ct)))}
        cell_type = np.array([ct_map[c] for c in ct])
    else:
        cell_type = np.zeros(N, dtype=np.int64)

    # Control centroid — per cell type if available, otherwise global.
    ctrl_mask = pert == "control"
    if ctrl_mask.any():
        ctrl_centroid_global = Z[ctrl_mask].mean(axis=0)
    else:
        ctrl_centroid_global = Z.mean(axis=0)

    # Build a single-step trajectory (control -> observed).
    z_seq = np.stack([
        np.broadcast_to(ctrl_centroid_global, Z.shape),
        Z,
    ], axis=1).astype(np.float32)                                       # (N, 2, d)

    # Default cytokine channels empty (no treatment info); populate from
    # metadata if a cytokine column is present.
    cyto = np.zeros((N, 1, n_cytokines), dtype=np.float32)

    drug_onehot = np.zeros((N, 1), dtype=np.float32)       # no drug info by default
    drug_id = np.zeros(N, dtype=np.int64)

    neighbors = _knn_graph(Z, neighbor_k, seed=seed)

    return RealDataBundle(
        z_seq=z_seq,
        gene_feat=gene_feat_per_cell,
        drug_onehot=drug_onehot,
        cyto_ext_seq=cyto,
        cyto_total_seq=cyto,
        neighbors=neighbors,
        cell_type=cell_type,
        gene_id=gene_id,
        drug_id=drug_id,
        gene_features_matrix=gene_features,
        meta={"n_perturbations": G, "latent_dim": d},
    )
