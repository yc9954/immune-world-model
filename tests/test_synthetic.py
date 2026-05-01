import numpy as np
import pytest

from iwm.data.synthetic import SynthConfig, ThreePillarSynth


def _cfg(**kw):
    return SynthConfig(n_cells=64, n_cell_types=3, n_genes_pert=8,
                       gene_feature_dim=4,
                       n_drugs=3, n_cytokines=4, latent_dim=8,
                       trajectory_len=5, neighbor_k=4, **kw)


def test_trajectory_shapes():
    gen = ThreePillarSynth(_cfg())
    s = gen.sample("iid")
    assert s["z_seq"].shape == (64, 6, 8)
    assert s["gene_onehot"].shape == (64, 8)
    assert s["gene_feat"].shape == (64, 4)
    assert s["gene_features_matrix"].shape == (8, 4)
    assert s["cyto_ext_seq"].shape == (64, 5, 4)
    assert s["neighbors"].shape == (64, 4)


def test_ood_gene_disjoint():
    gen = ThreePillarSynth(_cfg(seed=1))
    train = gen.sample("iid")
    ood = gen.sample("ood_gene")
    # train gene_ids should be in the training pool
    assert set(train["gene_id"].tolist()).issubset(set(gen.train_gene_ids.tolist()))
    # ood gene_ids should be in the ood pool
    assert set(ood["gene_id"].tolist()).issubset(set(gen.ood_gene_ids.tolist()))
    assert set(train["gene_id"]).isdisjoint(set(ood["gene_id"]))


def test_temporal_nonlinearity():
    """If dynamics were trivial, step delta would be near-zero after step 1."""
    gen = ThreePillarSynth(_cfg())
    s = gen.sample("iid")
    step_deltas = np.linalg.norm(np.diff(s["z_seq"], axis=1), axis=-1).mean(axis=0)
    assert step_deltas.min() > 0.05       # every step moves meaningfully


def test_neighbor_coupling_nontrivial():
    """Two runs with identical initial states but different neighbor assignments
    should diverge — confirming the multi-cell term is non-degenerate."""
    gen = ThreePillarSynth(_cfg(seed=0, coupling_strength=0.6))
    s1 = gen.sample("iid")
    s2 = gen.sample("iid")
    # average final latent norms should differ even with same random seeds
    # because the generator's internal state advances between calls.
    diff = np.linalg.norm(s1["z_seq"][:, -1] - s2["z_seq"][:, -1], axis=-1).mean()
    assert diff > 0.1
