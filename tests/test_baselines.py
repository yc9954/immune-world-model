import numpy as np

from iwm.data.synthetic import SynthConfig, ThreePillarSynth
from iwm.models.baselines import Identity, PerturbedMean, RidgeOneHot, MotifBilinear


def _gen():
    cfg = SynthConfig(n_cells=256, n_cell_types=3, n_genes_pert=8,
                      n_drugs=3, n_cytokines=4, latent_dim=8,
                      trajectory_len=6, neighbor_k=4, seed=0)
    g = ThreePillarSynth(cfg)
    return g, g.sample("iid"), g.sample("ood_gene")


def test_identity():
    g, iid, _ = _gen()
    b = Identity()
    b.fit(iid, 0, 4)
    pred = b.predict_next(iid, step=2)
    assert pred.shape == iid["z_seq"][:, 2].shape


def test_perturbed_mean_generalises_to_same_gene():
    g, iid, _ = _gen()
    b = PerturbedMean()
    b.fit(iid, 0, 4)
    pred = b.predict_next(iid, step=2)
    true = iid["z_seq"][:, 3]
    err = np.linalg.norm(pred - true, axis=-1).mean()
    assert err < np.linalg.norm(iid["z_seq"][:, 3] - iid["z_seq"][:, 2], axis=-1).mean() * 2


def test_ridge_fits():
    g, iid, _ = _gen()
    b = RidgeOneHot()
    b.fit(iid, 0, 4)
    pred = b.predict_next(iid, step=2)
    assert np.isfinite(pred).all()


def test_motif_bilinear_fits():
    g, iid, ood = _gen()
    b = MotifBilinear()
    b.fit(iid, 0, 4)
    pred_iid = b.predict_next(iid, step=2)
    pred_ood = b.predict_next(ood, step=2)
    assert np.isfinite(pred_iid).all()
    assert np.isfinite(pred_ood).all()
