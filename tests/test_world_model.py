import torch

from iwm.models.world_model import WorldModel, WorldModelCfg


def _make_wm(d: int = 8, F: int = 4) -> WorldModel:
    cfg = WorldModelCfg(
        latent_dim=d, gene_input_dim=F, n_drugs=3, n_cytokines=4,
        gene_embed_dim=8, drug_embed_dim=8, cyto_embed_dim=8,
        env_embed_dim=8, d_tok=16, n_layers=2, n_heads=2, ffn_dim=32,
    )
    return WorldModel(cfg)


def test_forward_shapes():
    wm = _make_wm()
    B, K = 4, 3
    out = wm(
        z_t=torch.randn(B, 8),
        gene_feat=torch.randn(B, 4),
        drug_onehot=torch.eye(3)[torch.arange(B) % 3],
        cyto_dose=torch.randn(B, 4),
        z_nbr=torch.randn(B, K, 8),
    )
    assert out["mu"].shape == (B, 8)
    assert out["log_sigma"].shape == (B, 8)
    assert out["cyto_hat"].shape == (B, 4)
    assert out["fate_logits"].shape == (B, 3)


def test_residual_stability():
    wm = _make_wm()
    z = torch.randn(4, 8)
    out = wm(
        z_t=z,
        gene_feat=torch.zeros(4, 4),
        drug_onehot=torch.eye(3)[torch.arange(4) % 3],
        cyto_dose=torch.zeros(4, 4),
        z_nbr=torch.zeros(4, 3, 8),
    )
    diff = (out["mu"] - z).norm(dim=-1).mean().item()
    assert diff < 1.5, f"init residual too large: {diff}"


def test_rollout_shape():
    wm = _make_wm()
    B, K, T = 3, 3, 4
    z0 = torch.randn(B, 8)
    traj = wm.rollout(
        z0=z0,
        gene_feat=torch.randn(B, 4),
        drug_onehot=torch.eye(3)[:B],
        cyto_dose_seq=torch.randn(B, T, 4),
    )
    assert traj.shape == (B, T + 1, 8)
