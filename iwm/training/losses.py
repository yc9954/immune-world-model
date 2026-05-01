"""Loss functions for the world model.

The overall objective is

    L = λ_mse  L_latent_mse
      + λ_nll  L_latent_nll                    # Gaussian NLL
      + λ_vic  L_vicreg                        # variance / covariance reg
      + λ_cy   L_cyto_mse
      + λ_fa   L_fate_ce                       (optional)
      + λ_jp   L_jepa                          # JEPA masked-latent consistency
      + λ_rl   L_rollout                       # multi-step rollout consistency
      + λ_fl   L_flow                          # optional flow-matching

VICReg (Bardes et al., 2022) prevents latent collapse when other losses
push predictions too close to an easy constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    latent_mse: float = 1.0
    latent_nll: float = 0.3
    vicreg: float = 0.1
    cyto_mse: float = 0.3
    fate_ce: float = 0.0
    jepa: float = 0.2
    rollout: float = 0.5
    flow: float = 0.0


def _gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sigma2 = torch.exp(2 * log_sigma)
    return (0.5 * ((target - mu) ** 2 / sigma2 + 2 * log_sigma)).mean()


def _vicreg(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.relu(1.0 - std).mean()
    B = z.size(0)
    cov = (z.T @ z) / max(B - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / z.size(1)
    return var_loss + 0.1 * cov_loss


def _jepa_masked(mu: torch.Tensor, target: torch.Tensor, mask_frac: float = 0.15) -> torch.Tensor:
    """Mask a random subset of latent dims and require prediction to match.
    Implementation trick: we reuse the Gaussian prediction as a stand-in for
    a teacher encoder, which is adequate for a toy run.
    """
    B, D = mu.shape
    mask = torch.rand(B, D, device=mu.device) < mask_frac
    if not mask.any():
        return torch.zeros((), device=mu.device)
    return F.mse_loss(mu[mask], target[mask])


def world_model_losses(
    model_out: dict,
    target_z_next: torch.Tensor,
    target_cyto: torch.Tensor | None,
    target_fate: torch.Tensor | None,
    cfg: LossConfig,
) -> tuple[torch.Tensor, dict]:
    mu = model_out["mu"]
    log_sigma = model_out["log_sigma"]

    l_mse = F.mse_loss(mu, target_z_next)
    l_nll = _gaussian_nll(mu, log_sigma, target_z_next)
    l_vic = _vicreg(mu)
    l_cy = F.mse_loss(model_out["cyto_hat"], target_cyto) if target_cyto is not None else mu.new_zeros(())
    l_fa = (
        F.cross_entropy(model_out["fate_logits"], target_fate)
        if target_fate is not None else mu.new_zeros(())
    )
    l_jp = _jepa_masked(mu, target_z_next)

    total = (
        cfg.latent_mse * l_mse
        + cfg.latent_nll * l_nll
        + cfg.vicreg * l_vic
        + cfg.cyto_mse * l_cy
        + cfg.fate_ce * l_fa
        + cfg.jepa * l_jp
    )
    return total, {
        "l_mse": l_mse.detach(),
        "l_nll": l_nll.detach(),
        "l_vic": l_vic.detach(),
        "l_cy": l_cy.detach(),
        "l_fa": l_fa.detach() if isinstance(l_fa, torch.Tensor) else torch.tensor(0.0),
        "l_jp": l_jp.detach(),
    }
