from typing import Dict, Optional
import torch
import torch.nn as nn
from .gaussian_diffusion import GaussianDiffusion

class CFGDiffusion(GaussianDiffusion):
    """Gaussian Diffusion with Classifier-Free Guidance support (Dual Loss).

    `unconditional_value` should be outside the support of real conditioning
    values. For example, if class labels are encoded as `0` and `1`, using
    `0.0` as the unconditional token will alias the unconditional branch with
    class `0` and degrade CFG behavior.
    """

    def __init__(
        self,
        unconditional_value: float = -1.0,
        schedule: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ):
        super().__init__(schedule, timesteps, beta_start, beta_end, prediction_type)
        # This scalar is broadcast to the full condition tensor during CFG.
        self.unconditional_value = unconditional_value

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes dual loss (Conditional + Unconditional)."""
        loss_dict = super().compute_loss(model, x, condition)
        cond_loss = loss_dict['loss']

        # Unconditional Loss
        # Create unconditional condition matching batch size and device
        # If condition is [B, D], we need [B, D] filled with unconditional_value
        if condition is not None:
            uncond_condition = torch.full_like(condition, self.unconditional_value)
        else:
            # If no condition provided, CFG doesn't make sense, but fail gracefully?
             raise ValueError("CFGDiffusion requires a condition to be passed.")

        loss_dict_uncond = super().compute_loss(model, x, uncond_condition)
        uncond_loss = loss_dict_uncond['loss']

        total_loss = cond_loss + uncond_loss

        return {
            "loss": total_loss,
            "cond_loss": cond_loss,
            "uncond_loss": uncond_loss
        }
