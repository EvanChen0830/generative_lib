import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from ...core.base_method import BaseMethod

class GaussianDiffusion(BaseMethod):
    """Gaussian Diffusion (DDPM) method.

    Standard variance preserving (VP) diffusion process.
    """

    def __init__(
        self,
        schedule: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon", # 'epsilon' (noise) or 'sample' (x_0) or 'v_prediction'
    ):
        super().__init__()
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        # Define betas and alphas
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
             # Simple cosine schedule approximation
             self.betas = torch.linspace(beta_start, beta_end, timesteps) # Placeholder for now
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Helper variables (register as buffers to save with state_dict)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes DDPM MSE loss."""
        batch_size = x.shape[0]
        device = x.device

        # 1. Sample Time t ~ Uniform(0, T)
        t_idx = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 2. Add Noise
        noise = torch.randn_like(x)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_idx].view(-1, *([1] * (x.ndim - 1)))
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, *([1] * (x.ndim - 1)))
        
        x_t = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
        
        # 3. Predict Noise
        # User Model is expected to handle t.
        # If user model expects embeddings (indices), we should pass indices.
        # If user model expects continuous (0-1), we should pass float.
        # Standard DDPM passes indices [0, 999].
        # Our BaseMethod `predict` wrapper handles broadcasting t if it's scalar.
        # Here t_idx is [B].
        
        # We pass t_idx (batch of integers) to the model.
        # The User Model (two_moons_diffusion.py) handles conversion.
        pred_noise = self.predict(model, x_t, t_idx, condition)
        
        # 4. Compute Loss
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x
        else:
             raise NotImplementedError(f"Pred type {self.prediction_type} not supp")

        loss = torch.nn.functional.mse_loss(pred_noise, target)
        
        return {"loss": loss}

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """SNR = alpha_bar / (1 - alpha_bar)."""
        # Ensure t is integer index
        if t.dtype == torch.float:
             t_idx = (t * self.timesteps).long().clamp(0, self.timesteps - 1)
        else:
             t_idx = t.long()
             
        alpha_bar = self.alphas_cumprod[t_idx]
        return alpha_bar / (1 - alpha_bar)
