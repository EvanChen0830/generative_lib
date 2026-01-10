import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from ...core.base_method import BaseMethod

class FlowMatching(BaseMethod):
    """Conditional Flow Matching (CFM) method.
    
    Implements the Optimal Transport path:
    x_t = (1 - t) * x_0 + t * x_1
    Target Velocity v_t = x_1 - x_0
    
    where x_0 ~ N(0, I) (Noise) and x_1 ~ Data.
    """

    def __init__(self, sigma_min: float = 0.0):
        super().__init__()
        self.sigma_min = sigma_min # Minimal noise level if needed

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes CFM MSE loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # x is x_1 (Data)
        x_1 = x
        
        # Sample x_0 (Noise)
        x_0 = torch.randn_like(x_1)
        
        # Sample t ~ Uniform(0, 1)
        # Note: We use continuous time [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Helper for broadcasting t to [B, 1, ...]
        def pad_t(t_tensor):
            return t_tensor.view(-1, *([1] * (x.ndim - 1)))
        
        t_expand = pad_t(t)
        
        # Linear Interpolation (OT path)
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 ? 
        # Simple OT: x_t = (1 - t) * x_0 + t * x_1
        
        # If sigma_min > 0, usually path is slightly different, but let's stick to standard OT-CFM
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Target Velocity
        v_target = x_1 - x_0
        
        # Predict Velocity
        # Pass t as is (0-1 float)
        v_pred = self.predict(model, x_t, t, condition)
        
        # 4. Compute Loss
        # Target v = x1 - (1 - sigma_min) x0
        target = x - (1 - self.sigma_min) * x0
        loss = torch.nn.functional.mse_loss(v_pred, target)
        return {"loss": loss}

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """SNR not typically defined for FM in same way as Diffusion."""
        return torch.ones_like(t) # Placeholder
        return torch.ones_like(t) # Placeholder
