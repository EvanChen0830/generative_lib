import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from ...core.base_method import BaseMethod

from scipy.optimize import linear_sum_assignment

class FlowMatching(BaseMethod):
    """Conditional Flow Matching (CFM) method.
    
    Implements the Optimal Transport path:
    x_t = (1 - t) * x_0 + t * x_1
    Target Velocity v_t = x_1 - x_0
    
    where x_0 ~ N(0, I) (Noise) and x_1 ~ Data.
    """

    def __init__(self, sigma_min: float = 0.0, ot_minibatch: bool = False):
        super().__init__()
        self.sigma_min = sigma_min # Minimal noise level
        self.ot_minibatch = ot_minibatch

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None, x_0: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes CFM MSE loss."""
        batch_size = x.shape[0]
        device = x.device
        
        # x is x_1 (Data)
        x_1 = x
        
        # Sample x_0 (Noise) if not provided
        if x_0 is None:
             x_0 = torch.randn_like(x_1)
        
        # Optimal Transport Minibatch Coupling
        if self.ot_minibatch:
            with torch.no_grad():
                # Flatten for distance computation
                x_0_flat = x_0.reshape(batch_size, -1)
                x_1_flat = x_1.reshape(batch_size, -1)
                
                # Compute Squared Euclidean Cost Matrix
                # dist[i, j] = ||x_0[i] - x_1[j]||^2
                dist = torch.cdist(x_0_flat, x_1_flat, p=2) ** 2
                
                # Solve Assignment
                # cpu().numpy() might be slow but necessary for scipy
                row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
                
                # Reorder x_1 to align with x_0 (Minimizes sum of distances)
                # x_0[i] pairs with x_1[col_ind[i]]
                x_1 = x_1[col_ind]

        # Sample t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Helper for broadcasting t to [B, 1, ...]
        def pad_t(t_tensor):
            return t_tensor.view(-1, *([1] * (x.ndim - 1)))
        
        t_expand = pad_t(t)
        
        # Linear Interpolation
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        x_t = (1 - (1 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1
        
        # Target Velocity
        # v_target = x_1 - (1 - sigma_min) * x_0
        v_target = x_1 - (1 - self.sigma_min) * x_0
        
        # Predict Velocity
        v_pred = self.predict(model, x_t, t, condition)
        
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        return {"loss": loss}

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """SNR not typically defined for FM in same way as Diffusion."""
        return torch.ones_like(t) # Placeholder
