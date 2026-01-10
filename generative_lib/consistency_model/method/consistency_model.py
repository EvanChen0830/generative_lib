import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from ...core.base_method import BaseMethod

class ConsistencyModel(BaseMethod):
    """Consistency Models (CM) method.
    
    Implements Consistency Training or Distillation.
    Basic idea: f(x_t, t) = f(x_{t'}, t') = x_0
    
    For now, implementing simplified Discrete Consistency Distillation (CD) loss stub.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Computes Consistency Training Loss.
        
        Args:
            model: Online model (theta).
            x: Data [B, D].
            condition: Condition [B, C].
            **kwargs: Must contain 'target_model' (theta_minus) and optionally 'steps' (N).
        """
        target_model = kwargs.get("target_model")
        if target_model is None:
            raise ValueError("Consistency Training requires 'target_model' in kwargs.")
        
        N = kwargs.get("total_steps", 100) # Default discrete steps for training schedule
        
        # 1. Sample discrete time points
        # t ~ [1, N-1]
        indices = torch.randint(0, N - 1, (x.shape[0],), device=x.device)
        
        # 2. Convert indices to time values (Karras schedule)
        # t_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho
        indices_next = indices + 1
        
        t_cur = self._karras_schedule(indices, N)
        t_next = self._karras_schedule(indices_next, N)
        
        # 3. Add noise
        z = torch.randn_like(x)
        x_cur = x + t_cur.view(-1, *([1]*(x.ndim-1))) * z
        x_next = x + t_next.view(-1, *([1]*(x.ndim-1))) * z
        
        # 4. Predict x_0
        # Online model on x_next (noisier)
        pred_online = self.predict(model, x_next, t_next, condition)
        
        # Target model on x_cur (less noisy)
        with torch.no_grad():
            pred_target = self.predict(target_model, x_cur, t_cur, condition)
            
        # 5. Loss: MSE(pred_online, pred_target)
        # Using SNRLoss or just MSE? Consistency Models usually use a specific metric.
        # Simple MSE for now.
        loss = nn.functional.mse_loss(pred_online, pred_target)
        
        return {"loss": loss, "distance": loss.detach()}

    def predict(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predicts x_0 directly.
        
        Applies skip connection scaling: 
        f(x, t) = c_skip(t)x + c_out(t)F(x, t)
        
        But if 'model' already handles this, we just call it.
        For raw MLP models (like in two_moons), they might not have c_skip/c_out.
        Standard CM parameterization:
        c_skip(t) = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2)
        c_out(t) = sigma_data * (t - epsilon) / sqrt(sigma_data^2 + t^2)
        
        We will implement the scaling here to support raw backbones.
        Assuming epsilon = 0.002 (sigma_min).
        """
        # Ensure t is correct shape
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor([t] * x_t.shape[0], device=x_t.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x_t.shape[0])
            
        # Broadcast t for calculation
        t_in = t.view(-1, *([1]*(x_t.ndim-1)))
        
        sigma_data = 0.5 # Standard assumption
        epsilon = self.sigma_min
        
        c_skip = sigma_data**2 / ((t_in - epsilon)**2 + sigma_data**2)
        c_out = sigma_data * (t_in - epsilon) / (sigma_data**2 + t_in**2).sqrt()
        c_in = 1.0 / (sigma_data**2 + t_in**2).sqrt()
        
        # Model usually predicts F(c_in * x, t)
        # We pass t directly to model
        # Note: We pass c_in * x_t to the model
        F_x = model(c_in * x_t, t, condition)
        
        return c_skip * x_t + c_out * F_x

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        # SNR = 1 / t^2
        return 1.0 / (t ** 2)
        
    def _karras_schedule(self, indices: torch.Tensor, N: int) -> torch.Tensor:
        """Converts integer indices to time values using Karras schedule."""
        inv_rho = 1.0 / self.rho
        start = self.sigma_min ** inv_rho
        end = self.sigma_max ** inv_rho
        
        t_i = start + (indices / (N - 1)) * (end - start)
        return t_i ** self.rho

