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

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, N: int = 40):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.N = N

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None, ema_model: Optional[torch.nn.Module] = None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        
        n = torch.randint(0, self.N - 1, (batch_size,), device=device)
        
        def get_sigma(idx):
            return (self.sigma_min**(1/self.rho) + idx / (self.N - 1) * (self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho)))**self.rho
            
        t_n = get_sigma(n).view(-1, 1).float()
        t_n1 = get_sigma(n + 1).view(-1, 1).float()
        
        z = torch.randn_like(x)
        
        x_n = x + z * t_n
        x_n1 = x + z * t_n1
        
        pred_n1 = self.predict(model, x_n1, t_n1.squeeze(-1), condition)
        
        teacher = ema_model if ema_model is not None else model
        with torch.no_grad():
            pred_n = self.predict(teacher, x_n, t_n.squeeze(-1), condition)
            
        loss = torch.nn.functional.mse_loss(pred_n1, pred_n)
        return {"loss": loss}

    def predict(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        sigma_data = 1.0
        
        if isinstance(t, float):
             t = torch.tensor([t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        elif t.numel() == 1 and x_t.shape[0] > 1:
             t = t.repeat(x_t.shape[0])
             
        while t.ndim < x_t.ndim:
             t = t.unsqueeze(-1)
             
        # Proper Consistency Model Preconditioning
        epsilon = self.sigma_min
        c_in = 1.0 / torch.sqrt(sigma_data**2 + t**2)
        c_skip = (sigma_data**2) / ((t - epsilon)**2 + sigma_data**2)
        c_out = (t - epsilon) * sigma_data / torch.sqrt(t**2 + sigma_data**2)
        
        # Scale inputs before prediction
        scaled_x_t = x_t * c_in
        model_out = model(scaled_x_t, t.squeeze(-1), condition)
        
        return c_skip * x_t + c_out * model_out

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 / (t ** 2)
