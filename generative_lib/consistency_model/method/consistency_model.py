import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from ...core.base_method import BaseMethod

class ConsistencyModel(BaseMethod):
    """Consistency Models (CM) method.
    
    Implements Consistency Training (CT) and Consistency Distillation (CD).
    
    References:
    - Consistency Models (Song et al., 2023)
    - Improved Techniques for Training Consistency Models (Karras et al., 2023)
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, sigma_data: float = 0.5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data

    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Computes Consistency Loss.
        
        Args:
            model: Online model (theta).
            x: Data [B, D].
            condition: Condition [B, C].
            **kwargs: 
                - target_model: (Required for CT) EMA model (theta_minus).
                - teacher_model: (Required for CD) Frozen diffusion model.
                - total_steps: (Optional) N for discretization.
        """
        target_model = kwargs.get("target_model")
        teacher_model = kwargs.get("teacher_model")
        
        N = kwargs.get("total_steps", 40) # N increases during training in paper, but strict schedule is complex. Fixed N for now.
        
        # 1. Sample discrete time points indices
        # t ~ [0, N-2]
        # t_n matches indices
        indices = torch.randint(0, N - 1, (x.shape[0],), device=x.device)
        indices_next = indices + 1
        
        # 2. Convert indices to time values (Karras schedule)
        t_cur = self._karras_schedule(indices, N)
        t_next = self._karras_schedule(indices_next, N)
        
        # 3. Add noise
        z = torch.randn_like(x)
        
        # Broadcast t
        t_cur_expand = t_cur.view(-1, *([1]*(x.ndim-1)))
        t_next_expand = t_next.view(-1, *([1]*(x.ndim-1)))
        
        x_cur = x + t_cur_expand * z
        x_next = x + t_next_expand * z # Only needed for CT if not using Heun solver ground truth
        
        # 4. Generate Target (Ground Truth for Consistency)
        # Distillation (CD): Use Teacher to estimate x_0 from x_next, then move to t_cur?
        # Actually CD paper: x_target = x_next - (t_next - t_cur) * phi(x_next, t_next)
        # i.e., One Euler step from x_next to t_cur using Teacher.
        # Then Target = f_theta_minus(x_target, t_cur)
        
        with torch.no_grad():
            if teacher_model is not None:
                # --- CONSISTENCY DISTILLATION (CD) ---
                # 1. Get Teacher Score / Velocity at (x_next, t_next)
                # Assuming Teacher output is score or velocity? 
                # Let's assume Teacher is a BaseMethod wrapper or we know its output.
                # If teacher is standard diffusion, usually predicts noise eps or score.
                # We need an interface. 
                # Simplification: Assume teacher model returns 'denoised x_0' or 'velocity'
                # Actually, best to rely on a 'ground truth' update.
                
                # Simple Euler Solver Step using Teacher:
                # dX_t = v(X_t, t) dt --(ODE)--> X_t_cur = X_t_next - (t_next - t_cur) * v(X_t_next)
                
                # We need 'v' from teacher.
                # If teacher is generic, this is hard.
                # Warning: We assume teacher returns 'velocity' or we can derive it.
                # If teacher predicts x_0, v = (x_t - x_0) / t
                
                # Let's assume teacher predicts x_0 directly for simplicity in this V1
                # For diffusion models trained on epsilon, x_0 = (x_t - sigma*eps)
                # This is getting complicated without a unified 'predict_x0' interface.
                
                # FALLBACK FOR V1: Expect teacher to output x_0.
                # If teacher_model is a BaseMethod, use its predict. Otherwise, assume it's a raw model.
                if isinstance(teacher_model, BaseMethod):
                    x_0_teacher = teacher_model.predict(teacher_model.model, x_next, t_next, condition) # Assuming teacher_model has a .model attribute
                else:
                    # If teacher_model is just an nn.Module, assume it predicts x_0 directly
                    x_0_teacher = self.predict(teacher_model, x_next, t_next, condition)
                
                dx = (x_next - x_0_teacher) / t_next_expand
                x_mid = x_next - (t_next_expand - t_cur_expand) * dx
                
                # Target = TargetModel(x_mid, t_cur)
                start_x = x_mid
                
            else:
                # --- CONSISTENCY TRAINING (CT) ---
                # No teacher. Use x_next directly? 
                # In CT, we usually use x_next as the starting point.
                start_x = x_next
        
            # Forward Pass of Target Model (EMA)
            target_out = self.predict(target_model, start_x, t_cur, condition)

        # 5. Online Model Prediction
        # f_theta(x_next, t_next)
        # Note: We predict from the NOISIER step (t_next) to match the LESS NOISY step's target (t_cur)
        online_out = self.predict(model, x_next, t_next, condition)
        
        # 6. Loss
        # Constraint: f(x_next, t_next) == f(x_cur, t_cur)
        # In CD: f(x_next, t_next) == f_target(x_mid_from_teacher, t_cur)
        # In CT: f(x_next, t_next) == f_target(x_next, t_cur) (Soft consistency)
        
        # Loss metric: L2
        # Can use SNRLoss weights if needed (usually 1 for CM?)
        loss = nn.functional.mse_loss(online_out, target_out)
        
        return {"loss": loss}

    def predict(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predicts x_0 directly with skip connections."""
        # Ensure t is correct shape
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor([t] * x_t.shape[0], device=x_t.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x_t.shape[0])
            
        # Broadcast t
        t_in = t.view(-1, *([1]*(x_t.ndim-1)))
        
        # Skip Scaling (EDM/CM style)
        # c_skip = sigma_data^2 / ((t - epsilon)^2 + sigma_data^2)
        # c_out = sigma_data * (t - epsilon) / sqrt(sigma_data^2 + t^2)
        # Here we use proper CM formulation
        
        epsilon = self.sigma_min
        
        c_skip = self.sigma_data**2 / ((t_in - epsilon)**2 + self.sigma_data**2)
        c_out = self.sigma_data * (t_in - epsilon) / (self.sigma_data**2 + t_in**2).sqrt()
        
        # Model F(x, t)
        F_x = model(x_t, t, condition)
        
        # f(x, t) = c_skip * x + c_out * F(x,t)
        return c_skip * x_t + c_out * F_x

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 / (t ** 2)
        
    def _karras_schedule(self, indices: torch.Tensor, N: int) -> torch.Tensor:
        """Karras noise schedule."""
        inv_rho = 1.0 / self.rho
        start = self.sigma_min ** inv_rho
        end = self.sigma_max ** inv_rho
        
        t_i = start + (indices / (N - 1)) * (end - start)
        return t_i ** self.rho

