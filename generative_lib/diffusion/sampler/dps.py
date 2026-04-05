import torch
from typing import Union, List, Optional
from tqdm import tqdm
from .base_inverse import BaseInverseSampler
from ...core.base_operator import BaseOperator

class DPSSampler(BaseInverseSampler):
    """Diffusion Posterior Sampling (DPS) inverse method.
    
    Implements the DPS algorithm (Chung et al., 2023) which applies gradient-based
    guidance from a measurement operator to steer the reverse diffusion process
    toward solutions consistent with the observation y = A(x).
    
    The gradient correction is applied to the DDPM posterior mean before adding
    stochastic noise, and gradients are normalized for stability.
    """

    def __init__(
        self, 
        method, model, device, steps=50, feature_keys=None,
        operator: Optional[BaseOperator] = None,
        y: Optional[torch.Tensor] = None,
        zeta: float = 1.0, 
        init_x: Optional[torch.Tensor] = None,
        start_ratio: float = 1.0
    ):
        super().__init__(method, model, device, steps, feature_keys, init_x, start_ratio)
        self.operator = operator
        self.y = y
        self.zeta = zeta
        
        if self.operator is not None:
             self.operator.to(self.device)
        if self.y is not None:
             self.y = self.y.to(self.device)

    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_t, time_seq = self._initialize_state(current_batch_size, shape)
        
        y_cond = self.y
        if y_cond is not None and y_cond.shape[0] == 1 and current_batch_size > 1:
            y_cond = y_cond.repeat(current_batch_size, *[1]*(y_cond.ndim - 1))
            
        use_guidance = self.operator is not None and y_cond is not None

        for i, t_idx in enumerate(tqdm(time_seq, desc="DPS Steps", leave=False)):
            prev_t_idx = t_idx - (self.method.timesteps // self.steps)
            if prev_t_idx < 0: prev_t_idx = -1 
            
            alpha_bar_t = self._get_alpha_bar(t_idx, len(shape))
            alpha_bar_prev = self._get_alpha_bar(prev_t_idx, len(shape))
            t_float = t_idx / self.method.timesteps
            
            # Single forward pass — with gradient tracking only when guidance is needed
            if use_guidance:
                x_t_input = x_t.detach().requires_grad_(True)
            else:
                x_t_input = x_t

            with torch.set_grad_enabled(use_guidance):
                pred_noise = self.method.predict(self.model, x_t_input, t_float, condition)
                pred_x0 = (x_t_input - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                
                if use_guidance:
                    # DPS: compute gradient of ||A(x̂₀) - y||² w.r.t. x_t
                    diff = self.operator(pred_x0) - y_cond
                    loss = torch.sum(diff ** 2) / 2.0
                    grad = torch.autograd.grad(loss, x_t_input)[0]
                    
                    # Normalize gradient for dimension-agnostic step size
                    grad_norm = torch.linalg.norm(grad)
                    grad_step = self.zeta * grad / (grad_norm + 1e-8)
                else:
                    grad_step = 0.0

            with torch.no_grad():
                pred_noise_d = pred_noise.detach()
                pred_x0_d = pred_x0.detach()
                
                if prev_t_idx >= 0:
                    # DDPM reverse step with stochastic noise
                    sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                    dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise_d
                    noise = torch.randn_like(x_t)
                    # DPS: gradient correction applied to the posterior MEAN, before noise
                    mean = torch.sqrt(alpha_bar_prev) * pred_x0_d + dir_xt
                    x_prev = mean - grad_step + sigma_t * noise
                else:
                    # Final step: deterministic
                    x_prev = pred_x0_d - grad_step
                    
                x_t = x_prev

        return x_t
