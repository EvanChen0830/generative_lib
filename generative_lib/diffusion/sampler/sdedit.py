import torch
from typing import Union, List, Optional
from tqdm import tqdm
from .base_inverse import BaseInverseSampler

class SDEditSampler(BaseInverseSampler):
    """SDEdit (Stochastic Differential Editing) method for Mid-Step Denoising."""

    def __init__(
        self, 
        method, model, device, steps=50, feature_keys=None,
        init_x: Optional[torch.Tensor] = None,
        start_ratio: float = 1.0
    ):
        super().__init__(method, model, device, steps, feature_keys, init_x, start_ratio)
        if init_x is None or start_ratio >= 1.0:
            print("Warning: SDEdit initialized without init_x or start_ratio < 1.0. This equates to unconditional generation.")

    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_t, time_seq = self._initialize_state(current_batch_size, shape)
        
        for i, t_idx in enumerate(tqdm(time_seq, desc="SDEdit Steps", leave=False)):
            prev_t_idx = t_idx - (self.method.timesteps // self.steps)
            if prev_t_idx < 0: prev_t_idx = -1 
            
            alpha_bar_t = self._get_alpha_bar(t_idx, len(shape))
            alpha_bar_prev = self._get_alpha_bar(prev_t_idx, len(shape))
            t_float = t_idx / self.method.timesteps
            
            with torch.no_grad():
                pred_noise = self.method.predict(self.model, x_t, t_float, condition)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
                x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                
                if prev_t_idx >= 0:
                    sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                    dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
                    noise = torch.randn_like(x_t)
                    x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
                        
                x_t = x_prev

        return x_t
