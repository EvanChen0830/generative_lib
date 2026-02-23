import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from ...core.base_sampler import BaseSampler
from ...core.base_method import BaseMethod

class BaseDiffusionSampler(BaseSampler):
    """Sampler for Gaussian Diffusion models."""

    def __init__(self, method: BaseMethod, model: torch.nn.Module, device: str, steps: int = 50, feature_keys: Optional[List[str]] = None):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps
        
    def sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Samples for a single/tensor condition."""
        if condition is not None:
            condition = condition.to(self.device)
            # Normalize shape: [B, C]
            # If condition is [C], unsqueeze to [1, C]
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            
            # Replicate condition: [B, C] -> [B, Num_Samples, C] -> [B*Num_Samples, C]
            B = condition.shape[0]
            cond_expanded = condition.repeat_interleave(num_samples, dim=0)
            target_bs = B * num_samples
        else:
            # Unconditional: Just generate num_samples
            target_bs = num_samples
            cond_expanded = None
            B = 1 # Effectively 1 "group"
        
        flat_samples = self._sample_batch(target_bs, shape, cond_expanded)
        
        # Reshape to [B, Num_Samples, *Shape] or [Num_Samples, *Shape]
        if B > 1 or (condition is not None):
             final_shape = (B, num_samples, *shape)
             return flat_samples.view(final_shape)
        else:
             return flat_samples # [Num, *Shape]

    def batch_sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Samples from dataloader."""
        all_samples = []
        print(f"Sampling from dataloader with {self.steps} steps...")
        for batch in tqdm(dataloader, desc="Dataloader Sampling"):
            cond = self._extract_condition(batch)
            if cond is None:
                 if isinstance(batch, dict):
                     current_bs = len(next(iter(batch.values())))
                 else:
                     current_bs = len(batch)
            else:
                current_bs = cond.shape[0]

            if cond is not None:
                cond_expanded = cond.repeat_interleave(num_samples, dim=0) 
            else:
                cond_expanded = None
            
            total_items = current_bs * num_samples
            with torch.no_grad():
                samples_flat = self._sample_batch(total_items, shape, cond_expanded)
            
            final_shape = (current_bs, num_samples, *shape)
            samples_batch = samples_flat.view(final_shape)
            all_samples.append(samples_batch.cpu())
            
        return torch.cat(all_samples, dim=0)

    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Internal helper."""
        batch_shape = (current_batch_size, *shape)
        x_t = torch.randn(batch_shape, device=self.device)
        
        time_seq = list(reversed(range(0, self.method.timesteps, self.method.timesteps // self.steps)))
        time_seq = time_seq[:self.steps]
        
        # Nested TQDM for time steps (leave=False to avoid clutter)
        for i, t_idx in enumerate(tqdm(time_seq, desc="Diffusion Steps", leave=False)):
            prev_t_idx = t_idx - (self.method.timesteps // self.steps)
            if prev_t_idx < 0: prev_t_idx = -1 
            
            def get_alpha_bar(idx):
                if idx < 0: return torch.tensor(1.0).to(self.device) 
                return self.method.alphas_cumprod[idx]

            alpha_bar_t = get_alpha_bar(t_idx)
            alpha_bar_prev = get_alpha_bar(prev_t_idx)
            
            t_float = t_idx / self.method.timesteps
            pred_noise = self.method.predict(self.model, x_t, float(t_float), condition)
            
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            x_t = x_prev
            
        return x_t
