import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from ...core.base_sampler import BaseSampler
from ...core.base_method import BaseMethod

class BaseConsistencyModelSampler(BaseSampler):
    """Sampler for Consistency Models."""

    def __init__(self, method: BaseMethod, model: torch.nn.Module, device: str, steps: int = 1, feature_keys: Optional[List[str]] = None):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps

    def sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if condition is not None:
            condition = condition.to(self.device)
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            B = condition.shape[0]
            cond_expanded = condition.repeat_interleave(num_samples, dim=0)
            target_bs = B * num_samples
        else:
            target_bs = num_samples
            cond_expanded = None
            B = 1
        
        flat_samples = self._sample_batch(target_bs, shape, cond_expanded)
        
        if B > 1 or (condition is not None):
             final_shape = (B, num_samples, *shape)
             return flat_samples.view(final_shape)
        else:
             return flat_samples

    def batch_sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        all_samples = []
        print(f"Sampling from dataloader (CM) with {self.steps} steps...")
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
            samples_flat = self._sample_batch(total_items, shape, cond_expanded)
            
            final_shape = (current_bs, num_samples, *shape)
            samples_batch = samples_flat.view(final_shape)
            all_samples.append(samples_batch.cpu())
            
        return torch.cat(all_samples, dim=0)

    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_shape = (current_batch_size, *shape)
        x = torch.randn(batch_shape, device=self.device) * self.method.sigma_max
        
        if self.steps <= 1:
            return self.method.predict(self.model, x, float(self.method.sigma_max), condition)
            
        t_seq = []
        for i in range(self.steps):
            t_curr = (self.method.sigma_max**(1/self.method.rho) + i / (self.steps - 1) * (self.method.sigma_min**(1/self.method.rho) - self.method.sigma_max**(1/self.method.rho)))**self.method.rho
            t_seq.append(t_curr)
            
        x = self.method.predict(self.model, x, float(t_seq[0]), condition)
        epsilon = self.method.sigma_min
        
        for i in range(1, self.steps):
            t_curr = t_seq[i]
            z = torch.randn_like(x)
            std = float(np.sqrt(max(0, t_curr**2 - epsilon**2)))
            x_noisy = x + z * std
            x = self.method.predict(self.model, x_noisy, float(t_curr), condition)
            
        return x
