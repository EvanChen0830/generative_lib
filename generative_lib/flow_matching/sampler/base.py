import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from ...core.base_sampler import BaseSampler
from ...core.base_method import BaseMethod

class BaseFlowMatchingSampler(BaseSampler):
    """Sampler for Flow Matching models."""

    def __init__(self, method: BaseMethod, model: torch.nn.Module, device: str, steps: int = 50, label_keys: Optional[List[str]] = None):
        super().__init__(method, model, device, label_keys=label_keys)
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
        """Samples from dataloader."""
        all_samples = []
        print(f"Sampling from dataloader (FM) with {self.steps} steps...")
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
        """Internal helper."""
        batch_shape = (current_batch_size, *shape)
        x_t = torch.randn(batch_shape, device=self.device)
        
        dt = 1.0 / self.steps
        time_seq = torch.linspace(0, 1, self.steps + 1)[:-1]
        
        for i, t_curr in enumerate(time_seq):
            t_curr_scalar = t_curr.item()
            v_pred = self.method.predict(self.model, x_t, float(t_curr_scalar), condition)
            x_t = x_t + v_pred * dt
            
        return x_t
