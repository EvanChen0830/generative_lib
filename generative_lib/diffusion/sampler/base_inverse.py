import torch
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from ...core.base_sampler import BaseSampler
from ...core.base_method import BaseMethod

class BaseInverseSampler(BaseSampler, ABC):
    """Base class for Inverse Problem samplers.
    
    Provides SDEdit (mid-step) initialization functionality and standard sampling wrappers.
    """

    def __init__(
        self, 
        method: BaseMethod, 
        model: torch.nn.Module, 
        device: str, 
        steps: int = 50, 
        feature_keys: Optional[List[str]] = None,
        init_x: Optional[torch.Tensor] = None,
        start_ratio: float = 1.0
    ):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps
        
        # SDEdit parameters
        self.init_x = init_x
        self.start_ratio = max(0.01, min(start_ratio, 1.0))
        
        if self.init_x is not None:
             self.init_x = self.init_x.to(self.device)

    def sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if condition is not None:
            condition = condition.to(self.device)
            if condition.ndim == 1: condition = condition.unsqueeze(0)
            B = condition.shape[0]
            cond_expanded = condition.repeat_interleave(num_samples, dim=0)
            target_bs = B * num_samples
        else:
            target_bs = num_samples
            cond_expanded = None
            B = 1 
            
        flat_samples = self._sample_batch(target_bs, shape, cond_expanded)
        
        if B > 1 or (condition is not None):
             return flat_samples.view(B, num_samples, *shape)
        else:
             return flat_samples

    def batch_sample(self, num_samples: int, shape: Union[torch.Size, List[int]], dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        all_samples = []
        for batch in tqdm(dataloader, desc="Dataloader Inverse Sampling"):
            cond = self._extract_condition(batch)
            current_bs = cond.shape[0] if cond is not None else len(next(iter(batch.values())))
            cond_expanded = cond.repeat_interleave(num_samples, dim=0) if cond is not None else None
            
            samples_flat = self._sample_batch(current_bs * num_samples, shape, cond_expanded)
            all_samples.append(samples_flat.view(current_bs, num_samples, *shape).cpu())
            
        return torch.cat(all_samples, dim=0)

    def _get_alpha_bar(self, idx, shape_len):
        if idx < 0: return torch.tensor(1.0).to(self.device).view(1, *([1]*shape_len))
        return self.method.alphas_cumprod[idx].to(self.device).view(1, *([1]*shape_len))

    def _initialize_state(self, current_batch_size: int, shape: Union[torch.Size, List[int]]):
        """Initializes time sequence and starting state x_t using SDEdit logic."""
        time_seq = list(reversed(range(0, self.method.timesteps, self.method.timesteps // self.steps)))
        start_idx = int(self.steps * (1.0 - self.start_ratio))
        if start_idx >= len(time_seq): start_idx = len(time_seq) - 1
        time_seq = time_seq[start_idx:]
        
        batch_shape = (current_batch_size, *shape)
        
        if self.init_x is not None and len(time_seq) > 0:
            t_start_idx = time_seq[0]
            alpha_bar = self._get_alpha_bar(t_start_idx, len(shape))
            
            init_x_expanded = self.init_x
            if init_x_expanded.shape[0] == 1 and current_batch_size > 1:
                init_x_expanded = init_x_expanded.repeat(current_batch_size, *[1]*(init_x_expanded.ndim - 1))
            elif init_x_expanded.shape[0] != current_batch_size:
                # Naive repeat context alignment (assuming num_samples distribution handling)
                init_x_expanded = init_x_expanded.repeat_interleave(current_batch_size // init_x_expanded.shape[0], dim=0)
                
            noise = torch.randn(batch_shape, device=self.device)
            x_t = torch.sqrt(alpha_bar) * init_x_expanded + torch.sqrt(1 - alpha_bar) * noise
        else:
            x_t = torch.randn(batch_shape, device=self.device)
            
        return x_t, time_seq

    @abstractmethod
    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """To be implemented by subclasses, relying on _initialize_state"""
        pass
