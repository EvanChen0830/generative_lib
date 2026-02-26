from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
from .base_method import BaseMethod

class BaseSampler(ABC):
    """Abstract base sampler for generative models.

    Handles the inference loop:
    - Iterating from T to 0 (Diffusion) or 0 to 1 (Flow).
    """

    def __init__(
        self,
        method: BaseMethod,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        feature_keys: Optional[List[str]] = None,
    ):
        self.method = method
        self.model = model
        self.device = device
        self.feature_keys = feature_keys
        self.model.to(self.device)
        self.model.eval()

    def _extract_condition(self, batch: Any) -> Optional[torch.Tensor]:
        """Helper to extract condition from batch using feature_keys."""
        if not self.feature_keys:
            if isinstance(batch, torch.Tensor):
                return batch.to(self.device)
            return None
            
        if not isinstance(batch, dict):
             return None

        conds = []
        for k in self.feature_keys:
            if k in batch:
                conds.append(batch[k].to(self.device))
        
        if conds:
            if len(conds) > 1:
                return torch.cat(conds, dim=-1)
            return conds[0]
        return None

    @abstractmethod
    def sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Samples for a single condition (or unconditional).
        
        Args:
            num_samples: Number of samples to generate per condition (if condition provided) or total.
            shape: Shape of a single sample (excluding batch dimension).
            condition: Optional conditioning tensor [B, C].
        
        Returns:
            Generated samples [B, Num_Samples, *Shape] or [Num_Samples, *Shape] if B=1.
        """
        pass

    @abstractmethod
    def batch_sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Samples iterating over a dataloader.

        Args:
            num_samples: Number of samples to generate per batch item.
            shape: Shape of a single sample.
            dataloader: DataLoader yielding batch conditions.

        Returns:
             Generated samples aggregated [Total_N, Num_Samples, *Shape].
        """
        pass
