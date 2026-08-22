from typing import List, Optional, Union

import torch
from tqdm import tqdm

from ...core.base_method import BaseMethod
from ...core.base_sampler import BaseSampler


class BaseConsistencyModelSampler(BaseSampler):
    """Sampler for consistency models."""

    def __init__(
        self,
        method: BaseMethod,
        model: torch.nn.Module,
        device: str,
        steps: int = 1,
        feature_keys: Optional[List[str]] = None,
    ):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps

    def sample(
        self,
        num_samples: int,
        shape: Union[torch.Size, List[int]],
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition is not None:
            condition = condition.to(self.device)
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            batch_conditions = condition.shape[0]
            cond_expanded = condition.repeat_interleave(num_samples, dim=0)
            target_bs = batch_conditions * num_samples
        else:
            target_bs = num_samples
            cond_expanded = None
            batch_conditions = 1

        flat_samples = self._sample_batch(target_bs, shape, cond_expanded)
        if batch_conditions > 1 or condition is not None:
            return flat_samples.view(batch_conditions, num_samples, *shape)
        return flat_samples

    def sample_unconditional(
        self,
        num_samples: int,
        shape: Union[torch.Size, List[int]],
    ) -> torch.Tensor:
        """Generates samples from an unconditional consistency model.

        Args:
            num_samples: Number of samples to generate.
            shape: Event shape of an individual sample.

        Returns:
            Generated samples with shape ``[num_samples, *shape]``.
        """
        return self._sample_batch(num_samples, shape)

    def sample_conditional(
        self,
        num_samples: int,
        shape: Union[torch.Size, List[int]],
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Generates samples for each supplied condition.

        Args:
            num_samples: Samples to generate per condition.
            shape: Event shape of an individual sample.
            condition: Conditions with shape ``[num_conditions, *condition_shape]``.

        Returns:
            Generated samples with shape ``[num_conditions, num_samples, *shape]``.
        """
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        condition = condition.to(self.device)
        num_conditions = condition.shape[0]
        expanded_condition = condition.repeat_interleave(num_samples, dim=0)
        samples = self._sample_batch(num_conditions * num_samples, shape, expanded_condition)
        return samples.view(num_conditions, num_samples, *shape)

    def batch_sample(
        self,
        num_samples: int,
        shape: Union[torch.Size, List[int]],
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        all_samples = []
        print(f"Sampling from dataloader (CM) with {self.steps} steps...")
        for batch in tqdm(dataloader, desc="Dataloader Sampling"):
            cond = self._extract_condition(batch)
            if cond is None:
                current_bs = len(next(iter(batch.values()))) if isinstance(batch, dict) else len(batch)
                cond_expanded = None
            else:
                current_bs = cond.shape[0]
                cond_expanded = cond.repeat_interleave(num_samples, dim=0)

            total_items = current_bs * num_samples
            samples_flat = self._sample_batch(total_items, shape, cond_expanded)
            all_samples.append(samples_flat.view(current_bs, num_samples, *shape).cpu())
        return torch.cat(all_samples, dim=0)

    def _sample_batch(
        self,
        current_batch_size: int,
        shape: Union[torch.Size, List[int]],
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_shape = (current_batch_size, *shape)
        x = torch.randn(batch_shape, device=self.device) * self.method.sigma_max

        if self.steps <= 1:
            return self.method.predict(self.model, x, float(self.method.sigma_max), condition)

        sigmas = self.method.sample_seq_sigmas(self.steps, self.device)
        x = self.method.predict(self.model, x, float(sigmas[0]), condition)

        for sigma in sigmas[1:]:
            sigma_value = float(sigma.item())
            z = torch.randn_like(x)
            noise_scale = max(sigma_value**2 - self.method.sigma_min**2, 0.0) ** 0.5
            x = x + noise_scale * z
            x = self.method.predict(self.model, x, sigma_value, condition)

        return x
