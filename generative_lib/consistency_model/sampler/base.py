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
        guidance_scale: float = 1.0,
        unconditional_value: float = -1.0,
    ):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.unconditional_value = unconditional_value

    def _guided_predict(
        self,
        x: torch.Tensor,
        t: float,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.guidance_scale == 1.0 or condition is None:
            return self.method.predict(self.model, x, t, condition)

        x_in = torch.cat([x, x], dim=0)
        uncond = torch.full_like(condition, self.unconditional_value)
        c_in = torch.cat([condition, uncond], dim=0)
        pred_all = self.method.predict(self.model, x_in, t, c_in)
        pred_cond, pred_uncond = torch.chunk(pred_all, 2, dim=0)
        return pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

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

    def batch_sample(
        self,
        num_samples: int,
        shape: Union[torch.Size, List[int]],
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        all_samples = []
        print(f"Sampling from dataloader (CM, w={self.guidance_scale}) with {self.steps} steps...")
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
        x = torch.randn(batch_shape, device=self.device) * (self.method.sigma_max * 1.5)

        if self.steps <= 1:
            return self._guided_predict(x, float(self.method.sigma_max), condition)

        sigmas = self.method.sample_seq_sigmas(self.steps, self.device, schedule="linear")
        x = self._guided_predict(x, float(self.method.sigma_max), condition)

        for sigma in sigmas[1:]:
            sigma = torch.clamp(sigma, min=self.method.sigma_min, max=self.method.sigma_max)
            sigma_value = float(sigma.item())
            z = torch.randn_like(x)
            noise_scale = max(sigma_value**2 - self.method.sigma_min**2, 0.0) ** 0.5
            x = x + noise_scale * z
            x = self._guided_predict(x, sigma_value, condition)

        return x
