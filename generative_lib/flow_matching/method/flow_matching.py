"""Conditional flow matching objectives."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ...core.base_method import BaseMethod


class FlowMatching(BaseMethod):
    """Conditional flow matching along a linear optimal-transport path.

    The path begins at standard Gaussian noise and ends at data plus an
    optional residual-noise term controlled by ``sigma_min``.
    """

    def __init__(self, sigma_min: float = 0.0) -> None:
        """Initializes the flow-matching process.

        Args:
            sigma_min: Residual noise retained at time one. Set to zero for a
                path ending exactly at the data distribution.
        """
        super().__init__()
        self.sigma_min = sigma_min

    @staticmethod
    def _append_dims(values: torch.Tensor, target_dims: int) -> torch.Tensor:
        """Expands batch-aligned scalars to match an event tensor's rank."""
        while values.ndim < target_dims:
            values = values.unsqueeze(-1)
        return values

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Constructs the training interpolation at time ``t``.

        Args:
            x_0: Gaussian source samples with shape ``[batch, *event]``.
            x_1: Data samples with shape ``[batch, *event]``.
            t: Batch-aligned times in ``[0, 1]``.

        Returns:
            Interpolated samples with shape ``[batch, *event]``.
        """
        time = self._append_dims(t, x_1.ndim)
        source_weight = 1.0 - (1.0 - self.sigma_min) * time
        return source_weight * x_0 + time * x_1

    def target_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Returns the constant velocity of the interpolation path."""
        return x_1 - (1.0 - self.sigma_min) * x_0

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes the conditional flow-matching mean-squared error.

        Args:
            model: Velocity model accepting ``(x_t, t, condition)``.
            x: Data samples with shape ``[batch, *event]``.
            condition: Optional batch-aligned conditioning values.

        Returns:
            Dictionary containing the scalar ``loss``.
        """
        x_0 = torch.randn_like(x)
        t = torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        x_t = self.interpolate(x_0, x, t)
        velocity = self.target_velocity(x_0, x)
        prediction = self.predict(model, x_t, t, condition)
        loss = torch.nn.functional.mse_loss(prediction, velocity)
        return {"loss": loss}
