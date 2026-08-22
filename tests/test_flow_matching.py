"""Numerical and sampling tests for flow matching."""

import torch
import torch.nn as nn

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler


class ZeroVelocity(nn.Module):
    """A velocity model that leaves the ODE state unchanged."""

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns a zero velocity with the input shape."""
        del t, condition
        return torch.zeros_like(x)


def test_sigma_min_path_and_velocity_are_consistent() -> None:
    """The analytic velocity equals the finite difference of the FM path."""
    method = FlowMatching(sigma_min=0.2)
    x_0 = torch.tensor([[1.0, -2.0]])
    x_1 = torch.tensor([[3.0, 4.0]])
    t_0 = torch.tensor([0.25])
    t_1 = torch.tensor([0.75])

    path_delta = method.interpolate(x_0, x_1, t_1) - method.interpolate(x_0, x_1, t_0)
    expected_delta = (t_1 - t_0).view(-1, 1) * method.target_velocity(x_0, x_1)

    torch.testing.assert_close(path_delta, expected_delta)
    torch.testing.assert_close(method.interpolate(x_0, x_1, torch.zeros(1)), x_0)
    torch.testing.assert_close(method.interpolate(x_0, x_1, torch.ones(1)), x_1 + 0.2 * x_0)


def test_flow_sampler_integrates_on_the_requested_device() -> None:
    """Zero velocity produces the seeded Gaussian source without gradients."""
    method = FlowMatching()
    sampler = BaseFlowMatchingSampler(method, ZeroVelocity(), device="cpu", steps=4)

    torch.manual_seed(7)
    expected = torch.randn(3, 2)
    torch.manual_seed(7)
    samples = sampler.sample(num_samples=3, shape=[2])

    torch.testing.assert_close(samples, expected)
    assert not samples.requires_grad
