import torch

from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler


class _RecordTimestepModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.timesteps: list[torch.Tensor] = []

    def forward(self, x, t, condition):
        self.timesteps.append(t.detach().cpu())
        return torch.zeros_like(x)


def test_sampler_uses_training_timestep_indices_and_terminal_step():
    model = _RecordTimestepModel()
    sampler = BaseDiffusionSampler(
        GaussianDiffusion(schedule="cosine", timesteps=10), model,
        device="cpu", steps=4, sampler_type="ddim",
    )

    sampler.sample(num_samples=2, shape=[1])

    assert [int(t[0]) for t in model.timesteps] == [9, 6, 3, 0]
    assert all(t.dtype == torch.long for t in model.timesteps)
    assert all(torch.equal(t, torch.full_like(t, t[0])) for t in model.timesteps)
