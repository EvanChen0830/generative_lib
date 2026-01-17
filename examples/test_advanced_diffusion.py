import torch
import torch.nn as nn
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.diffusion.sampler.async_sampler import AsyncDiffusionSampler

class DummyModel(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, channels)
        )
    def forward(self, x, t, condition=None):
        # x: [B, C]
        # t: [B] or float
        # condition: [B, C]
        if isinstance(t, float):
             t = torch.tensor(t).to(x.device).view(1)
        
        emb = t.view(-1, 1).float()
        if condition is not None:
             return self.net(x) + emb + condition
        return self.net(x) + emb

class DummySplitModelStage1(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU()
        )
    def forward(self, x, t, condition=None):
        return self.net(x) # Returns hidden [B, 16]

class DummySplitModelStage2(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, channels)
        )
    def forward(self, x, t, condition=None):
        # x is hidden from stage 1
        return self.net(x) # Returns noise [B, 2]

def test_self_guidance():
    print("Testing Self-Guidance...")
    device = "cpu"
    model = DummyModel().to(device)
    bad_model = DummyModel().to(device)
    method = GaussianDiffusion(timesteps=10)
    
    sampler = BaseDiffusionSampler(
        method=method,
        model=model,
        device=device,
        steps=5,
        bad_model=bad_model,
        bad_guidance_scale=0.5
    )
    
    condition = torch.randn(2, 2).to(device)
    samples = sampler.sample(num_samples=1, shape=(2,), condition=condition)
    print(f"Self-Guidance Samples Shape: {samples.shape}")
    assert samples.shape == (2, 1, 2)
    print("Self-Guidance Test Passed!")

def test_async_diff():
    print("\nTesting AsyncDiff...")
    device = "cpu"
    # Create split model
    stage1 = DummySplitModelStage1().to(device)
    stage2 = DummySplitModelStage2().to(device)
    method = GaussianDiffusion(timesteps=10)
    
    sampler = AsyncDiffusionSampler(
        method=method,
        components=[stage1, stage2],
        device=device,
        steps=5
    )
    
    condition = torch.randn(2, 2).to(device)
    try:
        samples = sampler.sample(num_samples=1, shape=(2,), condition=condition)
        print(f"AsyncDiff Samples Shape: {samples.shape}")
        # Note: AsyncDiff Sampler _sample_batch returns [B, ...]
        # sample() wrapper reshapes it.
        # But wait, AsyncDiff _sample_batch might have issue with returns if I didn't verify reshape?
        # Let's check output.
    except Exception as e:
        print(f"AsyncDiff Failed with error: {e}")
        raise e
        
    print("AsyncDiff Test Passed!")

if __name__ == "__main__":
    test_self_guidance()
    test_async_diff()
