import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
from pathlib import Path

from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.diffusion.sampler.dps import DPSSampler
from generative_lib.diffusion.sampler.repaint import RepaintSampler
from generative_lib.diffusion.sampler.sdedit import SDEditSampler
from generative_lib.core.base_operator import MaskingOperator

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleMLP(nn.Module):
    def __init__(self, data_dim=2, time_dim=32, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, condition=None):
        if t.dtype == torch.float: t = (t * 1000).long().clamp(0, 999)
        elif t.ndim == 2: t = t.squeeze(-1).long()
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(__file__).resolve().parent.parent / "runs" / "examples" / "two_moons_inverse"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "two_moons_inverse.png"
    
    X, y = make_moons(n_samples=20000, noise=0.05)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std
    
    dataset_list = []
    for i in range(len(X)):
        dataset_list.append({"position": torch.tensor(X_norm[i]).float()})

    train_loader = DataLoader(dataset_list, batch_size=256, shuffle=True)
    
    model = SimpleMLP(data_dim=2, time_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    method = GaussianDiffusion(timesteps=1000, schedule="linear")
    
    trainer = BaseDiffusionTrainer(
        method=method, model=model, optimizer=optimizer,
        feature_keys=[], label_keys=["position"], device=device, tracker=None
    )
    
    print("Training Unconditional Diffusion...")
    trainer.fit(train_loader, epochs=150)
    
    print("Inverse Problem Testing...")
    # Operator: Mask out X coordinate (index 0), retain Y coordinate (index 1)
    # The condition is that Y must exactly equal the denormed equivalent of 0.5
    target_y_denorm = 0.5
    target_y_norm = (target_y_denorm - X_mean[1]) / X_std[1]
    
    y_target = torch.tensor([0.0, target_y_norm], device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32).unsqueeze(0)
    
    operator = MaskingOperator(mask)
    
    # 1. DPS Guidance (Soft Gradients from Mid-Step)
    corrupted_moons = torch.tensor(X_norm[:200]).float().to(device) + torch.randn(200, 2, device=device) * 1.5
    
    sampler_dps = DPSSampler(
        method, model, device, steps=100, 
        operator=operator, y=y_target, zeta=1.0,
        init_x=corrupted_moons, start_ratio=0.5
    )
    s_dps = sampler_dps.sample(num_samples=200, shape=[2]).cpu().numpy()
    
    # 2. Hard Replacement (Repaint from Mid-Step)
    sampler_repaint = RepaintSampler(
        method, model, device, steps=100, 
        operator=operator, y=y_target,
        init_x=corrupted_moons, start_ratio=0.5
    )
    s_repaint = sampler_repaint.sample(num_samples=200, shape=[2]).cpu().numpy()
    
    # 3. SDEdit (Pure Mid-step recovery)
    sampler_sdedit = SDEditSampler(
        method, model, device, steps=100,
        init_x=corrupted_moons, start_ratio=0.5
    )
    s_sdedit = sampler_sdedit.sample(num_samples=200, shape=[2]).cpu().numpy()
    
    # Denormalize
    s_dps = s_dps.reshape(200, 2) * X_std + X_mean
    s_repaint = s_repaint.reshape(200, 2) * X_std + X_mean
    s_sdedit = s_sdedit.reshape(200, 2) * X_std + X_mean
    corrupted_moons_denorm = corrupted_moons.cpu().numpy() * X_std + X_mean
    
    print("DPS Y constraint mean:", s_dps[:, 1].mean(), "target:", target_y_denorm)
    print("DPS Y standard dev: ", s_dps[:, 1].std())
    print("RePaint Y constraint mean:", s_repaint[:, 1].mean(), "target:", target_y_denorm)
    print("RePaint Y standard dev:", s_repaint[:, 1].std())
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].scatter(X[:500, 0], X[:500, 1], alpha=0.1, label="Base Distribution")
    axes[0].scatter(s_dps[:, 0], s_dps[:, 1], c='red', label="After (DPS Guided)", s=15)
    axes[0].axhline(y=target_y_denorm, color='black', linestyle='--', label=f"Y={target_y_denorm} Target")
    axes[0].set_title("DPS Recovery + Guidance")
    axes[0].legend()
    
    axes[1].scatter(X[:500, 0], X[:500, 1], alpha=0.1, label="Base Distribution")
    axes[1].scatter(s_repaint[:, 0], s_repaint[:, 1], c='purple', label="After (Repaint)", s=15)
    axes[1].axhline(y=target_y_denorm, color='black', linestyle='--', label=f"Y={target_y_denorm} Target")
    axes[1].set_title("Repaint Recovery + Constraint")
    axes[1].legend()

    axes[2].scatter(X[:500, 0], X[:500, 1], alpha=0.1, label="Base Distribution")
    axes[2].scatter(corrupted_moons_denorm[:, 0], corrupted_moons_denorm[:, 1], c='gray', label="Before (Noisy)", s=10, alpha=0.5)
    axes[2].scatter(s_sdedit[:, 0], s_sdedit[:, 1], c='green', label="After (SDEdit Denoised)", s=15)
    axes[2].set_title("Pure SDEdit Denoising")
    axes[2].legend()

    
    plt.savefig(figure_path)
    print(f"Saved to {figure_path}")

if __name__ == "__main__":
    main()
