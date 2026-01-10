import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# Imports
from generative_lib.diffusion.method.cfg_diffusion import CFGDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Model Definition
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
    def __init__(self, data_dim=2, cond_dim=1, time_dim=32, hidden_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, condition):
        if t.dtype == torch.float: t = (t * 1000).long().clamp(0, 999)
        elif t.ndim == 2: t = t.squeeze(-1).long()
        t_emb = self.time_mlp(t)
        if condition is None: condition = torch.zeros(x.shape[0], 1, device=x.device)
        inp = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Data
    X, y = make_moons(n_samples=20000, noise=0.05)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std
    
    dataset_list = []
    for i in range(len(X)):
        dataset_list.append({
            "position": torch.tensor(X_norm[i]).float(),
            "class": torch.tensor([y[i]]).float()
        })

    train_loader = DataLoader(dataset_list, batch_size=256, shuffle=True)
    
    # 3. Components
    model = SimpleMLP(data_dim=2, cond_dim=1, time_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # CFG Diffusion
    # unconditional_value = -1.0 (Assume classes are 0 and 1, so -1 is distinct)
    method = CFGDiffusion(timesteps=1000, schedule="linear", unconditional_value=-1.0)
    
    logger = Logger(project_name="TwoMoons", run_name="CFG_DDIM_Check", use_mlflow=True, mlflow_uri="file:./examples/mlruns")
    tracker = ModelTracker(exp_name="TwoMoons", model_name="CFG_Diff", save_dir="./checkpoints/cfg_diff", logger=logger)
    
    trainer = BaseDiffusionTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["position"],
        label_keys=["class"], 
        device=device,
        tracker=tracker
    )
    
    # 4. Train
    print("Training CFG Diffusion (Dual Loss)...")
    trainer.fit(train_loader, epochs=50) # 50 epochs for speed
    
    # 5. Sampling
    print("Sampling Comparison...")
    
    cond_0 = torch.zeros(250, 1)
    cond_1 = torch.ones(250, 1)
    cond = torch.cat([cond_0, cond_1], dim=0)
    
    # Sampler 1: DDPM (Standard)
    # w=1.0 means cond only (no guidance)
    sampler_ddpm = BaseDiffusionSampler(method, model, device, steps=50, sampler_type="ddpm", guidance_scale=1.0, label_keys=["class"])
    s_ddpm = sampler_ddpm.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()
    
    # Sampler 2: DDIM (Deterministic) w=1.0
    sampler_ddim = BaseDiffusionSampler(method, model, device, steps=50, sampler_type="ddim", guidance_scale=1.0, label_keys=["class"])
    s_ddim = sampler_ddim.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()
    
    # Sampler 3: DDIM with CFG w=3.0
    sampler_cfg = BaseDiffusionSampler(method, model, device, steps=50, sampler_type="ddim", guidance_scale=3.0, unconditional_value=-1.0, label_keys=["class"])
    s_cfg = sampler_cfg.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()
    
    # 6. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    def plot_data(ax, data, title):
        # Denorm
        d = data * X_std + X_mean
        ax.scatter(d[:250, 0], d[:250, 1], c='cyan', label='Class 0', alpha=0.6, s=10)
        ax.scatter(d[250:, 0], d[250:, 1], c='orange', label='Class 1', alpha=0.6, s=10)
        ax.set_title(title)
        ax.legend()

    plot_data(axes[0], s_ddpm, "DDPM (w=1.0)")
    plot_data(axes[1], s_ddim, "DDIM (w=1.0)")
    plot_data(axes[2], s_cfg, "DDIM CFG (w=3.0)")
    
    plt.tight_layout()
    plt.savefig("examples/two_moons_cfg_comparison.png")
    print("Saved plot to examples/two_moons_cfg_comparison.png")
    logger.finish()

if __name__ == "__main__":
    main()
