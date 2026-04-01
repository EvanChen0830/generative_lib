import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import os

from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.trainer.base import BaseConsistencyModelTrainer
from generative_lib.consistency_model.sampler.base import BaseConsistencyModelSampler
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker
from generative_lib.metrics.distance import calculate_frechet_distance, compute_statistics

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
    def __init__(self, in_features=2, time_dim=32, hidden_features=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_features),
            nn.SiLU()
        )
        self.net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x, t, condition=None):
        if t.ndim == 1: pass
        elif t.ndim == 2: t = t.squeeze(-1)
        
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    X, _ = make_moons(n_samples=20000, noise=0.05)
    X = (X - X.mean(axis=0)) / X.std(axis=0) # Normalize for stability
    
    dataset_list = []
    for i in range(len(X)):
        dataset_list.append({
            "position": torch.tensor(X[i]).float()
        })
    
    train_loader = DataLoader(dataset_list, batch_size=256, shuffle=True)
    
    # Model
    model = SimpleMLP(in_features=2, hidden_features=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    
    # Method
    method = ConsistencyModel(sigma_min=0.002, sigma_max=80.0, rho=7.0, N=40)
    
    os.makedirs("./checkpoints/consistency", exist_ok=True)
    
    logger = Logger(project_name="TwoMoons", run_name="Consistency", use_mlflow=True, mlflow_uri="file:./examples/mlruns")
    tracker = ModelTracker(exp_name="TwoMoons", model_name="Consistency", save_dir="./checkpoints/consistency", logger=logger)
    
    trainer = BaseConsistencyModelTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=[],
        label_keys=["position"], 
        device=device,
        tracker=tracker
    )
    
    # Train
    print("Training Consistency Model (20k samples, 400 epochs)...")
    trainer.scheduler = scheduler
    trainer.fit(train_loader, epochs=400)
    
    # Sample
    print("Sampling...")
    sampler = BaseConsistencyModelSampler(method, model, device, steps=15, feature_keys=[])
    samples = sampler.sample(num_samples=500, shape=[2])
    
    # Verify
    samples_np = samples.detach().cpu().numpy()
    
    print("Computing metrics...")
    mu_real, sig_real = compute_statistics(X)
    mu_gen, sig_gen = compute_statistics(samples_np)
    fd_val = calculate_frechet_distance(mu_real, sig_real, mu_gen, sig_gen)
    print(f"FD Consistency: {fd_val:.4f}")
    
    # Plot results
    X_plot = X[:2000]
    plt.figure(figsize=(8, 6))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Generated", s=10)
    plt.legend()
    plt.title(f"Two Moons - Consistency Model (15 Steps, 400 Epochs)\nFD: {fd_val:.4f}")
    plt.savefig("examples/two_moons_consistency.png")
    print("Saved plot to examples/two_moons_consistency.png")

if __name__ == "__main__":
    main()
