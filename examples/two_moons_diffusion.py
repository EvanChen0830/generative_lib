import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Define User Model with Time Embeddings
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
        
        # Input: Data + Condition + TimeEmb
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
        # x: [B, 2] (Position)
        # t: [B] (Discrete indices or float, we expect discrete for embedding)
        # condition: [B, 1] (Class)
        
        # Ensure t is treated as indices for embedding
        # GaussianDiffusion might pass float(t/T) or int indices. 
        # Detailed Check:
        # If t is float (0-1), scale to 0-999.
        if t.dtype == torch.float:
            t = (t * 1000).long().clamp(0, 999)
        elif t.ndim == 2:
             # If [B, 1]
             t = t.squeeze(-1).long()
        
        t_emb = self.time_mlp(t) # [B, hidden]
        
        if condition is None:
            # Fallback if no condition passed (shouldn't happen in this setup)
            condition = torch.zeros(x.shape[0], 1, device=x.device)
            
        inp = torch.cat([x, condition, t_emb], dim=-1)
        return self.net(inp)

import math

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Prepare Data
    # Feature = Class (0 or 1) - We use this as CONDITION
    # Label = Position (x, y) - We use this as TARGET (x)
    X, y = make_moons(n_samples=20000, noise=0.05)
    
    # Normalization (Standardize Position)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std
    
    # Dataset dictionary: "position" (Target), "class" (Condition)
    dataset_list = []
    for i in range(len(X)):
        dataset_list.append({
            "position": torch.tensor(X_norm[i]).float(),
            "class": torch.tensor([y[i]]).float() # [1] dim
        })

    train_loader = DataLoader(dataset_list, batch_size=256, shuffle=True)
    
    # 3. Setup Components
    # We generate Position (dim=2) conditioned on Class (dim=1)
    model = SimpleMLP(data_dim=2, cond_dim=1, time_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    method = GaussianDiffusion(timesteps=1000, schedule="linear")
    
    # Logger
    # Use local file backend in examples/mlruns/
    logger = Logger(
        project_name="TwoMoons", 
        run_name="Diffusion_MLflow_Check", 
        use_mlflow=True, 
        mlflow_uri="file:./examples/mlruns"
    )

    # Tracker (Connects Logger to Trainer)
    tracker = ModelTracker(
        exp_name="TwoMoons", 
        model_name="Diff", 
        save_dir="./checkpoints/two_moons_diff", 
        logger=logger
    )
    
    # Trainer
    # feature_keys -> x (Target/Data to generate) -> "position"
    # label_keys -> cond (Condition) -> "class"
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
    print("Training Conditional Diffusion (20k samples, 100 epochs)...")
    trainer.fit(train_loader, epochs=100)
    
    # 5. Sample
    print("Sampling...")
    # Sampler needs to know how to extract condition if we pass dataloader
    # We pass label_keys to Sampler? 
    # Currently BaseDiffusionSampler init takes label_keys
    sampler = BaseDiffusionSampler(
        method, 
        model, 
        device, 
        steps=50, 
        label_keys=["class"] # To extract condition from dataloader
    )
    
    # We want to sample using validation/test conditions
    # Let's just use a subset of train loader for conditioning
    # For visualization, let's pick uniform classes 0 and 1
    
    # Manual Conditioning:
    # Generate 250 class 0, 250 class 1
    cond_0 = torch.zeros(250, 1)
    cond_1 = torch.ones(250, 1)
    cond = torch.cat([cond_0, cond_1], dim=0) # [500, 1]
    
    # We want 1 sample PER condition. Total 500.
    samples_norm = sampler.sample(num_samples=1, shape=[2], condition=cond)
    # Result is [500, 1, 2]. Squeeze to [500, 2]
    samples_norm = samples_norm.squeeze(1)
    
    # 6. Verify (Plot)
    samples_np = samples_norm.detach().cpu().numpy()
    
    # Denormalize
    samples_denorm = samples_np * X_std + X_mean
    
    # Downsample Data for plot clarity
    X_plot = X[:2000]
    y_plot = y[:2000]
    
    plt.figure(figsize=(8, 8))
    # Plot real data
    plt.scatter(X_plot[y_plot==0, 0], X_plot[y_plot==0, 1], alpha=0.3, label="Data 0", c='blue', s=10)
    plt.scatter(X_plot[y_plot==1, 0], X_plot[y_plot==1, 1], alpha=0.3, label="Data 1", c='red', s=10)
    
    # Plot generated
    # First 250 are 0, next 250 are 1
    plt.scatter(samples_denorm[:250, 0], samples_denorm[:250, 1], alpha=0.8, label="Gen 0", marker='x', c='cyan')
    plt.scatter(samples_denorm[250:, 0], samples_denorm[250:, 1], alpha=0.8, label="Gen 1", marker='x', c='orange')
    
    plt.legend()
    plt.title("Conditional Diffusion (20k Data, 100 Epochs)")
    plt.savefig("examples/two_moons_diffusion_cond.png")
    print("Saved plot to examples/two_moons_diffusion_cond.png")

if __name__ == "__main__":
    main()
