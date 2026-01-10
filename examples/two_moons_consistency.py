import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# Imports
from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.trainer.base import BaseConsistencyModelTrainer
from generative_lib.consistency_model.sampler.base import BaseConsistencyModelSampler
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Model Definition (Same as Diffusion)
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
        # CM passes float t, we handle it
        if t.ndim == 1:
            pass # already batch
        elif t.ndim == 0:
            t = t.unsqueeze(0).repeat(x.shape[0])
            
        # Scale t for embedding if needed, but Sinusoidal can handle 0-1 range if designed so
        # Standard Sinusoidal expects larger range usually, let's scale 0-1 to 0-1000 for embedding
        t_scaled = t * 1000
        
        t_emb = self.time_mlp(t_scaled)
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
    
    # Consistency Model
    # standard params: sigma_min=0.002, sigma_max=80.0, rho=7.0
    method = ConsistencyModel(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    
    logger = Logger(project_name="TwoMoonsCM", run_name="Consistency_Training_Check", use_mlflow=False) # Disable mlflow for quick check
    tracker = ModelTracker(exp_name="TwoMoonsCM", model_name="CM_MLP", save_dir="./checkpoints/cm", logger=logger)
    
    trainer = BaseConsistencyModelTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["position"],
        label_keys=["class"], 
        device=device,
        tracker=tracker,
        ema_decay=0.999, # Slow decay for stability
        total_steps=100
    )
    
    # 4. Train
    print("Training Consistency Model...")
    trainer.fit(train_loader, epochs=50) 
    # CM usually converges faster in terms of steps, but we train for 50 epochs to be safe.
    
    # 5. Sampling
    print("Sampling...")
    
    cond_0 = torch.zeros(250, 1)
    cond_1 = torch.ones(250, 1)
    cond = torch.cat([cond_0, cond_1], dim=0)
    
    # Sampler (1 Step)
    # We use the target_model (EMA) for sampling as per CM practice
    sampler = BaseConsistencyModelSampler(method, trainer.target_model, device, steps=1, label_keys=["class"])
    
    # Sample
    samples = sampler.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()
    
    # 6. Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Denorm
    d = samples * X_std + X_mean
    ax.scatter(d[:250, 0], d[:250, 1], c='cyan', label='Class 0', alpha=0.6, s=10)
    ax.scatter(d[250:, 0], d[250:, 1], c='orange', label='Class 1', alpha=0.6, s=10)
    ax.set_title("Consistency Model (1 Step Generation)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("examples/two_moons_consistency.png")
    print("Saved plot to examples/two_moons_consistency.png")

if __name__ == "__main__":
    main()
