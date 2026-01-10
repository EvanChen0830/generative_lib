import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.trainer.consistency_trainer import ConsistencyTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler 
# CM sampling is actually similar to Flow/Diffusion but simpler. 
# A true CM Sampler should be 1-step or multistep consistency sampling.
# For V1, we can use a custom sampler or reuse a base one if we adapted it.
# Let's create a minimal CM sampler here or in the library.

class ConsistencySampler:
    def __init__(self, method, model, device):
        self.method = method
        self.model = model
        self.device = device
        
    def sample(self, num_samples, shape):
        # 1-step generation
        # x_T ~ N(0, I)
        # x_0 = f(x_T, T)
        
        x_T = torch.randn(num_samples, *shape, device=self.device)
        
        # Max sigma (T)
        T_max = self.method.sigma_max
        t = torch.tensor([T_max] * num_samples, device=self.device)
        
        with torch.no_grad():
            x_0 = self.method.predict(self.model, x_T, t)
            
        return x_0

class SimpleMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=64):
        super().__init__()
        # Input: x(2) + t(1) = 3
        # No condition
        self.net = nn.Sequential(
            nn.Linear(in_features + 1, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x, t, condition=None):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # We need to handle broadcasing properly if t is (B, 1) and x is (B, D)
        if t.ndim == 0:
             t = t.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1)
        
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data
    X, _ = make_moons(n_samples=20000, noise=0.05)
    train_loader = DataLoader([{"features": x} for x in torch.from_numpy(X).float()], batch_size=256, shuffle=True)
    
    # Model
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    method = ConsistencyModel(sigma_max=5.0)
    
    trainer = ConsistencyTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["features"],
        label_keys=[], 
        device=device
    )
    
    print("Training Consistency Model (CT)...")
    trainer.fit(train_loader, epochs=50)
    
    # Sample
    sampler = ConsistencySampler(method, model, device)
    samples = sampler.sample(2000, [2])
    samples_np = samples.detach().cpu().numpy()
    
    X_plot = X[:2000]
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Generated (1-step)", s=10)
    plt.legend()
    plt.title("Two Moons - Consistency Training (1-step)")
    plt.savefig("examples/two_moons_consistency.png")
    print("Saved plot.")

if __name__ == "__main__":
    main()
