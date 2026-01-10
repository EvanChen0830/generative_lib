import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler
from generative_lib.utils.logger import Logger

class SimpleMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=64):
        super().__init__()
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
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Prepare Data
    X, _ = make_moons(n_samples=20000, noise=0.05)
    train_loader = DataLoader(
        [{"features": x} for x in torch.from_numpy(X).float()], 
        batch_size=256, 
        shuffle=True
    )
    
    # 3. Setup Components
    model = SimpleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    method = FlowMatching(sigma_min=0.0)
    
    trainer = BaseFlowMatchingTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["features"],
        label_keys=[], 
        device=device
    )
    
    # 4. Train
    print("Training Flow Matching (20k samples, 100 epochs)...")
    trainer.fit(train_loader, epochs=100)
    
    # 5. Sample
    print("Sampling...")
    sampler = BaseFlowMatchingSampler(method, model, device, steps=50)
    
    # Unconditional sampling check (or dummy conditional since we didn't train robustly on Cond for Flow yet?)
    # Wait, the Flow Trainer example above was UNCONDITIONAL (label_keys=[]).
    # X, _ = make_moons. Labels usage was implicit or ignored in training?
    # In two_moons_flow.py (Step 96), `label_keys=[]`. So it's unconditional.
    # So sample(num_samples=500) works as batch size.
    samples = sampler.sample(num_samples=500, shape=[2])
    # Returns [500, 2] because B=1.
    
    # 6. Verify
    samples_np = samples.detach().cpu().numpy()
    
    # Plot results
    # Downsample Data for plot clarity
    X_plot = X[:2000]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Generated")
    plt.legend()
    plt.title("Two Moons - Flow Matching (20k Data, 100 Epochs)")
    plt.savefig("examples/two_moons_flow.png")
    print("Saved plot to examples/two_moons_flow.png")

if __name__ == "__main__":
    main()
