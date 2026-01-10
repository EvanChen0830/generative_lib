import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler

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
    print(f"Using device: {device}")
    
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
    
    # Enable OT Minibatch
    method = FlowMatching(sigma_min=1e-4, ot_minibatch=True)
    
    trainer = BaseFlowMatchingTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["features"],
        label_keys=[], 
        device=device
    )
    
    # 4. Train
    print("Training Flow Matching (OT-Enabled)...")
    trainer.fit(train_loader, epochs=50) # OT should converge faster, so 50 epochs might be enough
    
    # 5. Sample
    print("Sampling...")
    sampler = BaseFlowMatchingSampler(method, model, device, steps=20) # Fewer steps often needed with OT
    
    samples = sampler.sample(num_samples=2000, shape=[2])
    
    # 6. Verify
    samples_np = samples.detach().cpu().numpy()
    
    # Plot results
    X_plot = X[:2000]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Generated (OT)", s=10)
    plt.legend()
    plt.title("Two Moons - OT Flow Matching")
    plt.savefig("examples/two_moons_flow_ot.png")
    print("Saved plot to examples/two_moons_flow_ot.png")

if __name__ == "__main__":
    main()
