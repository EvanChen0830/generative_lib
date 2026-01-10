import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler

class ConditionalMLP(nn.Module):
    def __init__(self, in_features=2, hidden_features=64):
        super().__init__()
        # Input: x (2) + t (1) + cond (1) = 4
        self.net = nn.Sequential(
            nn.Linear(in_features + 1 + 1, hidden_features),
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
        
        # Condition should be [B, 1]
        if condition is not None:
            if condition.ndim == 1:
                condition = condition.unsqueeze(-1)
            inp = torch.cat([x, t, condition], dim=-1)
        else:
            # Fallback (shouldn't happen in this script)
            # Add dummy zero
            dummy = torch.zeros_like(t)
            inp = torch.cat([x, t, dummy], dim=-1)
            
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Prepare Data
    X, y = make_moons(n_samples=20000, noise=0.05)
    # y is 0/1. Convert to float.
    train_loader = DataLoader(
        [{"features": x, "label": l} for x, l in zip(torch.from_numpy(X).float(), torch.from_numpy(y).float())], 
        batch_size=256, 
        shuffle=True
    )
    
    # 3. Setup Components
    model = ConditionalMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Standard FM
    method = FlowMatching(sigma_min=1e-4) 
    
    trainer = BaseFlowMatchingTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["features"],
        label_keys=["label"], # Use labels!
        device=device
    )
    
    # 4. Train
    print("Training Conditional Flow Matching...")
    trainer.fit(train_loader, epochs=50) 
    
    # 5. Sample
    print("Sampling Conditional...")
    sampler = BaseFlowMatchingSampler(method, model, device, steps=50)
    
    # Sample Class 0 (Top Moon)
    cond_0 = torch.zeros(1, 1).to(device) # Single condition
    samples_0 = sampler.sample(num_samples=1000, shape=[2], condition=cond_0)
    samples_0 = samples_0.squeeze(0).detach().cpu().numpy() # [1000, 2]
    
    # Sample Class 1 (Bottom Moon)
    cond_1 = torch.ones(1, 1).to(device)
    samples_1 = sampler.sample(num_samples=1000, shape=[2], condition=cond_1)
    samples_1 = samples_1.squeeze(0).detach().cpu().numpy()
    
    # 6. Verify
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], alpha=0.1, label="Data 0", c='blue')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], alpha=0.1, label="Data 1", c='red')
    plt.title("Ground Truth")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(samples_0[:, 0], samples_0[:, 1], alpha=0.6, label="Gen 0", c='cyan')
    plt.scatter(samples_1[:, 0], samples_1[:, 1], alpha=0.6, label="Gen 1", c='orange')
    plt.title("Conditional Generation")
    plt.legend()
    
    plt.savefig("examples/two_moons_flow_cond.png")
    print("Saved plot to examples/two_moons_flow_cond.png")

if __name__ == "__main__":
    main()
