import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler

def get_model(device):
    model = nn.Sequential(
        nn.Linear(3, 64),
        nn.BatchNorm1d(64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.SiLU(),
        nn.Linear(64, 2),
    ).to(device)
    return model

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x, t, condition=None):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Train Base Model (1-Flow)
    print("--- 1. Training Base Model ---")
    X, _ = make_moons(n_samples=10000, noise=0.05)
    train_loader = DataLoader([{"features": x} for x in torch.from_numpy(X).float()], batch_size=256, shuffle=True)
    
    base_model = SimpleMLP().to(device)
    base_opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    method = FlowMatching(sigma_min=1e-4) # Standard FM
    
    trainer = BaseFlowMatchingTrainer(method, base_model, base_opt, ["features"], [], device)
    trainer.fit(train_loader, epochs=20) # Quick training
    
    # 2. Generate Reflow Data Pairs (x_0, x_1_generated)
    print("--- 2. Generating Reflow Pairs ---")
    sampler = BaseFlowMatchingSampler(method, base_model, device, steps=20)
    
    # We need to capture x_0 used for sampling to form pairs.
    # The sampler currently doesn't return x_0. 
    # But we can manually sample since we have the model.
    # Actually, Reflow defines pairs (Z0, Z1) where Z1 is the *result* of simulating Z0 to T=1.
    
    num_reflow_samples = 10000
    z0 = torch.randn(num_reflow_samples, 2, device=device)
    
    # Use sampler internals or just call sample but it doesn't accept z0.
    # Let's check BaseSampler... it usually generates random noise internally.
    # I should probably update `BaseSampler` to accept `latent` or `noise`.
    # But for now, I will just call `_sample_batch` if I can access it or just manually loop.
    # `_sample_batch` creates `x_t = torch.randn`.
    
    # Let's just manually run the ODE here for clarity and control.
    dt = 1.0 / 20
    x_t = z0.clone()
    with torch.no_grad():
        for i in range(20):
            t_curr = i * dt
            # predict velocity
            t_tensor = torch.tensor([t_curr]*num_reflow_samples, device=device).unsqueeze(1)
            v_pred = base_model(x_t, t_tensor) # SimpleMLP forward is (x, t) -> cat -> net
            # Wait, FlowMatching.predict does wrapper.
            # But SimpleMLP forward signature is `forward(self, x, t, condition=None)`.
            # And `method.predict` handles broadcasting.
            
            # Using method.predict is safer.
            v_pred = method.predict(base_model, x_t, t_curr)
            x_t = x_t + v_pred * dt
            
    z1 = x_t
    
    # Now we have pairs (z0, z1).
    # Reflow training: We want to match vector field v = z1 - z0.
    # Loss: MSE(v_pred(x_t), z1 - z0).
    # where x_t is interpolation between z0 and z1.
    
    # Create dataset of pairs
    reflow_dataset = TensorDataset(z0.cpu(), z1.cpu())
    reflow_loader = DataLoader(reflow_dataset, batch_size=256, shuffle=True)
    
    # 3. Train Reflow Model (2-Flow)
    print("--- 3. Training Reflow Model ---")
    reflow_model = SimpleMLP().to(device)
    reflow_opt = torch.optim.Adam(reflow_model.parameters(), lr=1e-3)
    
    # Custom training loop needed because we need to pass x_0 to compute_loss
    # Or we can update BaseFlowMatchingTrainer to handle this special dataset?
    # Probably easiest to just write a simple loop here.
    
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for b_z0, b_z1 in reflow_loader:
            b_z0, b_z1 = b_z0.to(device), b_z1.to(device)
            
            reflow_opt.zero_grad()
            
            # For Reflow, x_1 is b_z1, and x_0 is b_z0.
            # We treat b_z1 as 'data' and b_z0 as 'noise' (though it is fixed).
            loss_dict = method.compute_loss(reflow_model, x=b_z1, x_0=b_z0)
            
            loss = loss_dict["loss"]
            loss.backward()
            reflow_opt.step()
            total_loss += loss.item()
            
        print(f"Reflow Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(reflow_loader):.4f}")
        
    # 4. Verify
    print("--- 4. Sampling from Reflow Model ---")
    reflow_sampler = BaseFlowMatchingSampler(method, reflow_model, device, steps=20)
    samples = reflow_sampler.sample(num_samples=2000, shape=[2])
    samples_np = samples.detach().cpu().numpy()
    
    X_plot = X[:2000]
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Reflow Generated", s=10)
    plt.legend()
    plt.title("Two Moons - Rectified Flow (Reflow)")
    plt.savefig("examples/two_moons_reflow.png")
    print("Saved plot to examples/two_moons_reflow.png")

if __name__ == "__main__":
    main()
