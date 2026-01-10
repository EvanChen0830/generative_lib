import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.trainer.distillation_trainer import ConsistencyDistillationTrainer
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.consistency_model.trainer.consistency_trainer import ConsistencyTrainer # Just for structure if needed

# Re-use SimpleMLP
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
        if t.ndim == 0:
             t = t.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

class ConsistencySampler:
    def __init__(self, method, model, device):
        self.method = method
        self.model = model
        self.device = device
        
    def sample(self, num_samples, shape):
        x_T = torch.randn(num_samples, *shape, device=self.device)
        T_max = self.method.sigma_max
        t = torch.tensor([T_max] * num_samples, device=self.device)
        with torch.no_grad():
            x_0 = self.method.predict(self.model, x_T, t)
        return x_0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data
    X, _ = make_moons(n_samples=10000, noise=0.05)
    train_loader = DataLoader([{"features": x} for x in torch.from_numpy(X).float()], batch_size=256, shuffle=True)
    
    # 1. Train Teacher (Diffusion) - Use a NEW one to be self-contained
    print("--- 1. Training Teacher (Diffusion) ---")
    teacher_model = SimpleMLP().to(device)
    teacher_opt = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    teacher_method = GaussianDiffusion() # Standard DDPM
    # We need a trainer for diffusion. Assume BaseTrainer is generic or BaseDiffusionTrainer exists.
    # We can use BaseFlowMatchingTrainer logic effectively (it's just a trainer).
    # But let's check if BaseTrainer is abstract. Yes.
    # We need a concrete trainer.
    # Let's import just for this script or use a simple loop.
    # Actually, we have `generative_lib.flow_matching.trainer.base.BaseFlowMatchingTrainer` which is just `BaseTrainer`.
    # Let's define a GenericTrainer here.
    from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer as GenericTrainer
    
    teacher_trainer = GenericTrainer(teacher_method, teacher_model, teacher_opt, ["features"], [], device)
    teacher_trainer.fit(train_loader, epochs=20) # Quick training
    
    # 2. Distill into Student (CM)
    print("--- 2. Distilling into Student (CM) ---")
    student_model = SimpleMLP().to(device)
    student_opt = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    
    cm_method = ConsistencyModel(sigma_max=5.0)
    
    distill_trainer = ConsistencyDistillationTrainer(
        method=cm_method,
        model=student_model,
        optimizer=student_opt,
        teacher_model=teacher_model,
        feature_keys=["features"],
        label_keys=[],
        device=device
    )
    
    distill_trainer.fit(train_loader, epochs=30)
    
    # 3. Verify
    sampler = ConsistencySampler(cm_method, student_model, device)
    samples = sampler.sample(2000, [2])
    samples_np = samples.detach().cpu().numpy()
    
    X_plot = X[:2000]
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Distilled (1-step)", s=10)
    plt.legend()
    plt.title("Two Moons - Consistency Distillation")
    plt.savefig("examples/two_moons_distill.png")
    print("Saved plot.")

if __name__ == "__main__":
    main()
