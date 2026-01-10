import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import math

from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker
from generative_lib.evaluator.evaluator import Evaluator

# 1. Define User Model
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
    X, y = make_moons(n_samples=500, noise=0.05)
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
    method = GaussianDiffusion(timesteps=1000, schedule="linear")
    
    logger = Logger(project_name="EvalTest", run_name="FID_ReCheck", use_mlflow=True, mlflow_uri="file:./examples/mlruns")
    tracker = ModelTracker(exp_name="EvalTest", model_name="Diff", save_dir="./checkpoints/eval_test", logger=logger)
    
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
    print("Training 1 epoch...")
    trainer.fit(train_loader, epochs=1)
    
    # 5. Evaluate
    print("Evaluating...")
    sampler = BaseDiffusionSampler(method, model, device, steps=10, label_keys=["class"])
    
    generated_data = sampler.batch_sample(num_samples=1, shape=[2], dataloader=train_loader)
    
    evaluator = Evaluator(feature_key="position", logger=logger)
    metrics = evaluator.evaluate(generated_data, train_loader, step=1)
    
    print(f"Computed Metrics: {metrics}")
    logger.finish()

if __name__ == "__main__":
    main()
