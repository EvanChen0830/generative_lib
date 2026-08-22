import argparse
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from generative_lib.flow_matching.method.flow_matching import FlowMatching
from generative_lib.flow_matching.trainer.base import BaseFlowMatchingTrainer
from generative_lib.flow_matching.sampler.base import BaseFlowMatchingSampler
from generative_lib.metrics.distance import calculate_frechet_distance, compute_statistics

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


def build_parser() -> argparse.ArgumentParser:
    """Builds the flow-matching example command-line parser."""
    parser = argparse.ArgumentParser(description="Two-moons flow-matching example")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser

def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or Path(__file__).resolve().parent.parent / "runs" / "examples" / "two_moons_flow"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "two_moons_flow.png"
    
    # 2. Prepare Data
    X, _ = make_moons(n_samples=args.train_size, noise=0.05, random_state=args.seed)
    train_loader = DataLoader(
        [{"features": x} for x in torch.from_numpy(X).float()], 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # 3. Setup Components
    model = SimpleMLP(hidden_features=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    method = FlowMatching(sigma_min=0.0)
    
    trainer = BaseFlowMatchingTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=[],
        label_keys=["features"], 
        device=device,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )
    
    # 4. Train
    print(f"Training Flow Matching ({args.train_size} samples, {args.epochs} epochs)...")
    trainer.fit(train_loader, epochs=args.epochs)
    
    # 5. Sample
    print("Sampling...")
    sampler = BaseFlowMatchingSampler(method, trainer.get_sampling_model(), device, steps=args.sample_steps)
    
    # Unconditional sampling check (or dummy conditional since we didn't train robustly on Cond for Flow yet?)
    # Wait, the Flow Trainer example above was UNCONDITIONAL (label_keys=[]).
    # X, _ = make_moons. Labels usage was implicit or ignored in training?
    # In two_moons_flow.py (Step 96), `label_keys=[]`. So it's unconditional.
    # So sample(num_samples=500) works as batch size.
    samples = sampler.sample(num_samples=args.num_samples, shape=[2])
    # Returns [500, 2] because B=1.
    
    # 6. Verify
    samples_np = samples.detach().cpu().numpy()
    
    print("Computing metrics...")
    mu_real, sig_real = compute_statistics(X)
    mu_gen, sig_gen = compute_statistics(samples_np)
    fd_flow = calculate_frechet_distance(mu_real, sig_real, mu_gen, sig_gen)
    print(f"FD Flow Matching: {fd_flow:.4f}")
    
    # Plot results
    # Downsample Data for plot clarity
    X_plot = X[:2000]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_plot[:, 0], X_plot[:, 1], alpha=0.3, label="Data", s=10)
    plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.8, label="Generated")
    plt.legend()
    plt.title(f"Two Moons - Flow Matching (20k Data, 200 Epochs)\nFD: {fd_flow:.4f}")
    plt.savefig(figure_path)
    print(f"Saved plot to {figure_path}")

if __name__ == "__main__":
    main()
