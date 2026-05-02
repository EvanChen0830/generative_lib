import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader

from generative_lib.diffusion.method.cfg_diffusion import CFGDiffusion
from generative_lib.diffusion.method.gaussian_diffusion import GaussianDiffusion
from generative_lib.diffusion.sampler.base import BaseDiffusionSampler
from generative_lib.diffusion.trainer.base import BaseDiffusionTrainer
from generative_lib.metrics.distance import calculate_frechet_distance, compute_statistics
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker

# 1. Model Definition
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
    def __init__(self, data_dim=2, cond_dim=64, time_dim=128, hidden_dim=256):
        super().__init__()
        self.cond_emb = nn.Embedding(3, cond_dim)
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

        # Map class labels {0, 1} and unconditional token {-1} into embedding ids.
        cond_ids = condition.squeeze(-1).long()
        cond_ids = torch.where(cond_ids < 0, torch.full_like(cond_ids, 2), cond_ids)
        cond_emb = self.cond_emb(cond_ids)

        inp = torch.cat([x, cond_emb, t_emb], dim=-1)
        return self.net(inp)
def build_parser():
    repo_root = Path(__file__).resolve().parent.parent
    default_output_dir = repo_root / "runs" / "examples" / "two_moons_diffusion"
    parser = argparse.ArgumentParser(description="Two-moons diffusion example")
    parser.add_argument("--method", choices=["cfg", "gaussian"], default="cfg")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampler-steps", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--cond-dim", type=int, default=64)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="CFG_DDIM_Check")
    parser.add_argument("--project-name", type=str, default="TwoMoons")
    parser.add_argument("--checkpoint-subdir", type=str, default="cfg_diff")
    parser.add_argument("--save-every", type=int, default=25)
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = (args.output_dir / args.method).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir = output_dir / "mlruns"
    checkpoint_dir = output_dir / "checkpoints" / f"{args.checkpoint_subdir}_{args.method}"
    figure_path = output_dir / f"two_moons_{args.method}_comparison.png"
    samples_path = output_dir / f"two_moons_{args.method}_samples.npz"
    summary_path = output_dir / f"two_moons_{args.method}_metrics.txt"

    X, y = make_moons(n_samples=args.train_size, noise=0.05, random_state=args.seed)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    dataset_list = [
        {
            "position": torch.tensor(X_norm[i]).float(),
            "class": torch.tensor([y[i]]).float(),
        }
        for i in range(len(X))
    ]

    train_loader = DataLoader(dataset_list, batch_size=args.batch_size, shuffle=True)

    model = SimpleMLP(
        data_dim=2,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if args.method == "cfg":
        method = CFGDiffusion(timesteps=1000, schedule="linear", unconditional_value=-1.0)
        method_name = "cfg_diffusion"
        model_name = "CFG_Diff"
        train_label = "Training CFG Diffusion (Dual Loss)..."
    else:
        method = GaussianDiffusion(timesteps=1000, schedule="linear")
        method_name = "gaussian_diffusion"
        model_name = "Gaussian_Diff"
        train_label = "Training Gaussian Diffusion..."

    logger = Logger(
        project_name=args.project_name,
        run_name=args.run_name,
        use_mlflow=True,
        mlflow_uri=args.mlflow_uri or f"file:{mlruns_dir}",
    )
    logger.log_params(
        {
            "epochs": args.epochs,
            "num_samples_per_class": args.num_samples,
            "train_size": args.train_size,
            "batch_size": args.batch_size,
            "sampler_steps": args.sampler_steps,
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "cond_dim": args.cond_dim,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "device": device,
            "method": args.method,
        }
    )
    logger.set_tags(
        {
            "example": "two_moons_diffusion",
            "method": method_name,
        }
    )

    tracker = ModelTracker(
        exp_name=args.project_name,
        model_name=model_name,
        save_dir=str(checkpoint_dir),
        logger=logger,
        save_every_n_epochs=args.save_every,
    )

    trainer = BaseDiffusionTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=["class"],
        label_keys=["position"],
        device=device,
        tracker=tracker,
    )

    print(train_label)
    trainer.fit(train_loader, epochs=args.epochs, resume=args.resume)

    print("Sampling Comparison...")
    cond_0 = torch.zeros(args.num_samples, 1)
    cond_1 = torch.ones(args.num_samples, 1)
    cond = torch.cat([cond_0, cond_1], dim=0)

    sampler_ddpm = BaseDiffusionSampler(
        method, model, device, steps=args.sampler_steps, sampler_type="ddpm", guidance_scale=1.0, feature_keys=["class"]
    )
    s_ddpm = sampler_ddpm.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()

    sampler_ddim = BaseDiffusionSampler(
        method, model, device, steps=args.sampler_steps, sampler_type="ddim", guidance_scale=1.0, feature_keys=["class"]
    )
    s_ddim = sampler_ddim.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()

    print("Computing metrics...")
    mu_real, sig_real = compute_statistics(X_norm)
    mu_ddpm, sig_ddpm = compute_statistics(s_ddpm)
    fd_ddpm = calculate_frechet_distance(mu_real, sig_real, mu_ddpm, sig_ddpm)

    mu_ddim, sig_ddim = compute_statistics(s_ddim)
    fd_ddim = calculate_frechet_distance(mu_real, sig_real, mu_ddim, sig_ddim)

    print(f"FD DDPM (w=1.0): {fd_ddpm:.4f}")
    print(f"FD DDIM (w=1.0): {fd_ddim:.4f}")
    metrics = {
        "fd_ddpm": float(fd_ddpm),
        "fd_ddim": float(fd_ddim),
    }
    samples_to_save = {
        "ddpm": s_ddpm,
        "ddim": s_ddim,
    }
    plot_specs = [
        ("DDPM (w=1.0)", s_ddpm, fd_ddpm),
        ("DDIM (w=1.0)", s_ddim, fd_ddim),
    ]

    if args.method == "cfg":
        sampler_cfg = BaseDiffusionSampler(
            method,
            model,
            device,
            steps=args.sampler_steps,
            sampler_type="ddim",
            guidance_scale=args.guidance_scale,
            unconditional_value=-1.0,
            feature_keys=["class"],
        )
        s_cfg = sampler_cfg.sample(num_samples=1, shape=[2], condition=cond).squeeze(1).detach().cpu().numpy()
        mu_cfg, sig_cfg = compute_statistics(s_cfg)
        fd_cfg = calculate_frechet_distance(mu_real, sig_real, mu_cfg, sig_cfg)
        print(f"FD DDIM CFG (w={args.guidance_scale:.1f}): {fd_cfg:.4f}")
        metrics["fd_cfg"] = float(fd_cfg)
        samples_to_save["cfg"] = s_cfg
        plot_specs.append((f"DDIM CFG (w={args.guidance_scale:.1f})", s_cfg, fd_cfg))

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(6 * len(plot_specs), 6))
    if len(plot_specs) == 1:
        axes = [axes]

    def plot_data(ax, data, title, fd):
        denorm = data * X_std + X_mean
        ax.scatter(denorm[: args.num_samples, 0], denorm[: args.num_samples, 1], c="cyan", label="Class 0", alpha=0.6, s=10)
        ax.scatter(denorm[args.num_samples :, 0], denorm[args.num_samples :, 1], c="orange", label="Class 1", alpha=0.6, s=10)
        ax.set_title(f"{title}\nFD: {fd:.4f}")
        ax.legend()

    for ax, (title, data, fd) in zip(axes, plot_specs):
        plot_data(ax, data, title, fd)

    plt.tight_layout()
    fig.savefig(figure_path)
    np.savez(
        samples_path,
        x_mean=X_mean,
        x_std=X_std,
        labels=cond.numpy(),
        **samples_to_save,
    )
    summary_lines = [
        f"Method: {args.method}",
        f"FD DDPM (w=1.0): {fd_ddpm:.4f}",
        f"FD DDIM (w=1.0): {fd_ddim:.4f}",
    ]
    if args.method == "cfg":
        summary_lines.append(f"FD DDIM CFG (w={args.guidance_scale:.1f}): {metrics['fd_cfg']:.4f}")
    summary_lines.extend(
        [
            f"Figure: {figure_path}",
            f"Samples: {samples_path}",
        ]
    )
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text)

    logger.log_metrics(metrics, step=args.epochs)
    logger.log_artifact(str(figure_path), artifact_path="figures")
    logger.log_artifact(str(samples_path), artifact_path="samples")
    logger.log_artifact(str(summary_path), artifact_path="reports")
    logger.finish()

    print(f"Saved plot to {figure_path}")
    print(f"Saved samples to {samples_path}")
    print(summary_text)
    plt.close(fig)

if __name__ == "__main__":
    main()
