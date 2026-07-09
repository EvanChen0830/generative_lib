import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader

from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.sampler.base import BaseConsistencyModelSampler
from generative_lib.consistency_model.trainer.base import BaseConsistencyModelTrainer
from generative_lib.metrics.distance import calculate_frechet_distance, compute_statistics
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class UnconditionalMLP(nn.Module):
    def __init__(self, data_dim=2, time_dim=192, hidden_dim=768):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, condition=None):
        if t.ndim == 2:
            t = t.squeeze(-1)
        t_emb = self.time_mlp(t.float())
        return self.net(torch.cat([x, t_emb], dim=-1))


class ConditionalMLP(nn.Module):
    def __init__(self, data_dim=2, cond_dim=128, time_dim=192, hidden_dim=768):
        super().__init__()
        self.cond_emb = nn.Embedding(3, cond_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, condition):
        if t.ndim == 2:
            t = t.squeeze(-1)
        t_emb = self.time_mlp(t.float())
        cond_ids = condition.squeeze(-1).long()
        cond_ids = torch.where(cond_ids < 0, torch.full_like(cond_ids, 2), cond_ids.clamp(0, 1))
        cond_emb = self.cond_emb(cond_ids)
        return self.net(torch.cat([x, cond_emb, t_emb], dim=-1))


def build_parser():
    repo_root = Path(__file__).resolve().parent.parent
    default_output_dir = repo_root / "runs" / "examples" / "two_moons_consistency"
    parser = argparse.ArgumentParser(description="Conditional two-moons consistency model example")
    parser.add_argument("--mode", choices=["conditional", "unconditional"], default="conditional")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=250)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--time-dim", type=int, default=192)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--num-scales", type=int, default=120)
    parser.add_argument("--min-scales", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=2)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--target-ema-start", type=float, default=0.95)
    parser.add_argument("--unconditional-value", type=float, default=-1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--cond-weight", type=float, default=1.0)
    parser.add_argument("--uncond-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="Consistency_TwoMoons")
    parser.add_argument("--project-name", type=str, default="TwoMoons")
    parser.add_argument("--checkpoint-subdir", type=str, default="consistency")
    parser.add_argument("--save-every", type=int, default=25)
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir = output_dir / "mlruns"
    checkpoint_dir = output_dir / "checkpoints" / args.checkpoint_subdir
    suffix = "cond" if args.mode == "conditional" else "uncond"
    figure_path = output_dir / f"two_moons_consistency_{suffix}_comparison.png"
    samples_path = output_dir / f"two_moons_consistency_{suffix}_samples.npz"
    summary_path = output_dir / f"two_moons_consistency_{suffix}_metrics.txt"

    X, y = make_moons(n_samples=args.train_size, noise=0.05, random_state=args.seed)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    dataset_list = []
    for i in range(len(X)):
        item = {"position": torch.tensor(X_norm[i]).float()}
        if args.mode == "conditional":
            item["class"] = torch.tensor([y[i]]).float()
        dataset_list.append(item)
    train_loader = DataLoader(dataset_list, batch_size=args.batch_size, shuffle=True)

    if args.mode == "conditional":
        model = ConditionalMLP(
            data_dim=2,
            cond_dim=args.cond_dim,
            time_dim=args.time_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        feature_keys = ["class"]
    else:
        model = UnconditionalMLP(
            data_dim=2,
            time_dim=args.time_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        feature_keys = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_schedule_epochs = max(args.pretrain_epochs + args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_schedule_epochs)
    method = ConsistencyModel(
        num_scales=args.num_scales,
        min_scales=args.min_scales,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        target_ema_start=args.target_ema_start,
        unconditional_value=args.unconditional_value,
        cond_weight=args.cond_weight,
        uncond_weight=args.uncond_weight,
    )

    logger = Logger(
        project_name=args.project_name,
        run_name=args.run_name,
        use_mlflow=True,
        mlflow_uri=args.mlflow_uri or f"file:{mlruns_dir}",
    )
    logger.log_params(
        {
            "epochs": args.epochs,
            "pretrain_epochs": args.pretrain_epochs,
            "mode": args.mode,
            "train_size": args.train_size,
            "batch_size": args.batch_size,
            "num_samples_per_class": args.num_samples,
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "cond_dim": args.cond_dim,
            "num_scales": args.num_scales,
            "min_scales": args.min_scales,
            "sample_steps": args.sample_steps,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "sigma_data": args.sigma_data,
            "target_ema_start": args.target_ema_start,
            "unconditional_value": args.unconditional_value,
            "guidance_scale": args.guidance_scale,
            "cond_weight": args.cond_weight,
            "uncond_weight": args.uncond_weight,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "seed": args.seed,
            "device": device,
            "method": "consistency_model",
        }
    )
    logger.set_tags({"example": "two_moons_consistency", "method": "consistency_model"})

    tracker = ModelTracker(
        exp_name=args.project_name,
        model_name="Consistency",
        save_dir=str(checkpoint_dir),
        logger=logger,
        save_every_n_epochs=args.save_every,
    )
    trainer = BaseConsistencyModelTrainer(
        method=method,
        model=model,
        optimizer=optimizer,
        feature_keys=feature_keys,
        label_keys=["position"],
        device=device,
        tracker=tracker,
        scheduler=scheduler,
        ema_decay=args.ema_decay,
    )

    print(f"Training {args.mode.title()} Consistency Model...")
    if args.pretrain_epochs > 0:
        print("Running diffusion pretraining...")
        trainer.pretrain_diffusion(train_loader, epochs=args.pretrain_epochs)
    trainer.fit(train_loader, epochs=args.epochs, resume=args.resume)

    print("Sampling Comparison...")
    sampler = BaseConsistencyModelSampler(
        method,
        trainer.ema_model,
        device,
        steps=args.sample_steps,
        feature_keys=feature_keys,
        guidance_scale=args.guidance_scale,
        unconditional_value=args.unconditional_value,
    )
    if args.mode == "conditional":
        cond_0 = torch.zeros(1, 1)
        cond_1 = torch.ones(1, 1)
        samples_0 = sampler.sample(num_samples=args.num_samples, shape=[2], condition=cond_0).squeeze(0).detach().cpu().numpy()
        samples_1 = sampler.sample(num_samples=args.num_samples, shape=[2], condition=cond_1).squeeze(0).detach().cpu().numpy()
        samples = np.concatenate([samples_0, samples_1], axis=0)
        labels = np.concatenate(
            [np.zeros((args.num_samples, 1), dtype=np.float32), np.ones((args.num_samples, 1), dtype=np.float32)],
            axis=0,
        )
    else:
        samples = sampler.sample(num_samples=args.num_samples * 2, shape=[2]).detach().cpu().numpy()
        labels = None

    print("Computing metrics...")
    mu_real, sig_real = compute_statistics(X_norm)
    mu_gen, sig_gen = compute_statistics(samples)
    fd_val = calculate_frechet_distance(mu_real, sig_real, mu_gen, sig_gen)
    print(f"FD Consistency: {fd_val:.4f}")

    denorm = samples * X_std + X_mean
    fig, ax = plt.subplots(figsize=(6, 6))
    if args.mode == "conditional":
        ax.scatter(denorm[: args.num_samples, 0], denorm[: args.num_samples, 1], c="cyan", label="Class 0", alpha=0.6, s=10)
        ax.scatter(denorm[args.num_samples :, 0], denorm[args.num_samples :, 1], c="orange", label="Class 1", alpha=0.6, s=10)
        ax.legend()
    else:
        ax.scatter(denorm[:, 0], denorm[:, 1], c="cyan", label="Generated", alpha=0.6, s=10)
        ax.legend()
    ax.set_title(f"Consistency Model ({args.mode})\nFD: {fd_val:.4f}")
    fig.tight_layout()
    fig.savefig(figure_path)

    np.savez(
        samples_path,
        x_mean=X_mean,
        x_std=X_std,
        samples=samples,
        labels=labels,
    )
    summary_lines = [
        "Method: consistency_model",
        f"Mode: {args.mode}",
        f"FD Consistency: {fd_val:.4f}",
        f"Guidance Scale: {args.guidance_scale}",
        f"Figure: {figure_path}",
        f"Samples: {samples_path}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    summary_path.write_text(summary_text)
    logger.log_metrics({"fd_consistency": float(fd_val)}, step=args.epochs)
    logger.log_artifact(str(figure_path))
    logger.log_artifact(str(samples_path))
    logger.log_text(summary_text, summary_path.name)
    plt.close(fig)

    print(f"Saved plot to {figure_path}")
    print(f"Saved samples to {samples_path}")
    print(summary_text, end="")


if __name__ == "__main__":
    main()
