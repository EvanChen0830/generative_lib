import copy
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...core.base_trainer import BaseTrainer


class BaseConsistencyModelTrainer(BaseTrainer):
    """Trainer for consistency models with an EMA target teacher.

    The target model is the paper's ``theta_minus`` model.  It is both the
    stop-gradient teacher during training and the model used for generation.
    ``ema_model`` is retained for diffusion pretraining compatibility only.
    """

    def __init__(self, *args, ema_decay: float = 0.999, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.ema_decay = ema_decay
        for param in self.ema_model.parameters():
            param.requires_grad = False
        for param in self.target_model.parameters():
            param.requires_grad = False

    def get_sampling_model(self) -> torch.nn.Module:
        """Returns the EMA target model used for consistency-model sampling.

        Returns:
            The target-teacher model (``theta_minus``).
        """
        return self.target_model

    def _update_ema_model(self) -> None:
        with torch.no_grad():
            for p_model, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
                p_ema.copy_(p_ema * self.ema_decay + p_model * (1 - self.ema_decay))
            for b_model, b_ema in zip(self.model.buffers(), self.ema_model.buffers()):
                b_ema.copy_(b_model)

    def _update_target_model(self, rate: float) -> None:
        with torch.no_grad():
            for p_model, p_target in zip(self.model.parameters(), self.target_model.parameters()):
                p_target.copy_(p_target * rate + p_model * (1 - rate))

    def _train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        self.ema_model.eval()
        self.target_model.train()

        total_metrics = {}
        count = 0
        total_steps = max(total_epochs * len(loader), 1)

        pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if not isinstance(batch, dict):
                raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)
            self.optimizer.zero_grad()

            global_step = (epoch - 1) * len(loader) + batch_idx
            self.method.set_training_progress(global_step, total_steps)

            loss_dict = self.method.compute_loss(
                self.model,
                x,
                cond,
                teacher_model=self.target_model,
            )
            final_loss = loss_dict["loss"]
            final_loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self._update_ema_model()
            self._update_target_model(self.method.compute_target_ema_rate())

            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})

        return {f"train_{k}": v / count for k, v in total_metrics.items()}

    def _diffusion_train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        self.ema_model.eval()

        total_metrics = {}
        count = 0
        total_steps = max(total_epochs * len(loader), 1)

        pbar = tqdm(loader, desc=f"Pretrain {epoch} Train", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if not isinstance(batch, dict):
                raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)
            self.optimizer.zero_grad()

            global_step = (epoch - 1) * len(loader) + batch_idx
            self.method.set_training_progress(global_step, total_steps)

            loss_dict = self.method.compute_diffusion_loss(self.model, x, cond)
            final_loss = loss_dict["loss"]
            final_loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self._update_ema_model()

            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics:
                    total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})

        return {f"pretrain_{k}": v / count for k, v in total_metrics.items()}

    def pretrain_diffusion(self, train_loader: DataLoader, epochs: int) -> None:
        print(f"Starting diffusion pretraining on {self.device} for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            train_metrics = self._diffusion_train_epoch(train_loader, epoch, epochs)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.tracker and self.tracker.logger:
                self.tracker.logger.log_metrics(train_metrics, step=epoch)
            log_str = f"Pretrain {epoch}/{epochs} | "
            log_str += " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            print(log_str)

        self.model.load_state_dict(self.ema_model.state_dict())
        self.target_model.load_state_dict(self.ema_model.state_dict())

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume: bool = False,
        early_stop_fn: Optional[Callable[[Dict[str, float]], bool]] = None,
    ):
        stop_fn = early_stop_fn or self.early_stop_fn
        start_epoch = 1

        if resume and self.tracker:
            checkpoint = self.tracker.load_last(self.model, self.optimizer)
            if checkpoint:
                ema_state = checkpoint.get("ema_model_state")
                target_state = checkpoint.get("target_model_state")
                if ema_state:
                    self.ema_model.load_state_dict(ema_state)
                else:
                    self.ema_model.load_state_dict(self.model.state_dict())
                if target_state:
                    self.target_model.load_state_dict(target_state)
                else:
                    self.target_model.load_state_dict(self.model.state_dict())
                scheduler_state = checkpoint.get("scheduler_state")
                if scheduler_state and self.scheduler is not None:
                    self.scheduler.load_state_dict(scheduler_state)

                start_epoch = checkpoint.get("epoch", 0) + 1
                run_id = checkpoint.get("run_id")
                if run_id and self.tracker.logger and hasattr(self.tracker.logger, "resume"):
                    self.tracker.logger.resume(run_id)

        print(f"Starting training on {self.device} from epoch {start_epoch} to {epochs}.")

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch, epochs)

            if self.scheduler is not None:
                self.scheduler.step()

            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader, epoch)

            epoch_metrics = {**train_metrics, **val_metrics}
            if self.tracker and self.tracker.logger:
                self.tracker.logger.log_metrics(epoch_metrics, step=epoch)

            log_str = f"Epoch {epoch}/{epochs} | "
            log_str += " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                log_str += " | " + " ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(log_str)

            if self.tracker:
                metric_val = val_metrics.get("val_loss", train_metrics.get("train_loss", 0.0))
                run_id = self.tracker.logger.run_id if (self.tracker.logger and hasattr(self.tracker.logger, "run_id")) else None
                self.tracker.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    metric_val,
                    run_id,
                    extra_state={
                        "ema_model_state": self.ema_model.state_dict(),
                        "target_model_state": self.target_model.state_dict(),
                        "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                    },
                )

            if stop_fn and stop_fn(epoch_metrics):
                print(f"Early stopping at epoch {epoch}.")
                return
