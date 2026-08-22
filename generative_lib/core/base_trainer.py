from abc import ABC
import copy
from typing import Callable, Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_method import BaseMethod

class BaseTrainer(ABC):
    """Abstract base trainer for generative models.

    Handles the boilerplate training loop:
    - Iterating epochs
    - Iterating batches
    - Optimization step
    - Validation loop
    - Model checkpointing (via Tracker, if provided)
    - Logging (via Logger, if provided)
    """

    def __init__(
        self,
        method: BaseMethod,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_keys: List[str],
        label_keys: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tracker: Optional[Any] = None, # Type assumed to be utils.tracker.ModelTracker
        scheduler: Optional[Any] = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        grad_clip: float = 1.0,
        early_stop_fn: Optional[Callable[[Dict[str, float]], bool]] = None,
    ):
        self.method = method.to(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.feature_keys = feature_keys
        self.label_keys = label_keys
        self.device = device
        self.tracker = tracker
        self.scheduler = scheduler
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_model = copy.deepcopy(self.model).to(device) if use_ema else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
        self.grad_clip = grad_clip
        self.early_stop_fn = early_stop_fn

    def get_sampling_model(self) -> nn.Module:
        """Returns the EMA model when enabled, otherwise the online model.

        Returns:
            Model to pass to an inference sampler.
        """
        return self.ema_model if self.ema_model is not None else self.model

    def _update_ema_model(self) -> None:
        """Updates the optional evaluation EMA after an optimizer step."""
        if self.ema_model is None:
            return
        with torch.no_grad():
            for model_param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.copy_(ema_param * self.ema_decay + model_param * (1 - self.ema_decay))
            for model_buffer, ema_buffer in zip(self.model.buffers(), self.ema_model.buffers()):
                ema_buffer.copy_(model_buffer)
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume: bool = False,
        early_stop_fn: Optional[Callable[[Dict[str, float]], bool]] = None,
    ):
        """Runs training until completion or the stopping criterion is met.

        Args:
            train_loader: Data loader used for optimization.
            val_loader: Optional data loader used for validation.
            epochs: Final epoch number to train through.
            resume: Whether to restore ``last.pt`` before training.
            early_stop_fn: Optional predicate receiving merged epoch metrics.
        """
        stop_fn = early_stop_fn or self.early_stop_fn
        start_epoch = 1
        
        # Resume Logic
        if resume and self.tracker:
            checkpoint = self.tracker.load_last(self.model, self.optimizer)
            if checkpoint:
                ema_state = checkpoint.get("ema_model_state")
                if self.ema_model is not None and ema_state is not None:
                    self.ema_model.load_state_dict(ema_state)
                scheduler_state = checkpoint.get("scheduler_state")
                if self.scheduler is not None and scheduler_state is not None:
                    self.scheduler.load_state_dict(scheduler_state)
                start_epoch = checkpoint.get("epoch", 0) + 1
                run_id = checkpoint.get("run_id")
                if run_id and self.tracker.logger and hasattr(self.tracker.logger, "resume"):
                    self.tracker.logger.resume(run_id)
                
        print(f"Starting training on {self.device} from epoch {start_epoch} to {epochs}.")
        
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
            
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader, epoch)

            epoch_metrics = {**train_metrics, **val_metrics}
            if self.tracker and self.tracker.logger:
                 self.tracker.logger.log_metrics(epoch_metrics, step=epoch)

            # Print progress
            log_str = f"Epoch {epoch}/{epochs} | "
            log_str += " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                 log_str += " | " + " ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(log_str)

            # Checkpoint
            if self.tracker:
                # Prefer validation loss when available, otherwise fall back to training loss.
                metric_val = val_metrics.get("val_loss", train_metrics.get("train_loss", 0.0))
                # Get run_id from Logger if exists
                run_id = self.tracker.logger.run_id if (self.tracker.logger and hasattr(self.tracker.logger, 'run_id')) else None
                self.tracker.save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    metric_val,
                    run_id,
                    extra_state={
                        "ema_model_state": self.ema_model.state_dict() if self.ema_model is not None else None,
                        "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                    },
                )

            if stop_fn and stop_fn(epoch_metrics):
                print(f"Early stopping at epoch {epoch}.")
                return
                
    def _process_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extracts features and labels from batch."""
        # Extract Features
        # feats = []
        # print(batch)
        cond = []
        for k in self.feature_keys:
            if k in batch:
                cond.append(batch[k].to(self.device))
            else:
                raise ValueError(f"Feature key '{k}' not found in batch keys: {list(batch.keys())}")
        # print(len(cond))
        if cond:
            if len(cond) > 1:
                cond = torch.cat(cond, dim=-1)
            else:
                cond = cond[0]
        else:
            cond = None

        # Extract Labels (x)
        x = None
        if self.label_keys:
            labels = []
            for k in self.label_keys:
                if k in batch:
                    val = batch[k].to(self.device)
                    # Create mask for NaNs if necessary, or just assume valid data
                    labels.append(val)
                else:
                    raise ValueError(f"Label key '{k}' not found in batch keys: {list(batch.keys())}")
        
            
            if labels:
                if len(labels) > 1:
                    x = torch.cat(labels, dim=-1)
                else:
                    x = labels[0]
        # print(x, cond)
        return x, cond

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs one epoch of training."""
        self.model.train()
        
        # Accumulators for all metric keys
        # We don't know keys ahead of time, so use dict
        total_metrics = {}
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch in pbar:
            if not isinstance(batch, dict):
                 raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)

            self.optimizer.zero_grad()
            
            # Compute Loss via Method (Physics)
            loss_dict = self.method.compute_loss(self.model, x, cond)
            
            # backward on "loss" key (convention) OR sum of all? 
            if "loss" in loss_dict:
                final_loss = loss_dict["loss"]
            else:
                final_loss = sum(loss_dict.values())
                loss_dict["loss"] = final_loss # Record total
            
            final_loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            self._update_ema_model()
            
            # Accumulate
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            # Update progress bar
            pbar.set_postfix({"loss": final_loss.item()})
            
        return {f"train_{k}": v / count for k, v in total_metrics.items()}

    def _validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs validation."""
        self.model.eval()
        total_metrics = {}
        count = 0
        
        with torch.no_grad():
            for batch in loader:
                x, cond = self._process_batch(batch)

                loss_dict = self.method.compute_loss(self.model, x, cond)
                
                # Ensure 'loss' key exists for consistency
                if "loss" not in loss_dict:
                    loss_dict["loss"] = sum(loss_dict.values())
                    
                count += 1
                for k, v in loss_dict.items():
                    if k not in total_metrics: total_metrics[k] = 0.0
                    total_metrics[k] += v.item()
                
        return {f"val_{k}": v / count for k, v in total_metrics.items()}
