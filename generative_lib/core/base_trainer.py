from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
        grad_clip: float = 1.0,
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
        self.grad_clip = grad_clip
        
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 100, resume: bool = False):
        """Main training loop."""
        start_epoch = 1
        
        # Resume Logic
        if resume and self.tracker:
            checkpoint = self.tracker.load_last(self.model, self.optimizer)
            if checkpoint:
                start_epoch = checkpoint.get("epoch", 0) + 1
                run_id = checkpoint.get("run_id")
                # If tracker has a logger, we update its run_id if not already set?
                # Actually, Logger init handles run_id. If we resume, we should have passed run_id to Logger init?
                
        print(f"Starting training on {self.device} from epoch {start_epoch} to {epochs}.")
        
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Log training metrics
            if self.tracker and self.tracker.logger:
                 self.tracker.logger.log_metrics(train_metrics, step=epoch)
            
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader, epoch)
                if self.tracker and self.tracker.logger:
                    self.tracker.logger.log_metrics(val_metrics, step=epoch)

            # Print progress
            log_str = f"Epoch {epoch}/{epochs} | "
            log_str += " ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                 log_str += " | " + " ".join([f"Val_{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(log_str)

            # Checkpoint
            if self.tracker:
                # We save based on the first metric in val_metrics if available, otherwise 'loss'
                metric_val = val_metrics.get("loss", train_metrics.get("loss", 0.0))
                # Get run_id from Logger if exists
                run_id = self.tracker.logger.run_id if (self.tracker.logger and hasattr(self.tracker.logger, 'run_id')) else None
                self.tracker.save_checkpoint(self.model, self.optimizer, epoch, metric_val, run_id)

    def _process_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extracts features and labels from batch."""
        # Extract Features
        feats = []
        for k in self.feature_keys:
            if k in batch:
                feats.append(batch[k].to(self.device))
            else:
                raise ValueError(f"Feature key '{k}' not found in batch keys: {list(batch.keys())}")
        
        if len(feats) > 1:
            x = torch.cat(feats, dim=-1)
        else:
            x = feats[0]

        # Extract Labels (Condition)
        cond = None
        if self.label_keys:
            labels = []
            for k in self.label_keys:
                if k in batch:
                    val = batch[k].to(self.device)
                    # Create mask for NaNs if necessary, or just assume valid data
                    labels.append(val)
            
            if labels:
                if len(labels) > 1:
                    cond = torch.cat(labels, dim=-1)
                else:
                    cond = labels[0]
        
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
            
            # Accumulate
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            # Update progress bar
            pbar.set_postfix({"loss": final_loss.item()})
            
        return {k: v / count for k, v in total_metrics.items()}

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
                
        return {k: v / count for k, v in total_metrics.items()}
