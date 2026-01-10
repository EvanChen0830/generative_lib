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
        self.method = method
        self.model = model.to(device)
        self.optimizer = optimizer
        self.feature_keys = feature_keys
        self.label_keys = label_keys
        self.device = device
        self.tracker = tracker
        self.scheduler = scheduler
        self.use_ema = use_ema
        self.grad_clip = grad_clip
        
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 100):
        """Main training loop."""
        print(f"Starting training on {self.device} for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
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
                self.tracker.save_checkpoint(self.model, self.optimizer, epoch, metric_val)

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
        total_loss = 0.0
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch in pbar:
            if not isinstance(batch, dict):
                 raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)

            self.optimizer.zero_grad()
            
            # Compute Loss via Method (Physics)
            loss_dict = self.method.compute_loss(self.model, x, cond)
            loss = loss_dict["loss"]
            
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

        return {"loss": total_loss / count}

    def _validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs validation."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in loader:
                x, cond = self._process_batch(batch)

                loss_dict = self.method.compute_loss(self.model, x, cond)
                total_loss += loss_dict["loss"].item()
                count += 1
                
        return {"loss": total_loss / count}
