import torch
import torch.nn as nn
import os
from typing import Optional, Any, Dict
from .logger import Logger

class ModelTracker:
    """Tracks model performance and saves checkpoints (Best & Last)."""

    def __init__(
        self,
        exp_name: str, # Not directly used for saving, but useful context
        model_name: str, # Not directly used for saving, but useful context
        save_dir: str,
        logger: Optional[Logger] = None,
        best_metric: str = "loss",
        mode: str = "min",
        save_every_n_epochs: Optional[int] = None,
    ):
        self.save_dir = save_dir
        self.logger = logger
        self.best_metric = best_metric
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        
        # Initialize best score
        self.best_score = float('inf') if mode == "min" else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)
        self._restore_best_score()

    def _restore_best_score(self):
        """Restores the best metric from disk when available."""
        best_path = os.path.join(self.save_dir, "best.pt")
        if not os.path.exists(best_path):
            return

        checkpoint = torch.load(best_path, map_location="cpu")
        metric = checkpoint.get("metric")
        if metric is not None:
            self.best_score = metric

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_val: float,
        run_id: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ):
        """Saves 'last.pt' and optionally 'best.pt'."""
        
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": metric_val,
            "run_id": run_id
        }
        if extra_state:
            state.update(extra_state)
        
        # Save Last
        last_path = os.path.join(self.save_dir, "last.pt")
        torch.save(state, last_path)

        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            epoch_path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.pt")
            torch.save(state, epoch_path)
        
        # Check Best
        is_best = False
        if self.mode == "min":
            if metric_val < self.best_score:
                is_best = True
        else: # max
            if metric_val > self.best_score:
                is_best = True
        
        if is_best:
            self.best_score = metric_val
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(state, best_path)

    def load_last(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Loads weights from last.pt and returns state info."""
        path = os.path.join(self.save_dir, "last.pt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Returning empty state.")
            return {}
            
        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        print(f"Loaded last model from {path} (Epoch {checkpoint['epoch']})")
        return checkpoint

    def load_best(self, model: nn.Module) -> Dict[str, Any]:
        """Loads weights from best.pt and returns state info."""
        path = os.path.join(self.save_dir, "best.pt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Returning empty state.")
            return {}
            
        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state"])
        
        print(f"Loaded best model from {path} (Epoch {checkpoint['epoch']})")
        return checkpoint
