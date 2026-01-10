import torch
import torch.nn as nn
import os
import copy
from typing import Optional, Any
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
        mode: str = "min"
    ):
        self.save_dir = save_dir
        self.logger = logger
        self.best_metric = best_metric
        self.mode = mode
        
        # Initialize best score
        self.best_score = float('inf') if mode == "min" else float('-inf')
        
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_val: float
    ):
        """Saves 'last.pt' and optionally 'best.pt'."""
        
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": metric_val
        }
        
        # Save Last
        last_path = os.path.join(self.save_dir, "last.pt")
        torch.save(state, last_path)
        
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
            # print(f"New best model saved at epoch {epoch} with {self.best_metric}={metric_val:.4f}")

    def load_best_valid(self, model: nn.Module) -> nn.Module:
        """Loads weights from best.pt."""
        path = os.path.join(self.save_dir, "best.pt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Returning current model.")
            return model
            
        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best model from {path} (Epoch {checkpoint['epoch']})")
        return model

    def load_last(self, model: nn.Module) -> nn.Module:
        """Loads weights from last.pt."""
        path = os.path.join(self.save_dir, "last.pt")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Returning current model.")
            return model

        checkpoint = torch.load(path, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded last model from {path} (Epoch {checkpoint['epoch']})")
        return model
