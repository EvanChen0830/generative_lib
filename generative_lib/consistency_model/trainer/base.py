import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional, Any, List
from torch.utils.data import DataLoader
from ...core.base_trainer import BaseTrainer
from ...core.base_method import BaseMethod

class BaseConsistencyModelTrainer(BaseTrainer):
    """Trainer specifically for Consistency Models.
    
    Handles:
    - Target Model (EMA) maintenance.
    - Passing target model to loss computation.
    """
    
    def __init__(
        self,
        method: BaseMethod,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        feature_keys: List[str],
        label_keys: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tracker: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        grad_clip: float = 1.0,
        ema_decay: float = 0.999,
        total_steps: int = 100, # Discrete steps for training schedule
    ):
        super().__init__(
            method=method,
            model=model,
            optimizer=optimizer,
            feature_keys=feature_keys,
            label_keys=label_keys,
            device=device,
            tracker=tracker,
            scheduler=scheduler,
            grad_clip=grad_clip,
            use_ema=True # We enable this flag conceptually, but handle it manually here
        )
        self.ema_decay = ema_decay
        self.total_steps = total_steps
        
        # Initialize Target Model
        self.target_model = copy.deepcopy(self.model)
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        self.target_model.to(self.device)

    def _update_ema(self):
        """Updates target model weights."""
        with torch.no_grad():
            for param_online, param_target in zip(self.model.parameters(), self.target_model.parameters()):
                param_target.data.mul_(self.ema_decay).add_(param_online.data, alpha=1 - self.ema_decay)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Runs one epoch of training with EMA updates."""
        self.model.train()
        
        total_metrics = {}
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train (CM)", leave=False)
        for batch in pbar:
            if not isinstance(batch, dict):
                 raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)

            self.optimizer.zero_grad()
            
            # Compute Loss via Method (passing target_model)
            loss_dict = self.method.compute_loss(
                self.model, 
                x, 
                cond, 
                target_model=self.target_model,
                total_steps=self.total_steps
            )
            
            if "loss" in loss_dict:
                final_loss = loss_dict["loss"]
            else:
                final_loss = sum(loss_dict.values())
                loss_dict["loss"] = final_loss
            
            final_loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # EMA Update
            self._update_ema()
            
            # Accumulate
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})
            
        return {k: v / count for k, v in total_metrics.items()}

