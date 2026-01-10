import torch
import torch.nn as nn
from typing import List, Optional, Any, Dict
import copy
from ...core.base_trainer import BaseTrainer
from ...core.base_method import BaseMethod
from tqdm import tqdm

class ConsistencyTrainer(BaseTrainer):
    """Trainer for Consistency Models (CT).
    
    Manages the 'Target Model' (EMA of Online Model) required for consistency loss.
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
        target_decay: float = 0.999,
        **kwargs
    ):
        super().__init__(method, model, optimizer, feature_keys, label_keys, device, tracker, **kwargs)
        self.target_decay = target_decay
        
        # Initialize Target Model (clone of Online Model)
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

    def _update_target_model(self):
        """Updates target model weights using EMA."""
        with torch.no_grad():
            for p_target, p_online in zip(self.target_model.parameters(), self.model.parameters()):
                p_target.data.mul_(self.target_decay).add_(p_online.data, alpha=1 - self.target_decay)

    def _train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        """Custom train epoch to handle EMA update and kwargs."""
        self.model.train()
        total_metrics = {}
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train (CT)", leave=False)
        for batch in pbar:
            if not isinstance(batch, dict):
                 raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)

            self.optimizer.zero_grad()
            
            # Pass target_model to compute_loss
            loss_dict = self.method.compute_loss(self.model, x, cond, target_model=self.target_model)
            
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
            self._update_target_model()
            
            # Accumulate
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})
            
        return {k: v / count for k, v in total_metrics.items()}
