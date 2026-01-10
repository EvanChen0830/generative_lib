import torch
import torch.nn as nn
from typing import List, Optional, Any, Dict
from ...core.base_method import BaseMethod
from .consistency_trainer import ConsistencyTrainer
from tqdm import tqdm

class ConsistencyDistillationTrainer(ConsistencyTrainer):
    """Trainer for Consistency Distillation (CD).
    
    Inherits from ConsistencyTrainer (handles EMA Target), but also optionally handles a Teacher Model.
    """

    def __init__(
        self,
        method: BaseMethod,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher_model: nn.Module,
        feature_keys: List[str],
        label_keys: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tracker: Optional[Any] = None,
        target_decay: float = 0.999,
        **kwargs
    ):
        super().__init__(method, model, optimizer, feature_keys, label_keys, device, tracker, target_decay, **kwargs)
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    def _train_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        """Distillation training epoch."""
        self.model.train()
        total_metrics = {}
        count = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train (CD)", leave=False)
        for batch in pbar:
            if not isinstance(batch, dict):
                 raise ValueError("BaseTrainer expects DataLoader to return dictionaries.")

            x, cond = self._process_batch(batch)

            self.optimizer.zero_grad()
            
            # Pass teacher_model AND target_model to compute_loss
            loss_dict = self.method.compute_loss(
                self.model, 
                x, 
                cond, 
                target_model=self.target_model, 
                teacher_model=self.teacher_model
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
            self._update_target_model()
            
            # Accumulate
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})
            
        return {k: v / count for k, v in total_metrics.items()}
