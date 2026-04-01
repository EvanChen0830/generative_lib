import torch
from ...core.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

class BaseConsistencyModelTrainer(BaseTrainer):
    """Trainer specifically for Consistency Models with robust EMA tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_decay = 0.999

    def _update_ema(self):
        with torch.no_grad():
            for p_model, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
                p_ema.copy_(p_ema * self.ema_decay + p_model * (1 - self.ema_decay))
            for b_model, b_ema in zip(self.model.buffers(), self.ema_model.buffers()):
                b_ema.copy_(b_model)

    def _train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        self.ema_model.eval()
        
        total_metrics = {}
        count = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch} Train", leave=False)
        for batch in pbar:
            x, cond = self._process_batch(batch)
            self.optimizer.zero_grad()
            
            # Pass both models to method
            loss_dict = self.method.compute_loss(self.model, x, cond, ema_model=self.ema_model)
            final_loss = loss_dict["loss"]
            final_loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            # Update EMA (including buffers for BatchNorm)
            self._update_ema()
                     
            count += 1
            for k, v in loss_dict.items():
                if k not in total_metrics: total_metrics[k] = 0.0
                total_metrics[k] += v.item()

            pbar.set_postfix({"loss": final_loss.item()})
            
        return {f"train_{k}": v / count for k, v in total_metrics.items()}
