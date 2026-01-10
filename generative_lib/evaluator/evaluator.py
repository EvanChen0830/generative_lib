from typing import Dict, List, Optional, Union
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..core.base_evaluator import BaseEvaluator
from ..metrics.distance import calculate_frechet_distance, compute_statistics
from ..utils.logger import Logger

class Evaluator(BaseEvaluator):
    def __init__(self, feature_key: str, logger: Optional[Logger] = None):
        """
        Concrete implementation of Evaluator.
        
        Args:
            feature_key: Key to extract real data from dataloader batches. 
                         Example: "position" for Two Moons.
            logger: MLflow logger instance.
        """
        super().__init__(logger)
        self.feature_key = feature_key

    def evaluate(self, generated_data: torch.Tensor, dataloader: DataLoader, step: Optional[int] = None) -> Dict[str, float]:
        """Evaluates generated data against real data using Fréchet Distance.
        
        Args:
            generated_data: [N, D] or [N, C, H, W] tensor.
            dataloader: Source of truth.
            step: Optional step for logging.
        """
        # Generated Data to Numpy
        if isinstance(generated_data, torch.Tensor):
            gen_np = generated_data.detach().cpu().numpy()
        else:
            gen_np = np.array(generated_data)

        # Collect Real Data
        real_data_list = []
        for batch in dataloader:
            # Handle different batch types
            if isinstance(batch, dict):
                if self.feature_key in batch:
                    real = batch[self.feature_key]
                else:
                    # Fallback or error? For now warn and pick first? 
                    # Better to strict error.
                    raise KeyError(f"Feature key '{self.feature_key}' not found in batch keys: {batch.keys()}")
            elif isinstance(batch, (list, tuple)):
                # Fallback: assume first element if index 0
                real = batch[0] 
            else:
                real = batch
            
            if isinstance(real, torch.Tensor):
                real = real.detach().cpu().numpy()
            
            real_data_list.append(real)
            
        # Concatenate all batches
        if len(real_data_list) > 0:
            real_np = np.concatenate(real_data_list, axis=0)
        else:
            raise ValueError("DataLoader provided no data.")

        # Flatten if > 2D (e.g. images) to allow simple Gaussian statistics
        if len(gen_np.shape) > 2:
            gen_np = gen_np.reshape(gen_np.shape[0], -1)
        if len(real_np.shape) > 2:
            real_np = real_np.reshape(real_np.shape[0], -1)

        # Compute Statistics
        mu1, sigma1 = compute_statistics(real_np)
        mu2, sigma2 = compute_statistics(gen_np)

        # Compute FID
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        metrics = {"fid": float(fid)}
        
        # Log to MLflow
        if self.logger:
            self.logger.log_metrics(metrics, step=step)
            print(f"Evaluator Metrics Logged: {metrics}")
            
        return metrics
