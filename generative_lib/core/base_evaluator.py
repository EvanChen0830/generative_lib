from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from ..utils.logger import Logger

class BaseEvaluator(ABC):
    """Abstract base class for Evaluators."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Args:
            logger: Optional Logger instance to log metrics to MLflow/Console.
        """
        self.logger = logger
        
    @abstractmethod
    def evaluate(self, generated_data: torch.Tensor, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates generated data against real data from dataloader.
        
        Args:
            generated_data: Tensor of generated samples [N, D] or [N, C, H, W].
            dataloader: DataLoader providing real samples.
            
        Returns:
            Dictionary of calculated metrics.
        """
        pass
