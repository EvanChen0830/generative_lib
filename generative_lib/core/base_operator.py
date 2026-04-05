import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseOperator(nn.Module, ABC):
    """Base class for observation operators in inverse problems: y = A(x) + epsilon"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the forward measurement operator A to x."""
        pass


class MaskingOperator(BaseOperator):
    """An operator that masks a portion of the input tensor.
    Useful for defining strictly known versus unknown regions (e.g. Inpainting).
    """
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        # mask should be broadcasting compatible with x: 1 for known observed, 0 for missing.
        self.register_buffer("mask", mask)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask
