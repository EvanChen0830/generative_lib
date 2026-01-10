from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

class BaseMethod(ABC, nn.Module):
    """Abstract base class for all generative methods (Diffusion, Flow Matching, etc.).

    This class defines the interface for the physics/math of the generative process.
    It handles:
    1. Time sampling (for training)
    2. Signal-to-Noise Ratio (SNR) calculations (if applicable)
    3. The `predict` wrapper to unify model calls.
    4. The abstract `compute_loss` method.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_loss(self, model: nn.Module, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes the training loss.

        Args:
            model: The neural network model (must support `forward(x, t, condition)`).
            x: The target data [Batch, D].
            condition: Optional conditioning information [Batch, C].

        Returns:
            A dictionary containing the scalar loss key "loss" and other metrics.
        """
        pass

    def predict(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Wrapper for model forward pass.

        Ensures consistent input handling (e.g., ensuring `t` is the right shape).

        Args:
            model: The neural network.
            x_t: Noisy input at time t.
            t: Time tensor (scalar or batch).
            condition: Conditioning information.

        Returns:
            The model output (e.g., noise, velocity, or x_0).
        """
        # Ensure t is broadcastable if it's a scalar or 1D
        if isinstance(t, float) or isinstance(t, int):
             t = torch.tensor([t] * x_t.shape[0], device=x_t.device)
        
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x_t.shape[0])
            
        return model(x_t, t, condition)

    def sample_times(self, batch_size: int, device: torch.device, mode: str = "uniform") -> torch.Tensor:
        """Samples training time steps.

        Args:
            batch_size: Number of time steps to sample.
            device: Device to place the tensor on.
            mode: Sampling mode ("uniform", "log_normal", etc.). Defaults to "uniform".

        Returns:
            Tensor of times [Batch].
        """
        if mode == "uniform":
            # Sample t ~ U(0, 1) usually, but concrete classes might scale this (e.g. 0 to 1000)
            # Keeping it abstract 0-1 for continuous time methods
            return torch.rand(batch_size, device=device)
        else:
             raise NotImplementedError(f"Time sampling mode {mode} not implemented yet.")
    
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Calculates Signal-to-Noise Ratio for a given time t.
        
        Optional: Not all methods explicitly use SNR, but it's common in Diffusion/Flow.
        
        Args:
            t: Time tensor.
            
        Returns:
            SNR tensor.
        """
        raise NotImplementedError("get_snr not implemented for this method.")
