import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...core.base_method import BaseMethod


class ConsistencyModel(BaseMethod):
    """Consistency Model trained against an exponential-moving-average teacher."""

    def __init__(
        self,
        sigma_min: float = 0.05,
        sigma_max: float = 1.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_scales: int = 120,
        min_scales: int = 2,
        target_ema_start: float = 0.95,
        use_log_noise_conditioning: bool = True,
        use_scale_schedule: bool = True,
        use_ema_schedule: bool = True,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.num_scales = num_scales
        self.min_scales = min_scales
        self.target_ema_start = target_ema_start
        self.use_log_noise_conditioning = use_log_noise_conditioning
        self.use_scale_schedule = use_scale_schedule
        self.use_ema_schedule = use_ema_schedule
        self.current_train_step = 0
        self.total_train_steps = 1

    def set_training_progress(self, train_step: int, total_train_steps: int) -> None:
        self.current_train_step = max(train_step, 0)
        self.total_train_steps = max(total_train_steps, 1)

    def _append_dims(self, x: torch.Tensor, target_dims: int) -> torch.Tensor:
        while x.ndim < target_dims:
            x = x.unsqueeze(-1)
        return x

    def _build_karras_sigmas(self, n: int, device: torch.device) -> torch.Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

    def sample_seq_sigmas(self, n: int, device: torch.device, schedule: str = "karras") -> torch.Tensor:
        if schedule == "karras":
            sigmas = self._build_karras_sigmas(n, device)
        elif schedule == "linear":
            sigmas = torch.linspace(self.sigma_max, self.sigma_min, n, device=device)
        elif schedule == "exponential":
            sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), n, device=device).exp()
        else:
            raise ValueError(f"Unknown consistency schedule: {schedule}")
        return sigmas

    def current_num_scales(self) -> int:
        """Returns the active number of noise levels for CT."""
        if not self.use_scale_schedule:
            return self.num_scales
        progress = self.current_train_step / max(self.total_train_steps, 1)
        max_term = (self.num_scales + 1) ** 2 - self.min_scales**2
        scales = math.ceil(math.sqrt(progress * max_term + self.min_scales**2) - 1)
        return max(scales + 1, 2)

    def compute_target_ema_rate(self) -> float:
        """Returns the target-teacher EMA decay for the current CT step."""
        if not self.use_ema_schedule:
            return self.target_ema_start
        current_scales = self.current_num_scales()
        c = -math.log(self.target_ema_start) * self.min_scales
        return math.exp(-c / max(current_scales, 1e-12))

    def sample_training_sigmas(
        self,
        batch_size: int,
        device: torch.device,
        mode: str = "loglogistic",
    ) -> torch.Tensor:
        if mode == "discrete":
            sigmas = self._build_karras_sigmas(self.current_num_scales(), device)
            idx = torch.randint(0, sigmas.shape[0], (batch_size,), device=device)
            return sigmas[idx]
        if mode != "loglogistic":
            raise ValueError(f"Unknown diffusion sigma sampling mode: {mode}")

        loc = math.log(self.sigma_data)
        scale = 0.5
        min_cdf = torch.tensor((math.log(self.sigma_min) - loc) / scale, device=device).sigmoid()
        max_cdf = torch.tensor((math.log(self.sigma_max) - loc) / scale, device=device).sigmoid()
        u = torch.rand(batch_size, device=device) * (max_cdf - min_cdf) + min_cdf
        return torch.logit(u).mul(scale).add(loc).exp()

    def compute_diffusion_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        sigma_sampling: str = "loglogistic",
    ) -> Dict[str, torch.Tensor]:
        device = x.device
        sigma = self.sample_training_sigmas(x.shape[0], device, mode=sigma_sampling)
        noise = torch.randn_like(x)
        sigma_expanded = self._append_dims(sigma, x.ndim)
        x_noisy = x + noise * sigma_expanded

        c_skip = self.sigma_data**2 / (sigma_expanded**2 + self.sigma_data**2)
        c_out = sigma_expanded * self.sigma_data / torch.sqrt(sigma_expanded**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma_expanded**2 + self.sigma_data**2)

        model_t = sigma
        if self.use_log_noise_conditioning:
            model_t = 0.25 * torch.log(model_t.clamp_min(1e-40))

        model_out = model(c_in * x_noisy, model_t, condition)
        target = (x - c_skip * x_noisy) / c_out
        loss = torch.nn.functional.mse_loss(model_out, target)
        return {"loss": loss, "diffusion_loss": loss}

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        teacher_model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes the appropriate consistency-training objective.

        Args:
            model: Online consistency model.
            x: Clean training batch.
            condition: Optional batch-aligned conditioning tensor.
            teacher_model: EMA target teacher. Defaults to ``model``.

        Returns:
            The scalar consistency loss.
        """
        if condition is None:
            return self.compute_unconditional_loss(model, x, teacher_model)
        return self.compute_conditional_loss(model, x, condition, teacher_model)

    def compute_unconditional_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        teacher_model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes CT loss for an unconditional model.

        Args:
            model: Online consistency model.
            x: Clean training batch.
            teacher_model: EMA target teacher. Defaults to ``model``.

        Returns:
            The scalar consistency loss.
        """
        return self._compute_consistency_loss(model, x, None, teacher_model)

    def compute_conditional_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor,
        teacher_model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Computes CT loss for a conditional model.

        Args:
            model: Online consistency model.
            x: Clean training batch.
            condition: Batch-aligned conditioning tensor.
            teacher_model: EMA target teacher. Defaults to ``model``.

        Returns:
            The scalar consistency loss.
        """
        return self._compute_consistency_loss(model, x, condition, teacher_model)

    def _compute_consistency_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor],
        teacher_model: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Computes CT on adjacent Karras noise levels."""
        batch_size = x.shape[0]
        device = x.device
        teacher = teacher_model if teacher_model is not None else model

        current_scales = self.current_num_scales()
        sigmas = self._build_karras_sigmas(current_scales, device)
        idx = torch.randint(0, current_scales - 1, (batch_size,), device=device)
        t_student = sigmas[idx]
        t_teacher = sigmas[idx + 1]

        noise = torch.randn_like(x)
        x_student = x + noise * self._append_dims(t_student, x.ndim)
        x_teacher = x + noise * self._append_dims(t_teacher, x.ndim)

        pred_student = self.predict(model, x_student, t_student, condition)
        with torch.no_grad():
            pred_teacher = self.predict(teacher, x_teacher, t_teacher, condition)
        consistency_loss = torch.nn.functional.mse_loss(pred_student, pred_teacher)
        return {"loss": consistency_loss, "consistency_loss": consistency_loss}

    def predict(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(t, (float, int)):
            t = torch.full((x_t.shape[0],), float(t), device=x_t.device, dtype=torch.float32)
        elif t.ndim == 0:
            t = t.unsqueeze(0).repeat(x_t.shape[0])
        elif t.numel() == 1 and x_t.shape[0] > 1:
            t = t.repeat(x_t.shape[0])

        t = t.to(x_t.device, dtype=torch.float32)
        sigma = self._append_dims(t, x_t.ndim)
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)

        model_t = t
        if self.use_log_noise_conditioning:
            model_t = 0.25 * torch.log(model_t.clamp_min(1e-40))

        model_out = model(c_in * x_t, model_t, condition)
        return c_skip * x_t + c_out * model_out

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        return (self.sigma_data / t) ** 2
