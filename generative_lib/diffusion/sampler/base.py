import torch
import numpy as np
from tqdm import tqdm
from typing import Union, List, Optional
from ...core.base_sampler import BaseSampler
from ...core.base_method import BaseMethod

class BaseDiffusionSampler(BaseSampler):
    """Sampler for Gaussian Diffusion models (DDPM & DDIM) with CFG support."""

    def __init__(
        self, 
        method: BaseMethod, 
        model: torch.nn.Module, 
        device: str, 
        steps: int = 50, 
        feature_keys: Optional[List[str]] = None,
        sampler_type: str = "ddpm", # 'ddpm' or 'ddim'
        guidance_scale: float = 1.0,
        unconditional_value: float = 0.0
    ):
        super().__init__(method, model, device, feature_keys=feature_keys)
        self.steps = steps
        self.sampler_type = sampler_type
        self.guidance_scale = guidance_scale
        self.unconditional_value = unconditional_value
        
    def sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Samples for a single/tensor condition."""
        if condition is not None:
            condition = condition.to(self.device)
            # Normalize shape: [B, C]
            if condition.ndim == 1:
                condition = condition.unsqueeze(0)
            
            # Replicate condition: [B, C] -> [B, Num_Samples, C] -> [B*Num_Samples, C]
            B = condition.shape[0]
            cond_expanded = condition.repeat_interleave(num_samples, dim=0)
            target_bs = B * num_samples
        else:
            # Unconditional: Just generate num_samples
            target_bs = num_samples
            cond_expanded = None
            B = 1 
        
        flat_samples = self._sample_batch(target_bs, shape, cond_expanded)
        
        # Reshape to [B, Num_Samples, *Shape] or [Num_Samples, *Shape]
        if B > 1 or (condition is not None):
             final_shape = (B, num_samples, *shape)
             return flat_samples.view(final_shape)
        else:
             return flat_samples # [Num, *Shape]

    def batch_sample(
        self, 
        num_samples: int, 
        shape: Union[torch.Size, List[int]], 
        dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Samples from dataloader."""
        all_samples = []
        print(f"Sampling ({self.sampler_type}, w={self.guidance_scale}) from loader with {self.steps} steps...")
        for batch in tqdm(dataloader, desc="Dataloader Sampling"):
            cond = self._extract_condition(batch)
            if cond is None:
                 if isinstance(batch, dict):
                     current_bs = len(next(iter(batch.values())))
                 else:
                     current_bs = len(batch)
            else:
                current_bs = cond.shape[0]

            if cond is not None:
                cond_expanded = cond.repeat_interleave(num_samples, dim=0) 
            else:
                cond_expanded = None
            
            total_items = current_bs * num_samples
            with torch.no_grad():
                samples_flat = self._sample_batch(total_items, shape, cond_expanded)
            
            final_shape = (current_bs, num_samples, *shape)
            samples_batch = samples_flat.view(final_shape)
            all_samples.append(samples_batch.cpu())
            
        return torch.cat(all_samples, dim=0)

    def _sample_batch(self, current_batch_size: int, shape: Union[torch.Size, List[int]], condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Internal helper handling DDPM/DDIM and CFG."""
        batch_shape = (current_batch_size, *shape)
        x_t = torch.randn(batch_shape, device=self.device)
        
        time_seq = list(reversed(range(0, self.method.timesteps, self.method.timesteps // self.steps)))
        time_seq = time_seq[:self.steps]
        
        # Nested TQDM for time steps
        for i, t_idx in enumerate(tqdm(time_seq, desc="Diffusion Steps", leave=False)):
            prev_t_idx = t_idx - (self.method.timesteps // self.steps)
            if prev_t_idx < 0: prev_t_idx = -1 
            
            def get_alpha_bar(idx):
                if idx < 0: return torch.tensor(1.0).to(self.device).view(1, *([1]*len(shape)))
                # Reshape alpha to match batch rank for broadcasting
                return self.method.alphas_cumprod[idx].to(self.device).view(1, *([1]*len(shape)))

            alpha_bar_t = get_alpha_bar(t_idx)
            alpha_bar_prev = get_alpha_bar(prev_t_idx)
            
            t_float = t_idx / self.method.timesteps
            
            # Predict Noise
            if self.guidance_scale != 1.0 and condition is not None:
                # CFG: Run Conditioned and Unconditioned
                # We do this by concatenating batch to avoid 2 forward passes overhead if possible
                # But here we call method.predict which might expect specific batch sizes.
                # Let's do concatenation.
                
                x_in = torch.cat([x_t, x_t], dim=0)
                # t_float is scalar, method.predict handles broadcasting
                
                # Create Unconditional Condition
                uncond = torch.full_like(condition, self.unconditional_value)
                c_in = torch.cat([condition, uncond], dim=0)
                
                # Predict
                noise_pred_all = self.method.predict(self.model, x_in, t_float, c_in)
                eps_cond, eps_uncond = torch.chunk(noise_pred_all, 2, dim=0)
                
                pred_noise = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
                
            else:
                # Standard
                pred_noise = self.method.predict(self.model, x_t, t_float, condition)

            # Update Step
            if self.sampler_type == "ddim":
                # DDIM Step (Deterministic, sigma=0)
                # x_{t-1} = sqrt(alpha_prev) * ( (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t) ) + sqrt(1-alpha_prev)*eps
                
                # "Predicted x0"
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                
                # Direction pointing to xt
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
                
                # x_{t-1}
                x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                
                x_t = x_prev
                
            elif self.sampler_type == "ddpm":
                # DDPM Step
                # Recalculate x0 (same formula)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
                x_mean = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
                
                # Add noise (if not last step)
                if prev_t_idx >= 0:
                    # Calculate posterior variance (beta_t for simple case, or posterior)
                    # Simple approx: beta_t * (1-alpha_bar_prev)/(1-alpha_bar_t)
                    # We can use simple beta_t from method if available, strictly we need to compute it.
                    # Given simple linear schedule, let's use sigma = sqrt(1 - alpha_bar / alpha_bar_prev) corresponds to beta_t approx
                    # Actually standard DDPM sampler usually defined in method or here.
                    # Let's reuse the logic derived:
                    # x_{t-1} = x_mean + sigma * z
                    # sigma can be derived. 
                    
                    # For consistency with previous implementation:
                    # Previous was: x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt which is effectively DDIM with sigma=0!
                    # Wait, the PREVIOUS implementation in this repo WAS actually deterministic (DDIM-like style) but labeled "Diffusion Steps".
                    # Real DDPM adds noise.
                    
                    noise = torch.randn_like(x_t)
                    # Sigma T
                    # beta_t = 1 - alpha_t / alpha_prev
                    # We only have alpha_cumprod. alpha_t = alpha_bar_t / alpha_bar_prev (?)
                    # This is getting complex for strided sampling.
                    # Standard practice for strided DDPM is just DDIM with eta=1.0.
                    # Let's implement Generalized DDIM where eta=1.0 -> DDPM, eta=0.0 -> DDIM.
                    
                    # eta = 1.0 (DDPM)
                    eta = 1.0
                    sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                    
                    # Re-calculate direction with sigma
                    dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
                    
                    x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
                    x_t = x_prev
                else:
                    x_t = x_mean

            else:
                 raise ValueError(f"Unknown sampler type: {self.sampler_type}")

        return x_t
