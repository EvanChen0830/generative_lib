from ...core.base_trainer import BaseTrainer

class BaseDiffusionTrainer(BaseTrainer):
    """Trainer specifically for Diffusion models.
    
    Currently identical to BaseTrainer, but allows for future extensions
    (e.g., logging generated images during training).
    """
    pass
