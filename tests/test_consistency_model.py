"""Tests for unconditional and conditional consistency models."""

import torch
import torch.nn as nn

from generative_lib.consistency_model.method.consistency_model import ConsistencyModel
from generative_lib.consistency_model.sampler.base import BaseConsistencyModelSampler
from generative_lib.consistency_model.trainer.base import BaseConsistencyModelTrainer
from generative_lib.utils.tracker import ModelTracker


class RecordingModel(nn.Module):
    """Small consistency network that records its conditioning inputs."""

    def __init__(self) -> None:
        """Initializes the recording model."""
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.conditions: list[torch.Tensor | None] = []

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Records the condition and returns a simple denoiser output.

        Args:
            x: Noisy inputs.
            t: Noise levels.
            condition: Optional class condition.

        Returns:
            Tensor with the same shape as ``x``.
        """
        del t
        self.conditions.append(None if condition is None else condition.detach().clone())
        return self.scale * x


class BatchNormModel(nn.Module):
    """Small model used to verify target normalization behavior."""

    def __init__(self) -> None:
        """Initializes the batch-normalized model."""
        super().__init__()
        self.norm = nn.BatchNorm1d(2)
        self.output = nn.Linear(2, 2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies normalization and a learned projection."""
        del t, condition
        return self.output(self.norm(x))


def test_consistency_boundary_is_identity() -> None:
    """The consistency parameterization is exactly identity at sigma_min."""
    method = ConsistencyModel(sigma_min=0.05)
    model = RecordingModel()
    x = torch.randn(4, 2)

    prediction = method.predict(model, x, method.sigma_min)

    torch.testing.assert_close(prediction, x)


def test_conditional_training_never_creates_an_unconditional_branch() -> None:
    """Conditional CT uses the supplied condition for online and target models."""
    method = ConsistencyModel(num_scales=4, min_scales=4)
    online_model = RecordingModel()
    target_model = RecordingModel()
    condition = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    loss_dict = method.compute_loss(
        online_model,
        torch.randn(4, 2),
        condition,
        teacher_model=target_model,
    )

    assert set(loss_dict) == {"loss", "consistency_loss"}
    torch.testing.assert_close(online_model.conditions[0], condition)
    torch.testing.assert_close(target_model.conditions[0], condition)


def test_explicit_conditional_and_unconditional_losses_are_isolated() -> None:
    """The explicit objectives never cross-contaminate their conditioning mode."""
    method = ConsistencyModel(num_scales=4, min_scales=4)
    conditional_model = RecordingModel()
    unconditional_model = RecordingModel()
    condition = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    batch = torch.randn(4, 2)

    method.compute_conditional_loss(conditional_model, batch, condition)
    method.compute_unconditional_loss(unconditional_model, batch)

    torch.testing.assert_close(conditional_model.conditions[0], condition)
    assert unconditional_model.conditions[0] is None


def test_unconditional_sampling_does_not_supply_a_condition() -> None:
    """Unconditional sampling keeps every model call unconditional."""
    method = ConsistencyModel(num_scales=4, min_scales=4)
    model = RecordingModel()
    sampler = BaseConsistencyModelSampler(method, model, device="cpu", steps=4)

    samples = sampler.sample_unconditional(num_samples=6, shape=[2])

    assert samples.shape == (6, 2)
    assert torch.isfinite(samples).all()
    assert all(condition is None for condition in model.conditions)


def test_conditional_sampling_only_uses_supplied_condition() -> None:
    """Conditional sampling exposes a dedicated condition-preserving API."""
    method = ConsistencyModel(num_scales=4, min_scales=4)
    model = RecordingModel()
    sampler = BaseConsistencyModelSampler(method, model, device="cpu", steps=2)
    condition = torch.tensor([[1.0]])

    samples = sampler.sample_conditional(num_samples=3, shape=[2], condition=condition)

    assert samples.shape == (1, 3, 2)
    assert all(torch.equal(observed, torch.ones(3, 1)) for observed in model.conditions)


def test_consistency_sampling_uses_the_target_teacher() -> None:
    """The sampling model is the EMA target teacher, not the auxiliary EMA."""
    model = RecordingModel()
    trainer = BaseConsistencyModelTrainer(
        method=ConsistencyModel(),
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
    )

    assert trainer.get_sampling_model() is trainer.target_model


def test_consistency_resume_restores_target_and_scheduler_state(tmp_path) -> None:
    """A resumed CT run restores its teacher and learning-rate schedule."""
    batch = [{"position": torch.randn(4, 2)}]
    model = RecordingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    tracker = ModelTracker("test", "consistency", str(tmp_path))
    trainer = BaseConsistencyModelTrainer(
        method=ConsistencyModel(num_scales=4, min_scales=4),
        model=model,
        optimizer=optimizer,
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
        tracker=tracker,
        scheduler=scheduler,
    )
    trainer.fit(batch, epochs=1)
    expected_scheduler_state = scheduler.state_dict()
    expected_target_state = {key: value.clone() for key, value in trainer.target_model.state_dict().items()}

    resumed_model = RecordingModel()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.5)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=4)
    resumed_trainer = BaseConsistencyModelTrainer(
        method=ConsistencyModel(num_scales=4, min_scales=4),
        model=resumed_model,
        optimizer=resumed_optimizer,
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
        tracker=ModelTracker("test", "consistency", str(tmp_path)),
        scheduler=resumed_scheduler,
    )
    resumed_trainer.fit(batch, epochs=1, resume=True)

    assert resumed_scheduler.state_dict() == expected_scheduler_state
    for key, value in expected_target_state.items():
        torch.testing.assert_close(resumed_trainer.target_model.state_dict()[key], value)


def test_target_teacher_uses_its_own_batchnorm_statistics(tmp_path) -> None:
    """The EMA target stays in train mode during CT, matching the reference loop."""
    model = BatchNormModel()
    trainer = BaseConsistencyModelTrainer(
        method=ConsistencyModel(num_scales=4, min_scales=4),
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
        tracker=ModelTracker("test", "consistency", str(tmp_path)),
    )

    trainer.fit([{"position": torch.randn(4, 2)}], epochs=1)

    assert trainer.target_model.training
    assert trainer.target_model.norm.num_batches_tracked.item() == 1


def test_fixed_ct_configuration_keeps_the_paper_warm_start_settings() -> None:
    """Warm-start CT can disable schedules and keep a fixed target EMA."""
    method = ConsistencyModel(
        num_scales=32,
        min_scales=2,
        target_ema_start=0.99,
        use_scale_schedule=False,
        use_ema_schedule=False,
    )
    method.set_training_progress(train_step=999, total_train_steps=1000)

    assert method.current_num_scales() == 32
    assert method.compute_target_ema_rate() == 0.99
