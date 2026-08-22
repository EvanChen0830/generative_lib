"""Regression tests for shared training infrastructure."""

import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

from generative_lib.core.base_method import BaseMethod
from generative_lib.core.base_trainer import BaseTrainer
from generative_lib.utils.logger import Logger
from generative_lib.utils.tracker import ModelTracker


class DummyMethod(BaseMethod):
    """Minimal method used to exercise trainer control flow."""

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns a placeholder loss.

        Args:
            model: Unused model.
            x: Unused training data.
            condition: Unused conditioning data.

        Returns:
            Placeholder scalar loss.
        """
        del model, x, condition
        return {"loss": torch.tensor(0.0)}


class TrainableMethod(BaseMethod):
    """Minimal differentiable method used to test EMA checkpointing."""

    def compute_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Computes a simple squared prediction loss."""
        del condition
        return {"loss": model(x).square().mean()}


class RecordingLogger:
    """Records metrics and resume IDs without an external backend."""

    def __init__(self) -> None:
        self.run_id = "original-run"
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.resumed_run_ids: list[str] = []

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Records one epoch's metrics.

        Args:
            metrics: Metrics to record.
            step: Epoch number.
        """
        self.metrics.append((metrics, step))

    def resume(self, run_id: str) -> None:
        """Records the run ID selected for resumption.

        Args:
            run_id: Backend run ID from a checkpoint.
        """
        self.run_id = run_id
        self.resumed_run_ids.append(run_id)


class ControlledTrainer(BaseTrainer):
    """Trainer with deterministic epoch metrics."""

    def _train_epoch(self, loader: object, epoch: int) -> dict[str, float]:
        """Returns a deterministic training loss.

        Args:
            loader: Unused loader.
            epoch: Current epoch.

        Returns:
            Training loss for the epoch.
        """
        del loader
        return {"train_loss": float(epoch)}

    def _validate(self, loader: object, epoch: int) -> dict[str, float]:
        """Returns a deterministic validation loss.

        Args:
            loader: Unused loader.
            epoch: Current epoch.

        Returns:
            Validation loss for the epoch.
        """
        del loader
        return {"val_loss": 1.0 / epoch}


def make_trainer(tmp_path, logger: RecordingLogger, early_stop_fn):
    """Builds a controlled trainer and its checkpoint tracker.

    Args:
        tmp_path: Temporary checkpoint directory.
        logger: Logger used by the tracker.
        early_stop_fn: Callable stopping criterion.

    Returns:
        Configured trainer and tracker.
    """
    model = nn.Linear(1, 1)
    return (
        ControlledTrainer(
            method=DummyMethod(),
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            feature_keys=[],
            label_keys=[],
            device="cpu",
            tracker=ModelTracker("test", "model", str(tmp_path), logger=logger),
            early_stop_fn=early_stop_fn,
        ),
        model,
    )


def test_early_stop_logs_and_checkpoints_current_epoch(tmp_path) -> None:
    """Logs combined losses and saves the stopping epoch before returning."""
    logger = RecordingLogger()
    trainer, model = make_trainer(tmp_path, logger, lambda metrics: metrics["val_loss"] <= 0.5)

    trainer.fit([], val_loader=[None], epochs=5)

    assert [step for _, step in logger.metrics] == [1, 2]
    assert logger.metrics[1][0] == {"train_loss": 2.0, "val_loss": 0.5}
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["run_id"] == "original-run"

    resumed_logger = RecordingLogger()
    resumed_trainer, _ = make_trainer(tmp_path, resumed_logger, lambda metrics: metrics["val_loss"] <= 0.25)
    resumed_trainer.fit([], val_loader=[None], epochs=5, resume=True)

    assert resumed_logger.resumed_run_ids == ["original-run"]
    assert [step for _, step in resumed_logger.metrics] == [3, 4]


def test_logger_resumes_the_checkpoint_run_id(monkeypatch, tmp_path) -> None:
    """Reinitializes W&B with the exact checkpoint run ID."""
    init_calls = []

    class FakeRun:
        """Minimal W&B run implementation."""

        def __init__(self, run_id: str) -> None:
            self.id = run_id
            self.config = SimpleNamespace(update=lambda *args, **kwargs: None)
            self.tags = ()
            self.finished = False

        def log(self, metrics: dict[str, float], step: int | None = None) -> None:
            """Records metrics for the fake run."""
            del metrics, step

        def finish(self) -> None:
            """Marks the run as finished."""
            self.finished = True

    def init(**kwargs):
        """Creates a fake W&B run.

        Args:
            **kwargs: W&B initialization arguments.

        Returns:
            Fake W&B run.
        """
        init_calls.append(kwargs)
        return FakeRun(kwargs["id"] or "new-run")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = Logger("project", "run", log_dir=str(tmp_path))
    logger.resume("checkpoint-run")
    logger.log_metrics({"train_loss": 0.1}, step=3)

    assert init_calls[1]["id"] == "checkpoint-run"
    assert init_calls[1]["resume"] == "allow"
    assert logger.run_id == "checkpoint-run"


def test_optional_ema_is_checkpointed_and_restored(tmp_path) -> None:
    """The generic trainer samples from and resumes its optional EMA model."""
    batch = [{"position": torch.ones(4, 1)}]
    model = nn.Linear(1, 1)
    trainer = BaseTrainer(
        method=TrainableMethod(),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
        tracker=ModelTracker("test", "ema", str(tmp_path)),
        use_ema=True,
        ema_decay=0.5,
    )
    trainer.fit(batch, epochs=1)
    assert trainer.get_sampling_model() is trainer.ema_model
    expected_ema = {key: value.clone() for key, value in trainer.ema_model.state_dict().items()}

    resumed_model = nn.Linear(1, 1)
    resumed_trainer = BaseTrainer(
        method=TrainableMethod(),
        model=resumed_model,
        optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0.1),
        feature_keys=[],
        label_keys=["position"],
        device="cpu",
        tracker=ModelTracker("test", "ema", str(tmp_path)),
        use_ema=True,
        ema_decay=0.5,
    )
    resumed_trainer.fit(batch, epochs=1, resume=True)

    for key, value in expected_ema.items():
        torch.testing.assert_close(resumed_trainer.ema_model.state_dict()[key], value)
