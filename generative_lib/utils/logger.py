"""Experiment logging through Weights & Biases."""

from pathlib import Path
from typing import Any


class Logger:
    """Unified Weights & Biases experiment logger.

    Args:
        project_name: Weights & Biases project that owns the run.
        run_name: Human-readable run name.
        log_dir: Directory used by Weights & Biases for local run files.
        use_wandb: Whether to initialize and use Weights & Biases.
        run_id: Existing run ID to resume. When omitted, a new run is created.
        wandb_mode: Optional Weights & Biases mode, such as ``"offline"``.
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_dir: str = "./logs",
        use_wandb: bool = True,
        run_id: str | None = None,
        wandb_mode: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.run_name = run_name
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.run_id = run_id
        self.wandb_mode = wandb_mode
        self.run = None

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if use_wandb:
            import wandb

            self.wandb = wandb
            self._start_run(run_id)

    def _start_run(self, run_id: str | None) -> None:
        """Starts a new run or reconnects to an existing run.

        Args:
            run_id: Existing run ID to resume, or ``None`` for a new run.
        """
        init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "dir": self.log_dir,
            "id": run_id,
            "resume": "allow" if run_id else None,
        }
        if self.wandb_mode is not None:
            init_kwargs["mode"] = self.wandb_mode

        self.run = self.wandb.init(**init_kwargs)
        self.run_id = self.run.id

    def is_active(self) -> bool:
        """Returns whether a Weights & Biases run is active.

        Returns:
            ``True`` when logging is enabled.
        """
        return self.use_wandb and self.run is not None

    def resume(self, run_id: str) -> None:
        """Reconnects the logger to a checkpoint's Weights & Biases run.

        Args:
            run_id: Weights & Biases run ID stored in a checkpoint.
        """
        if not self.use_wandb:
            return
        if self.run_id == run_id:
            return

        self.finish()
        self._start_run(run_id)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs scalar metrics at an optional training step.

        Args:
            metrics: Metric names mapped to scalar values.
            step: Training step associated with the metrics.
        """
        if self.is_active():
            self.run.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Stores run configuration values.

        Args:
            params: Parameter names mapped to serializable values.
        """
        if self.is_active():
            self.run.config.update(params, allow_val_change=True)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Adds tags to the active run.

        Args:
            tags: Tag names mapped to values.
        """
        if self.is_active():
            self.run.tags = tuple(set(self.run.tags).union(tags.keys()))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Uploads one local file as a Weights & Biases artifact.

        Args:
            local_path: File to upload.
            artifact_path: Optional path inside the artifact.
        """
        if self.is_active():
            artifact = self.wandb.Artifact(Path(local_path).stem, type="artifact")
            artifact.add_file(local_path, name=artifact_path)
            self.run.log_artifact(artifact)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        """Uploads a local directory as a Weights & Biases artifact.

        Args:
            local_dir: Directory to upload.
            artifact_path: Optional artifact name.
        """
        if self.is_active():
            artifact = self.wandb.Artifact(artifact_path or Path(local_dir).name, type="artifact")
            artifact.add_dir(local_dir)
            self.run.log_artifact(artifact)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Logs text content to the active run.

        Args:
            text: Text to record.
            artifact_file: Metric key used for the text content.
        """
        if self.is_active():
            self.run.log({artifact_file: self.wandb.Html(text)})

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Logs a matplotlib figure to the active run.

        Args:
            figure: Figure object to upload.
            artifact_file: Metric key used for the image.
        """
        if self.is_active():
            self.run.log({artifact_file: self.wandb.Image(figure)})

    def finish(self) -> None:
        """Finishes the active Weights & Biases run."""
        if self.is_active():
            self.run.finish()
            self.run = None
