from typing import Dict, Optional, Any
import os
from pathlib import Path

class Logger:
    """Unified Logger wrapper (MLflow)."""

    def __init__(
        self,
        project_name: str,
        run_name: str,
        mlflow_uri: Optional[str] = None,
        log_dir: str = "./logs",
        use_mlflow: bool = True,
        run_id: Optional[str] = None
    ):
        self.use_mlflow = use_mlflow
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = run_id
        
        if self.use_mlflow:
            import mlflow
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
            
            mlflow.set_experiment(project_name)
            
            if run_id:
                # Resume existing run
                mlflow.start_run(run_id=run_id)
            else:
                # Start new run
                run = mlflow.start_run(run_name=run_name)
                self.run_id = run.info.run_id
                
            self.mlflow = mlflow

    def is_active(self) -> bool:
        return self.use_mlflow and hasattr(self, "mlflow")
    
    def resume(self, run_id: str):
        """Resumes an existing MLflow run."""
        if not self.use_mlflow:
            return

        import mlflow
        self.run_id = run_id

        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            return

        if active_run:
            mlflow.end_run()

        mlflow.start_run(run_id=run_id)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs scalar metrics."""
        if self.use_mlflow:
            self.mlflow.log_metrics(metrics, step=step or 0)

    def log_params(self, params: Dict[str, Any]):
        """Logs hyperparameters."""
        if self.use_mlflow:
            self.mlflow.log_params(params)

    def set_tags(self, tags: Dict[str, Any]):
        """Logs run tags."""
        if self.is_active():
            self.mlflow.set_tags(tags)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs a single file artifact."""
        if self.is_active():
            self.mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Logs all artifacts from a directory."""
        if self.is_active():
            self.mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def log_text(self, text: str, artifact_file: str):
        """Logs a text artifact."""
        if self.is_active():
            self.mlflow.log_text(text, artifact_file)

    def log_figure(self, figure: Any, artifact_file: str):
        """Logs a matplotlib figure artifact."""
        if self.is_active():
            self.mlflow.log_figure(figure, artifact_file)

    def finish(self):
        """Ends the run."""
        if self.is_active():
            self.mlflow.end_run()
