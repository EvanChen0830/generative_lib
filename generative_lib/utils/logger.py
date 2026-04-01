from typing import Dict, Optional, Any
import os

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
    
    def resume(self, run_id: str):
        """Resumes an existing MLflow run and optionally cleans up the currently initialized stray run."""
        if not self.use_mlflow:
            return

        import mlflow
        self.run_id = run_id

        # End the currently active stray run.
        stray_run = mlflow.active_run()
        if stray_run:
            mlflow.end_run()
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.delete_run(stray_run.info.run_id)
                print(f"Deleted stray MLflow run: {stray_run.info.run_id}")
            except Exception as e:
                print(f"Failed to delete stray MLflow run {stray_run.info.run_id}: {e}")
            
        # Start the correct run
        mlflow.start_run(run_id=run_id)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs scalar metrics."""
        if self.use_mlflow:
            self.mlflow.log_metrics(metrics, step=step or 0)

    def log_params(self, params: Dict[str, Any]):
        """Logs hyperparameters."""
        if self.use_mlflow:
            self.mlflow.log_params(params)

    def finish(self):
        """Ends the run."""
        if self.use_mlflow:
            self.mlflow.end_run()
