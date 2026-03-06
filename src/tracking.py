from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class NoOpTracker:
    """Tracker that does nothing — used when MLflow is disabled."""

    @contextmanager
    def start_run(self, run_name: str | None = None):
        yield

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def log_artifacts(self, directory: str) -> None:
        pass


class MlflowTracker:
    """Tracker that logs to MLflow."""

    def __init__(self, experiment_name: str) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str | None = None):
        with self._mlflow.start_run(run_name=run_name):
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        flat = _flatten_dict(metrics, prefix=prefix)
        self._mlflow.log_metrics(flat)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def log_artifacts(self, directory: str) -> None:
        self._mlflow.log_artifacts(directory)


def get_tracker(enabled: bool, experiment_name: str) -> MlflowTracker | NoOpTracker:
    if enabled:
        return MlflowTracker(experiment_name)
    return NoOpTracker()


def _flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, float]:
    """Flatten a nested dict into a single-level dict with dotted keys, keeping only numeric values."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        elif isinstance(v, (int, float)):
            out[key] = v
    return out
