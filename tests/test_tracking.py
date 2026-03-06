from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tracking import MlflowTracker, NoOpTracker, _flatten_dict, get_tracker


class TestNoOpTracker:
    def test_start_run_context_manager(self):
        tracker = NoOpTracker()
        with tracker.start_run(run_name="test"):
            pass  # should not raise

    def test_log_params_does_nothing(self):
        tracker = NoOpTracker()
        tracker.log_params({"a": 1, "b": "hello"})

    def test_log_metrics_does_nothing(self):
        tracker = NoOpTracker()
        tracker.log_metrics({"auc": 0.95, "f1": 0.8})

    def test_log_artifact_does_nothing(self):
        tracker = NoOpTracker()
        tracker.log_artifact("/some/path.json")

    def test_log_artifacts_does_nothing(self):
        tracker = NoOpTracker()
        tracker.log_artifacts("/some/directory")


class TestMlflowTracker:
    @patch("src.tracking.mlflow", create=True)
    def test_init_sets_experiment(self, mock_mlflow):
        # Patch the import inside __init__
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("my-experiment")
            mock_mlflow.set_experiment.assert_called_once_with("my-experiment")

    @patch("src.tracking.mlflow", create=True)
    def test_log_params(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            tracker.log_params({"C": "2.0", "max_iter": "1000"})
            mock_mlflow.log_params.assert_called_once_with({"C": "2.0", "max_iter": "1000"})

    @patch("src.tracking.mlflow", create=True)
    def test_log_metrics_flat(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            tracker.log_metrics({"auc": 0.95, "f1": 0.8})
            mock_mlflow.log_metrics.assert_called_once_with({"auc": 0.95, "f1": 0.8})

    @patch("src.tracking.mlflow", create=True)
    def test_log_metrics_with_prefix(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            tracker.log_metrics({"auc": 0.95}, prefix="overall")
            mock_mlflow.log_metrics.assert_called_once_with({"overall.auc": 0.95})

    @patch("src.tracking.mlflow", create=True)
    def test_log_artifact(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            tracker.log_artifact("/path/to/file.json")
            mock_mlflow.log_artifact.assert_called_once_with("/path/to/file.json")

    @patch("src.tracking.mlflow", create=True)
    def test_log_artifacts(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            tracker.log_artifacts("/path/to/dir")
            mock_mlflow.log_artifacts.assert_called_once_with("/path/to/dir")

    @patch("src.tracking.mlflow", create=True)
    def test_start_run(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = MlflowTracker("test")
            with tracker.start_run(run_name="v1.0"):
                mock_mlflow.start_run.assert_called_once_with(run_name="v1.0")


class TestGetTracker:
    def test_disabled_returns_noop(self):
        tracker = get_tracker(enabled=False, experiment_name="test")
        assert isinstance(tracker, NoOpTracker)

    @patch("src.tracking.mlflow", create=True)
    def test_enabled_returns_mlflow_tracker(self, mock_mlflow):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            tracker = get_tracker(enabled=True, experiment_name="test")
            assert isinstance(tracker, MlflowTracker)


class TestFlattenDict:
    def test_flat_dict(self):
        assert _flatten_dict({"a": 1, "b": 2.0}) == {"a": 1, "b": 2.0}

    def test_nested_dict(self):
        result = _flatten_dict({"overall": {"auc": 0.95, "ap": 0.8}})
        assert result == {"overall.auc": 0.95, "overall.ap": 0.8}

    def test_with_prefix(self):
        result = _flatten_dict({"auc": 0.95}, prefix="train")
        assert result == {"train.auc": 0.95}

    def test_skips_non_numeric(self):
        result = _flatten_dict({"auc": 0.95, "name": "test", "count": 100})
        assert result == {"auc": 0.95, "count": 100}

    def test_empty_dict(self):
        assert _flatten_dict({}) == {}

    def test_deeply_nested(self):
        result = _flatten_dict({"a": {"b": {"c": 1}}})
        assert result == {"a.b.c": 1}


class TestTrainCLIArgs:
    def test_mlflow_flag_accepted(self):
        from src.train import arg_parser

        parser = arg_parser()
        args = parser.parse_args([
            "--data-path", "data.csv",
            "--artifact-dir", "artifacts",
            "--model-version", "1.0",
            "--mlflow",
            "--experiment-name", "my-exp",
        ])
        assert args.mlflow is True
        assert args.experiment_name == "my-exp"

    def test_mlflow_flag_defaults(self):
        from src.train import arg_parser

        parser = arg_parser()
        args = parser.parse_args([
            "--data-path", "data.csv",
            "--artifact-dir", "artifacts",
            "--model-version", "1.0",
        ])
        assert args.mlflow is False
        assert args.experiment_name is None


class TestEvaluateCLIArgs:
    def test_mlflow_flag_accepted(self):
        from src.evaluate import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--artifact-dir", "artifacts/1.0/toxicity",
            "--eval-data-path", "eval.csv",
            "--mlflow",
            "--experiment-name", "my-eval-exp",
        ])
        assert args.mlflow is True
        assert args.experiment_name == "my-eval-exp"

    def test_mlflow_flag_defaults(self):
        from src.evaluate import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([
            "--artifact-dir", "artifacts/1.0/toxicity",
            "--eval-data-path", "eval.csv",
        ])
        assert args.mlflow is False
        assert args.experiment_name is None
