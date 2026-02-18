from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import compute_binary_metrics, compute_overall_metrics, write_metrics


class TestComputeBinaryMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_binary_metrics(y_true, y_proba, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2])
        result = compute_binary_metrics(y_true, y_proba, threshold=0.5)
        assert result["tp"] == 0
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 2

    def test_threshold_affects_labels(self):
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.6, 0.7, 0.4, 0.3])
        low = compute_binary_metrics(y_true, y_proba, threshold=0.3)
        high = compute_binary_metrics(y_true, y_proba, threshold=0.8)
        assert low["recall"] >= high["recall"]

    def test_returns_expected_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7])
        result = compute_binary_metrics(y_true, y_proba)
        expected_keys = {
            "auc", "precision", "recall", "f1_score",
            "specificity", "sensitivity",
            "tp", "tn", "fp", "fn", "false_positive_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_false_positive_rate(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.6, 0.2, 0.8, 0.9])
        result = compute_binary_metrics(y_true, y_proba, threshold=0.5)
        assert result["false_positive_rate"] == pytest.approx(0.5)


class TestComputeOverallMetrics:
    def test_returns_auc_and_avg_precision(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.2, 0.8])
        result = compute_overall_metrics(y_true, y_score)
        assert "overall_auc" in result
        assert "overall_average_precision" in result
        assert result["overall_auc"] == 1.0

    def test_single_class_returns_nan_auc(self):
        y_true = np.array([1, 1, 1, 1])
        y_score = np.array([0.5, 0.6, 0.7, 0.8])
        result = compute_overall_metrics(y_true, y_score)
        assert np.isnan(result["overall_auc"])


class TestWriteMetrics:
    def test_writes_json_file(self, tmp_path):
        metrics = {"auc": 0.95, "precision": 0.9}
        out_path = str(tmp_path / "subdir" / "metrics.json")
        write_metrics(metrics, out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == metrics
