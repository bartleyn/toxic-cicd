from __future__ import annotations

import numpy as np
import pytest

from src.signals.toxicity import ModelMetadata, ModelSpec, ToxicityModel


def _fit_tiny_model():
    model = ToxicityModel(
        spec=ModelSpec(),
        metadata=ModelMetadata(model_version="test"),
    )
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([1, 0, 1, 0])
    model.fit(X, y)
    return model


def test_fit_sets_is_fitted():
    model = _fit_tiny_model()
    assert model.is_fitted is True


def test_predict_proba_shape():
    model = _fit_tiny_model()
    proba = model.predict_proba(np.array([[1, 0], [0, 1]]))
    assert proba.shape == (2, 2)


def test_score_returns_positive_class_proba():
    model = _fit_tiny_model()
    scores = model.score(np.array([[1, 0]]))
    assert scores.shape == (1,)
    assert 0.0 <= scores[0] <= 1.0


def test_predict_labels_uses_threshold():
    model = _fit_tiny_model()
    labels = model.predict_labels(np.array([[1, 0], [0, 1]]), threshold=0.5)
    assert set(labels).issubset({0, 1})


def test_predict_before_fit_raises():
    model = ToxicityModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.array([[1, 0]]))


def test_save_and_load_roundtrip(tmp_path):
    model = _fit_tiny_model()
    model.save(str(tmp_path))
    loaded = ToxicityModel.load(str(tmp_path))
    assert loaded.is_fitted is True
    assert loaded.metadata.model_version == "test"
    X = np.array([[1, 0]])
    np.testing.assert_array_almost_equal(model.score(X), loaded.score(X))
