from __future__ import annotations

import numpy as np
import pytest

from src.signals.base import BaseSignal
from src.signals.toxicity import ModelMetadata, ModelSpec, ToxicityModel


def _fit_tiny_model():
    model = ToxicityModel(
        spec=ModelSpec(min_df=1),
        metadata=ModelMetadata(model_version="test"),
    )
    texts = ["good text", "bad text", "great stuff", "awful stuff"]
    y = np.array([0, 1, 0, 1])
    model.fit(texts, y)
    return model


def test_fit_sets_is_fitted():
    model = _fit_tiny_model()
    assert model.is_fitted is True


def test_score_returns_probabilities():
    model = _fit_tiny_model()
    scores = model.score(["good text", "bad text"])
    assert scores.shape == (2,)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_predict_labels_uses_threshold():
    model = _fit_tiny_model()
    labels = model.predict_labels(["good text", "bad text"], threshold=0.5)
    assert set(labels).issubset({0, 1})


def test_predict_before_fit_raises():
    model = ToxicityModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.score(["hello"])


def test_save_and_load_roundtrip(tmp_path):
    model = _fit_tiny_model()
    model.save(str(tmp_path))
    loaded = ToxicityModel.load(str(tmp_path))
    assert loaded.is_fitted is True
    assert loaded.metadata.model_version == "test"
    texts = ["good text"]
    np.testing.assert_array_almost_equal(model.score(texts), loaded.score(texts))


def test_toxicity_model_is_base_signal():
    model = ToxicityModel()
    assert isinstance(model, BaseSignal)
    assert model.name == "toxicity"
