from __future__ import annotations

import numpy as np
import pytest

from src.signals.base import BaseSignal
from src.signals.hatespeech import HateSpeechMetadata, HateSpeechModel, HateSpeechSpec


def _fit_tiny_model():
    model = HateSpeechModel(
        spec=HateSpeechSpec(word_min_df=1, char_min_df=1),
        metadata=HateSpeechMetadata(model_version="test"),
    )
    texts = [
        "i hate you all",
        "you are wonderful",
        "die you scum",
        "have a nice day",
        "go back where you came from",
        "thanks for your help",
    ]
    y = np.array([1, 0, 1, 0, 1, 0])
    model.fit(texts, y)
    return model


def test_fit_sets_is_fitted():
    model = _fit_tiny_model()
    assert model.is_fitted is True


def test_score_returns_probabilities():
    model = _fit_tiny_model()
    scores = model.score(["i hate you", "have a nice day"])
    assert scores.shape == (2,)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_score_before_fit_raises():
    model = HateSpeechModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.score(["hello"])


def test_save_and_load_roundtrip(tmp_path):
    model = _fit_tiny_model()
    model.save(str(tmp_path))
    loaded = HateSpeechModel.load(str(tmp_path))
    assert loaded.is_fitted is True
    assert loaded.metadata.model_version == "test"
    texts = ["i hate you"]
    np.testing.assert_array_almost_equal(model.score(texts), loaded.score(texts))


def test_hatespeech_model_is_base_signal():
    model = HateSpeechModel()
    assert isinstance(model, BaseSignal)
    assert model.name == "hatespeech"


def test_spec_controls_vectorizer_params():
    spec = HateSpeechSpec(word_max_features=500, char_max_features=200, word_min_df=1, char_min_df=1)
    model = HateSpeechModel(spec=spec)
    assert model.word_vectorizer.max_features == 500
    assert model.char_vectorizer.max_features == 200
