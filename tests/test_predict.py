from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.predict import Predictor


def _make_predictor(tox_scores=None, sent_scores=None, threshold=0.5):
    """Build a Predictor with mocked internals (no artifact dir needed)."""
    tox_scores = tox_scores if tox_scores is not None else np.array([0.9, 0.1])
    sent_scores = sent_scores if sent_scores is not None else np.array([0.3, -0.4])

    tox_model = MagicMock()
    tox_model.name = "toxicity"
    tox_model.score.return_value = tox_scores
    tox_model.metadata = MagicMock(decision_threshold=threshold, model_version="1.0.0")

    sent_model = MagicMock()
    sent_model.name = "sentiment"
    sent_model.score.return_value = sent_scores

    predictor = object.__new__(Predictor)
    predictor.artifact_dir = "fake/dir"
    predictor.models = [tox_model, sent_model]
    predictor.default_threshold = threshold
    predictor.model_version = "1.0.0"
    return predictor


class TestScoreTexts:
    def test_returns_dict_keyed_by_model_name(self):
        predictor = _make_predictor()
        results = predictor.score_texts(["hello", "world"])
        assert isinstance(results, dict)
        assert set(results.keys()) == {"toxicity", "sentiment"}

    def test_score_values_match_mocks(self):
        tox = np.array([0.8, 0.2])
        sent = np.array([0.5, -0.1])
        predictor = _make_predictor(tox_scores=tox, sent_scores=sent)
        results = predictor.score_texts(["a", "b"])
        np.testing.assert_array_equal(results["toxicity"], tox)
        np.testing.assert_array_equal(results["sentiment"], sent)

    def test_all_models_receive_normalized_texts(self):
        predictor = _make_predictor()
        predictor.score_texts(["Hello", "World"])
        for model in predictor.models:
            call_arg = model.score.call_args[0][0]
            assert call_arg == ["hello", "world"]


class TestPredict:
    def test_predict_calls_score_texts_once(self):
        """predict() should only invoke score_texts a single time."""
        predictor = _make_predictor()
        with patch.object(predictor, "score_texts", wraps=predictor.score_texts) as spy:
            predictor.predict(["hello", "world"])
            assert spy.call_count == 1

    def test_predict_returns_expected_structure(self):
        """predict() returns model_version, threshold, and results list."""
        predictor = _make_predictor()
        result = predictor.predict(["hello", "world"])
        assert result["model_version"] == "1.0.0"
        assert result["threshold"] == 0.5
        assert len(result["results"]) == 2
        item = result["results"][0]
        assert hasattr(item, "label")
        assert hasattr(item, "scores")
        assert "toxicity" in item.scores
        assert "sentiment" in item.scores
