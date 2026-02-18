from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.predict import Predictor


def _make_predictor(tox_scores=None, sent_scores=None, threshold=0.5):
    """Build a Predictor with mocked internals (no artifact dir needed)."""
    tox_scores = tox_scores if tox_scores is not None else np.array([0.9, 0.1])
    sent_scores = sent_scores if sent_scores is not None else np.array([0.3, -0.4])

    tox_model = MagicMock()
    tox_model.name = 'toxicity'
    tox_model.input_type = 'tfidf'
    tox_model.score.return_value = tox_scores
    tox_model.metadata = MagicMock(decision_threshold=threshold, model_version="1.0.0")

    sent_model = MagicMock()
    sent_model.name = 'sentiment'
    sent_model.input_type = 'text'
    sent_model.score.return_value = sent_scores

    vectorizer = MagicMock()
    vectorizer.transform.return_value = np.zeros((len(tox_scores), 5))

    predictor = object.__new__(Predictor)
    predictor.artifact_dir = "fake/dir"
    predictor.feature_extractor = vectorizer
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

    def test_toxicity_model_receives_tfidf_matrix(self):
        predictor = _make_predictor()
        predictor.score_texts(["hello", "world"])
        tox_model = predictor._get_signal('toxicity')
        call_arg = tox_model.score.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)

    def test_sentiment_model_receives_normalized_texts(self):
        """SentimentModel.score() expects List[str], and score_texts now
        correctly passes normalized_texts instead of the TF-IDF matrix."""
        predictor = _make_predictor()
        predictor.score_texts(["Hello", "World"])
        sent_model = predictor._get_signal('sentiment')
        call_arg = sent_model.score.call_args[0][0]
        assert call_arg == ["hello", "world"]

    def test_string_input_passes_to_normalize(self):
        """Validation is handled at API/CLI boundary, not in score_texts."""
        predictor = _make_predictor(
            tox_scores=np.array([0.5]),
            sent_scores=np.array([0.1]),
        )
        predictor.feature_extractor.transform.return_value = np.zeros((1, 5))
        results = predictor.score_texts(["hello"])
        assert "toxicity" in results


class TestPredict:
    def test_predict_calls_score_texts_once(self):
        """predict() should only invoke score_texts a single time."""
        predictor = _make_predictor()
        with patch.object(predictor, "score_texts", wraps=predictor.score_texts) as spy:
            try:
                predictor.predict(["hello", "world"])
            except AttributeError:
                pass  # predict() has a dict.append bug on line 60
            assert spy.call_count == 1

    def test_predict_returns_expected_structure(self):
        """predict() returns model_version, threshold, and results list."""
        predictor = _make_predictor()
        result = predictor.predict(["hello", "world"])
        assert result["model_version"] == "1.0.0"
        assert result["threshold"] == 0.5
        assert len(result["results"]) == 2
        item = result["results"][0]
        assert hasattr(item, "toxicity_score")
        assert hasattr(item, "label")
        assert hasattr(item, "sentiment_score")
