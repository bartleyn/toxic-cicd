from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.signals.pii import PIIModel


@pytest.fixture(scope="module")
def model():
    return PIIModel()


def _fake_model(recognizer_results, threshold=None):
    """PIIModel with a stubbed analyzer, so no spaCy model is needed."""
    with patch("src.signals.pii.AnalyzerEngine") as engine_cls:
        engine_cls.return_value.analyze.return_value = recognizer_results
        return PIIModel(score_threshold=threshold)


def test_score_shape(model):
    scores = model.score(["I love this", "I hate this"])
    assert scores.shape == (2,)


def test_pii_scores_positive(model):
    scores = model.score(["My email is test@example.com"])
    assert scores[0] > 0


def test_no_pii_scores_negative(model):
    scores = model.score(["Nothing private here"])
    assert scores[0] <= 0


def test_entities_returns_types(model):
    res = model.entities(["My email is ntbartley@gmail.com"])
    assert "EMAIL_ADDRESS" in res[0]


def test_entities_clean_text_returns_empty(model):
    res = model.entities(["Nothing private here"])
    assert res[0] == []


def test_analyzer_runs_once_per_text():
    """analyze() must not sweep the same text twice to produce scores and entities."""
    pii = _fake_model([SimpleNamespace(entity_type="EMAIL_ADDRESS", score=1.0)])
    pii._analyzer.analyze.reset_mock()

    scores, entities = pii.analyze(["a@b.com", "c@d.com", "nothing"])

    assert pii._analyzer.analyze.call_count == 3
    assert list(scores) == [1.0, 1.0, 1.0]
    assert entities[0] == ["EMAIL_ADDRESS"]


def test_default_threshold_keeps_spacy_entities():
    """spaCy NER entities score 0.85 and must survive the default threshold."""
    pii = _fake_model(
        [
            SimpleNamespace(entity_type="PERSON", score=0.85),
            SimpleNamespace(entity_type="LOCATION", score=0.85),
            SimpleNamespace(entity_type="PHONE_NUMBER", score=0.40),
        ]
    )
    scores, entities = pii.analyze(["my name is john smith and i live in seattle"])

    assert entities[0] == ["PERSON", "LOCATION"]
    assert scores[0] == 1.0


def test_threshold_from_env(monkeypatch):
    monkeypatch.setenv("PII_SCORE_THRESHOLD", "0.4")
    with patch("src.signals.pii.AnalyzerEngine", MagicMock()):
        assert PIIModel().score_threshold == 0.4


def test_score_and_entities_agree():
    pii = _fake_model([SimpleNamespace(entity_type="PERSON", score=0.85)])
    texts = ["john smith"]
    assert (pii.score(texts)[0] == 1.0) == bool(pii.entities(texts)[0])
