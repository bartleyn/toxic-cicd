import pytest

from src.signals.pii import PIIModel


@pytest.fixture(scope="module")
def model():
    return PIIModel()


def test_score_shape(model):
    scores = model.score(["I love this", "I hate this"])
    assert scores.shape == (2,)


def test_pii_scores_positive(model):
    scores = model.score(["My number is 123-456-7899"])
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
