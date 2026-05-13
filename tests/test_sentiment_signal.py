import numpy as np
import pytest
from src.signals.sentiment import SentimentModel


@pytest.fixture(scope='module')
def model():
    return SentimentModel()

def test_score_shape(model):
    scores = model.score(["I love this", "I hate this"])
    assert scores.shape == (2,)

def test_positive_text_scores_positive(model):
    scores = model.score(["This is wonderful and amazing!"])
    assert scores[0] > 0

def test_negative_text_scores_negative(model):
    scores = model.score(["This is terrible and disgusting"])
    assert scores[0] < 0

def test_score_range(model):
    scores = model.score(["Hello world", "great job", "awful experience"])
    assert np.all(scores >= -1) and np.all(scores <= 1)