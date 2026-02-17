from __future__ import annotations

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.signals.vader_labels import vader_compound


class SentimentModel:
    """Drop-in sentiment signal backed by VADER.

    Implements the same ``score(texts)`` interface for easier swapping later
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer

    def vader_compound(self, texts: list[str]) -> list[float]:
        return np.array([self.sentiment_analyzer.polarity_scores(text)["compound"] for text in texts], dtype=np.float32)

    def score(self, texts: list[str]) -> np.ndarray:
        return vader_compound(texts)
