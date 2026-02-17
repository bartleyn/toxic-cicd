
from __future__ import annotations

from typing import List

import numpy as np

from src.signals.vader_labels import vader_compound
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentModel:
    """Drop-in sentiment signal backed by VADER.

    Implements the same ``score(texts)`` interface for easier swapping later
    """

    def __init__(self): 
        self.sentiment_analyzer = SentimentIntensityAnalyzer

    def vader_compound(self, texts: List[str]) -> List[float]:
        return np.array([self.sentiment_analyzer.polarity_scores(text)['compound'] for text in texts], dtype=np.float32)
    
    def score(self, texts: List[str]) -> np.ndarray:
        return vader_compound(texts)
