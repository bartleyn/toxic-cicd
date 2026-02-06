
from __future__ import annotations

from typing import List
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_compound(texts: List[str]) -> List[float]:
    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(text)['compound'] for text in texts], dtype=np.float32)