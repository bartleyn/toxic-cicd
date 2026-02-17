from __future__ import annotations

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def vader_compound(texts: list[str]) -> list[float]:
    analyzer = SentimentIntensityAnalyzer()
    return np.array([analyzer.polarity_scores(text)["compound"] for text in texts], dtype=np.float32)
