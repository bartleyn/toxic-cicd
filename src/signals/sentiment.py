from __future__ import annotations

import numpy as np

from src.signals.base import BaseSignal
from src.signals.vader_labels import vader_compound


class SentimentModel(BaseSignal):
    """Drop-in sentiment signal backed by VADER.

    Implements the same ``score(texts)`` interface for easier swapping later
    """
    name = "sentiment"
    input_type = "text"

    def score(self, texts: list[str]) -> np.ndarray:
        return vader_compound(texts)
