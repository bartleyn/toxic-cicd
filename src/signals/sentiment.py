
from __future__ import annotations

from typing import List

import numpy as np

from .vader_labels import vader_compound


class SentimentModel:
    """Drop-in sentiment signal backed by VADER.

    Implements the same ``score(texts)`` interface for easier swapping later
    """

    def score(self, texts: List[str]) -> np.ndarray:
        return vader_compound(texts)
