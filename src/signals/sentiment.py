from __future__ import annotations

import numpy as np
from transformers import pipeline

from src.signals.base import BaseSignal


class SentimentModel(BaseSignal):
    """Drop-in sentiment signal backed by VADER.

    Implements the same ``score(texts)`` interface for easier swapping later
    """

    name = "sentiment"

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self._pipe = pipeline(
            "sentiment-analysis", revision="714eb0f", model=model_name, truncation=True, max_length=512
        )

    def score(self, texts: list[str]) -> np.ndarray:
        results = self._pipe(texts)
        scores = []
        for r in results:
            score = r["score"] if r["label"] == "POSITIVE" else -r["score"]
            scores.append(score)
        return np.array(scores, dtype=np.float32)
