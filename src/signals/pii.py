import os

import numpy as np
from presidio_analyzer import AnalyzerEngine

from src.signals.base import BaseSignal

# keep it around 0.85 so the Presidio NER entities can pick up more types of entities
DEFAULT_SCORE_THRESHOLD = 0.85


class PIIModel(BaseSignal):
    name = "pii"

    def __init__(self, score_threshold: float | None = None):
        self._analyzer = AnalyzerEngine()
        if score_threshold is None:
            score_threshold = float(os.getenv("PII_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD))
        self.score_threshold = score_threshold

    def analyze(self, texts: list[str]) -> tuple[np.ndarray, list[list[str]]]:
        entities = [
            [r.entity_type for r in self._analyzer.analyze(text=text, language="en") if r.score >= self.score_threshold]
            for text in texts
        ]
        scores = np.array([1.0 if found else 0.0 for found in entities])
        return scores, entities

    def score(self, texts: list[str]) -> np.ndarray:
        return self.analyze(texts)[0]

    def entities(self, texts: list[str]) -> list[list[str]]:
        return self.analyze(texts)[1]
