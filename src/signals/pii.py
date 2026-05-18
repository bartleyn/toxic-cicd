import numpy as np
from presidio_analyzer import AnalyzerEngine

from src.signals.base import BaseSignal


class PIIModel(BaseSignal):
    name = "pii"

    def __init__(self):
        self._analyzer = AnalyzerEngine()

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([1.0 if self.entities([t])[0] else 0.0 for t in texts])

    def entities(self, texts: list[str]) -> list[list[str]]:
        result = []
        for text in texts:
            found = self._analyzer.analyze(text=text, language="en")
            result.append([r.entity_type for r in found if r.score >= 0.95])
        return result
