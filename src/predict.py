from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np

from src.features import normalize_texts
from src.schemas import ItemResult
from src.signals.sentiment import SentimentModel
from src.signals.toxicity import ToxicityModel
from src.signals.base import BaseSignal


class Predictor:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

        tox_dir = os.path.join(artifact_dir, "toxicity")
        self.feature_extractor = joblib.load(os.path.join(tox_dir, "vectorizer.joblib"))
        self.models: list[BaseSignal] = [
            ToxicityModel.load(artifact_dir=tox_dir),
            SentimentModel()
        ]

        tox = self._get_signal('toxicity')

        self.default_threshold = (
            tox.metadata.decision_threshold if tox.metadata else 0.5
        )
        self.model_version = (
            tox.model_version if tox.metadata else "unknown"
        )
    def _get_signal(self, name: str) -> BaseSignal:
        return next(s for s in self.models if s.name == name)

    def score_texts(self, texts: list[str]) -> np.ndarray:
        normalized_texts = normalize_texts(texts)
        X = self.feature_extractor.transform(normalized_texts)
        inputs = {'tfidf': X, 'text': normalized_texts}
        return {
            model.name: model.score(inputs[model.input_type])
            for model in self.models
        }

    def predict(self, texts: list[str], threshold: float | None = None) -> dict[str, Any]:
        threshold = threshold if threshold is not None else self.default_threshold
        scores = self.score_texts(texts)
        toxicity_scores = scores.get("toxicity", np.array([0.0] * len(texts)))
        labels = (toxicity_scores >= threshold).astype(int)

        results = []
        for i in range(len(texts)):
            results.append(
                ItemResult(
                    toxicity_score=float(toxicity_scores[i]),
                    label=int(labels[i]),
                    sentiment_score=float(scores.get('sentiment', np.zeros(len(texts)))[i])
                )
            )

        return {"model_version": self.model_version, "threshold": threshold, "results": results}

    def info(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "model_version": self.model_version,
            "default_threshold": self.default_threshold,
        }
