
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import joblib 
import numpy as np


from .features import validate_texts, normalize_texts
from .model import ToxicityModel
from .signals.sentiment import SentimentModel


class Predictor:

    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

        tox_dir = os.path.join(artifact_dir, 'toxicity')
        self.feature_extractor = joblib.load(os.path.join(tox_dir, 'vectorizer.joblib'))
        self.model = ToxicityModel.load(artifact_dir=tox_dir)

        self.default_threshold = self.model.metadata.decision_threshold if self.model.metadata else 0.5
        self.model_version = self.model.metadata.model_version if self.model.metadata else 'unknown'

        self.sentiment_model = SentimentModel()

    def score_texts(self, texts: List[str]) -> np.ndarray:
        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        X = self.feature_extractor.transform(normalized_texts)
        return self.model.score(X)

    def predict_labels(self, texts: List[str], threshold: float) -> np.ndarray:
        proba = self.score_texts(texts)
        return (proba >= threshold).astype(int)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> Dict[str, Any]:
        threshold = threshold if threshold is not None else self.default_threshold
        scores = self.score_texts(texts)
        labels = (scores >= threshold).astype(int)
        sentiment_scores = self.sentiment_model.score(texts)

        return {
            'model_version': self.model_version,
            'threshold': threshold,
            'results': [
                {
                    'toxicity_score': float(s),
                    'label': int(l),
                    'sentiment_score': float(ss),
                }
                for s, l, ss in zip(scores, labels, sentiment_scores)
            ],
        }

    def info(self) -> Dict[str, Any]:
        return {
            'artifact_dir': self.artifact_dir,
            'model_version': self.model_version,
            'default_threshold': self.default_threshold,
        }