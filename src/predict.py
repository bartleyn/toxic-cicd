
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
        self.models = {
            'toxicity': (ToxicityModel.load(artifact_dir=tox_dir), 'binary'),
            'sentiment': (SentimentModel(), 'numeric')
        }

        self.default_threshold = self.models['toxicity'][0].metadata.decision_threshold if self.models['toxicity'][0].metadata else 0.5
        self.model_version = self.models['toxicity'][0].metadata.model_version if self.models['toxicity'][0].metadata else 'unknown'


    def score_texts(self, texts: List[str]) -> np.ndarray:
        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        X = self.feature_extractor.transform(normalized_texts)
        return {
            name: model.score(X) if name == 'toxicity' else model.score(normalized_texts)
            for name, (model, type) in self.models.items()
        }

    #def predict_labels(self, texts: List[str], threshold: float) -> np.ndarray:
    #    proba = self.score_texts(texts)
    #    return [(name, label) for name, type, label in [(n, t, p) for n, t, p in [(name, type, score) for name, type, score in [(name, type, score) for name, type, score in [(name, type, score) for name, type, score in [(name, type, score) for name, type, score in proba.items()]] if type == 'binary'] if score >= threshold]]
    
    def predict(self, texts: List[str], threshold: Optional[float] = None) -> Dict[str, Any]:
        threshold = threshold if threshold is not None else self.default_threshold
        scores = self.score_texts(texts)
        toxicity_scores = scores.get('toxicity', np.array([0.0] * len(texts)))
        labels = (toxicity_scores >= threshold).astype(int)

        results = []
        for i in range(len(texts)):
            result = {
                'toxicity_score': float(toxicity_scores[i]),
                'label': int(labels[i]),
            }
            if 'sentiment' in scores:
                result['sentiment_score'] = float(scores['sentiment'][i])
            results.append(result)

        return {
            'model_version': self.model_version,
            'threshold': threshold,
            'results': results
        }
        

    def info(self) -> Dict[str, Any]:
        return {
            'artifact_dir': self.artifact_dir,
            'model_version': self.model_version,
            'default_threshold': self.default_threshold,
        }