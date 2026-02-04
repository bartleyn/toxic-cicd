
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import joblib 
import numpy as np


from .features import validate_texts, normalize_texts
from .model import ToxicityModel


class Predictor:


    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir

        fe_path = os.path.join(artifact_dir, 'vectorizer.joblib')

        self.feature_extractor = joblib.load(fe_path)
        self.model = ToxicityModel.load(artifact_dir=artifact_dir)


        self.default_threshold = self.model.metadata.decision_threshold if self.model.metadata else 0.5
        self.model_version = self.model.metadata.model_version if self.model.metadata else 'unknown'

    def score_texts(self, texts: List[str]) -> np.ndarray:

        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        X = self.feature_extractor.transform(normalized_texts)
        return self.model.score(X)
    

    def predict_labels(self, texts: List[str], threshold: float) -> np.ndarray:
        proba = self.score_texts(texts)

        return  (proba >= threshold).astype(int)
    

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> Dict[str, Any]:
        threshold = threshold if threshold is not None else self.default_threshold
        scores = self.score_texts(texts)
        labels = self.predict_labels(texts, threshold)

        return { 'model_version': self.model_version, 'threshold': threshold, 'results': [ {'score': x[0], 'label': int(x[1])}  for x in zip( scores, labels)] }
    


    def info(self) -> Dict[str, Any]:
        return {
            'artifact_dir': self.artifact_dir,
            'model_version': self.model_version,
            'default_threshold': self.default_threshold
        }