

from __future__ import annotations

import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from dataclasses import dataclass, asdict

from typing import Any, Dict, Optional

import json

@dataclass(frozen=True)
class SentimentSpec:
    model_type: str = 'ridge'
    alpha: float = 1.0
    max_iter: int = 1000
    random_state: int = 42


@dataclass(frozen=True)
class SentimentMetadata:

    model_version: str
    target_range_min: float = -1.0
    target_range_max: float = 1.0
    label_source: str = 'vader_compound'


class SentimentModel:


    def __init__(self, spec: SentimentSpec = None, metadata: SentimentMetadata = None):
        self.spec = spec if spec is not None else SentimentSpec()
        self.metadata = metadata if metadata is not None else SentimentMetadata(model_version='0.1.0')
        self.model = Ridge(
            alpha = self.spec.alpha,
            max_iter = self.spec.max_iter,
            random_state = self.spec.random_state
        )
        self.is_fitted = False

    def fit(self, X, y) -> "SentimentModel":
        self.model.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        pred = self.model.predict(X)
        pred_clipped = np.clip(pred, self.metadata.target_range_min, self.metadata.target_range_max)
        return pred_clipped
    
    def save(self, artifact_dir: str) -> None:
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(artifact_dir, 'model.joblib'))
        with open(os.path.join(artifact_dir, 'spec.json'), 'w') as f:
            json.dump(asdict(self.spec), f, indent=2, sort_keys=True)
        with open(os.path.join(artifact_dir, 'metadata.json'), 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, artifact_dir: str) -> "SentimentModel":
        model = joblib.load(os.path.join(artifact_dir, 'model.joblib'))
        with open(os.path.join(artifact_dir, 'spec.json'), 'r') as f:
            spec_dict = json.load(f)
            spec = SentimentSpec(**spec_dict)
        with open(os.path.join(artifact_dir, 'metadata.json'), 'r') as f:
            metadata_dict = json.load(f)
            metadata = SentimentMetadata(**metadata_dict)
        instance = cls(spec=spec, metadata=metadata)
        instance.model = model
        instance.is_fitted = True
        return instance
    
    def score(self, X) -> np.ndarray:
        return self.predict(X)