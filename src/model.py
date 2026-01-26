

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple
import json
import joblib
import numpy as np


from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class ModelSpec:
    model_type: str = 'logistic'
    C: float = 2.0
    max_iter: int = 200
    random_state: int = 42
    class_weight: str = 'balanced'


@dataclass(frozen=True)
class ModelMetadata:

    model_version: str
    decision_threshold: float = 0.5
    label_positive: str = 'toxic'
    label_negative: str = 'non_toxic'


class ToxicityModel:


    def __init__(self, spec: ModelSpec = None, metadata: ModelMetadata = None):
        self.spec = spec if spec is not None else ModelSpec()
        self.metadata = metadata 
        self.clf = LogisticRegression(
            C=self.spec.C,
            max_iter=self.spec.max_iter,
            random_state=self.spec.random_state,
            class_weight=self.spec.class_weight
        )
        self.is_fitted = False


    def fit(self, X, y) -> "ToxicityModel":
        self.clf.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.clf.predict_proba(X)
    
    def score(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba[:, 1]  # Probability of the positive class


    def predict_labels(self, X, threshold: float = None) -> np.ndarray:
        if threshold is None:
            threshold = self.metadata.decision_threshold if self.metadata else 0.5

        return (self.score(X) >= threshold).astype(int)
    

    '''
    I/O
    '''

    def save(self, artifact_dir: str) -> None:
        '''
        Save the model, hyperparameters and metadata
        '''

        os.makedirs(artifact_dir, exist_ok=True)

        joblib.dump(self.clf, os.path.join(artifact_dir, 'model.joblib'))

        with open(os.path.join(artifact_dir, 'spec.json'), 'w') as f:
            json.dump(self.spec, f, indent=2, sort_keys=True)

        if self.metadata:
            with open(os.path.join(artifact_dir, 'metadata.json'), 'w') as f:
                json.dump(self.metadata, f, indent=2, sort_keys=True)


    def load(cls, artifact_dir: str) -> "ToxicityModel":
        '''
        Load the model, hyperparameters and metadata
        '''

        with open(os.path.join(artifact_dir, 'spec.json'), 'r') as f:
            spec_dict = json.load(f)
            spec = ModelSpec(**spec_dict)

        metadata_path = os.path.join(artifact_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata(**metadata_dict)
        else:
            metadata = None

        model = cls(spec=spec, metadata=metadata)
        model.clf = joblib.load(os.path.join(artifact_dir, 'model.joblib'))
        model.is_fitted = True

        return model