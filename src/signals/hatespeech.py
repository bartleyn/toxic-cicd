from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.features import normalize_texts
from src.signals.model import BaseModel
from src.signals.vader_labels import vader_compound


@dataclass(frozen=True)
class HateSpeechSpec:
    # classifier params
    C: float = 2.0
    max_iter: int = 200
    random_state: int = 42
    class_weight: str = "balanced"
    # word tfidf params
    word_max_features: int = 10000
    word_ngram_range: tuple = (1, 2)
    word_min_df: int = 2
    # char ngram params
    char_max_features: int = 5000
    char_ngram_range: tuple = (3, 5)
    char_min_df: int = 2


@dataclass(frozen=True)
class HateSpeechMetadata:
    model_version: str
    decision_threshold: float = 0.5
    label_positive: str = "hate_speech"
    label_negative: str = "non_hate_speech"


class HateSpeechModel(BaseModel):
    name = "hatespeech"

    def __init__(self, spec: HateSpeechSpec = None, metadata: HateSpeechMetadata = None):
        self.spec = spec if spec is not None else HateSpeechSpec()
        self.metadata = metadata
        self.clf = LogisticRegression(
            C=self.spec.C,
            max_iter=self.spec.max_iter,
            random_state=self.spec.random_state,
            class_weight=self.spec.class_weight,
        )
        self.word_vectorizer = TfidfVectorizer(
            max_features=self.spec.word_max_features,
            ngram_range=self.spec.word_ngram_range,
            min_df=self.spec.word_min_df,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            max_features=self.spec.char_max_features,
            ngram_range=self.spec.char_ngram_range,
            min_df=self.spec.char_min_df,
        )
        self.is_fitted = False

    def _build_features(self, texts: list[str], fit: bool = False):
        """Build combined feature matrix from word TF-IDF, char n-grams, and sentiment."""
        if fit:
            X_word = self.word_vectorizer.fit_transform(texts)
            X_char = self.char_vectorizer.fit_transform(texts)
        else:
            X_word = self.word_vectorizer.transform(texts)
            X_char = self.char_vectorizer.transform(texts)

        sentiment_scores = vader_compound(texts)
        X_sentiment = np.array(sentiment_scores).reshape(-1, 1)

        return hstack([X_word, X_char, X_sentiment])

    def fit(self, texts: list[str], y) -> HateSpeechModel:
        normalized = normalize_texts(texts)
        X = self._build_features(normalized, fit=True)
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def score(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        normalized = normalize_texts(texts)
        X = self._build_features(normalized, fit=False)
        return self.clf.predict_proba(X)[:, 1]

    def save(self, artifact_dir: str) -> None:
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.clf, os.path.join(artifact_dir, "model.joblib"))
        joblib.dump(self.word_vectorizer, os.path.join(artifact_dir, "word_vectorizer.joblib"))
        joblib.dump(self.char_vectorizer, os.path.join(artifact_dir, "char_vectorizer.joblib"))
        with open(os.path.join(artifact_dir, "spec.json"), "w") as f:
            json.dump(asdict(self.spec), f, indent=2, sort_keys=True)
        if self.metadata:
            with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
                json.dump(asdict(self.metadata), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, artifact_dir: str) -> HateSpeechModel:
        with open(os.path.join(artifact_dir, "spec.json")) as f:
            spec = HateSpeechSpec(**json.load(f))
        metadata_path = os.path.join(artifact_dir, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = HateSpeechMetadata(**json.load(f))
        model = cls(spec=spec, metadata=metadata)
        model.clf = joblib.load(os.path.join(artifact_dir, "model.joblib"))
        model.word_vectorizer = joblib.load(os.path.join(artifact_dir, "word_vectorizer.joblib"))
        model.char_vectorizer = joblib.load(os.path.join(artifact_dir, "char_vectorizer.joblib"))
        model.is_fitted = True
        return model
