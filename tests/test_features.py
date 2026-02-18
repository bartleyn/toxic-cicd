from __future__ import annotations

import numpy as np
import pytest

from src.features import TextFeatureExtractor, normalize_texts


def test_normalize_texts_lowercases():
    assert normalize_texts(["Hello", "WORLD"]) == ["hello", "world"]


def test_normalize_texts_strips_whitespace():
    assert normalize_texts(["  hi  ", "there "]) == ["hi", "there"]


def test_extractor_fit_transform_returns_array():
    texts = ["the cat sat", "the dog ran", "a bird flew"]
    ext = TextFeatureExtractor(max_features=50, min_df=1)
    X = ext.fit_transform(texts)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 3


def test_extractor_transform_before_fit_raises():
    ext = TextFeatureExtractor()
    with pytest.raises(RuntimeError, match="fitted"):
        ext.transform(["hello"])


def test_extractor_empty_input_raises():
    ext = TextFeatureExtractor()
    with pytest.raises(ValueError):
        ext.fit([])
