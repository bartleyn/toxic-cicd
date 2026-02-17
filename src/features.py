import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import validate_texts

"""
Normalization
"""


def normalize_texts(texts: list[str]) -> list[str]:
    """
    Normalize a list of texts by converting to lowercase and removing extra whitespace.

    :param texts: List of strings to normalize
    :type texts: List[str]
    :return: Normalized list of strings
    :rtype: List[str]
    """
    return [t.strip().lower() for t in texts]


class TextFeatureExtractor:
    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2), min_df: int = 2):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
        self.is_fitted = False

    def fit(self, texts: list[str]) -> None:
        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        self.vectorizer.fit(normalized_texts)
        self.is_fitted = True

        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The extractor must be fitted before use")
        normalized_texts = normalize_texts(texts)
        return self.vectorizer.transform(normalized_texts).toarray()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
