from typing import List
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


'''
Validating text data
'''

def validate_texts(texts: List[str]) -> None:
    '''
    Docstring for validate_texts
    
    :param texts: Description
    :type texts: List[str]
    '''

    if not isinstance(texts, list):
        raise TypeError("Input must be a list of strings.")
    
    if len(texts) == 0:
        raise ValueError("Input text list is empty")
    
    for t in texts:
        if not isinstance(t, str):
            raise TypeError("Each text in the input must be a string.")
        if len(t.strip()) == 0:
            raise ValueError("Texts must not be empty or whitespace only.")
        

'''
Normalization
'''
def normalize_texts(texts: List[str]) -> List[str]:
    '''
    Normalize a list of texts by converting to lowercase and removing extra whitespace.

    :param texts: List of strings to normalize
    :type texts: List[str]
    :return: Normalized list of strings
    :rtype: List[str]
    '''
    return [t.strip().lower() for t in texts]


class TextFeatureExtractor:
    def __init__(self, max_features: int = 10000, 
                 ngram_range: tuple = (1, 2), 
                 min_df: int = 2):
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                          ngram_range=ngram_range,
                                          min_df=min_df)
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        self.vectorizer.fit(normalized_texts)
        self.is_fitted = True

        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The extractor must be fitted before use")
        validate_texts(texts)
        normalized_texts = normalize_texts(texts)
        return self.vectorizer.transform(normalized_texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
    
    