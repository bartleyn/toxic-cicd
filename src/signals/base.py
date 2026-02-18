from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSignal(ABC):
    """
    This is an interface all scoring signals need to implement
    """

    name: str
    input_type: str  # so far, "tfidf" | "text"

    @abstractmethod
    def score(self, inputs) -> np.ndarray: ...
