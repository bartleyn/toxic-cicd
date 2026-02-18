from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSignal(ABC):
    """Interface all scoring signals must implement."""

    name: str

    @abstractmethod
    def score(self, texts: list[str]) -> np.ndarray:
        """Return a 1-D array of scores, one per input text."""
        ...
