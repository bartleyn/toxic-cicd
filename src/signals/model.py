from abc import abstractmethod

from src.signals.base import BaseSignal


class BaseModel(BaseSignal):

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def save(self, artifact_dir: str): ...

    @classmethod
    @abstractmethod
    def load(cls, artifact_dir: str): ...
