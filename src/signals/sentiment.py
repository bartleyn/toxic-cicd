from __future__ import annotations

import os

import numpy as np
import torch
from transformers import pipeline

from src.signals.base import BaseSignal

DEFAULT_CHUNK_SIZE = 32


class SentimentModel(BaseSignal):
    """Drop-in sentiment signal backed by DistilBERT SST-2.

    Texts are sorted by length before batching.
    """

    name = "sentiment"

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        chunk_size: int | None = None,
    ):
        self._pipe = pipeline(
            "sentiment-analysis", revision="714eb0f", model=model_name, truncation=True, max_length=512
        )
        self._model = self._pipe.model.eval()
        self._tokenizer = self._pipe.tokenizer

        if chunk_size is None:
            chunk_size = int(os.getenv("SENTIMENT_CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
        self.chunk_size = chunk_size

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        id2label = self._model.config.id2label
        scores = np.empty(len(texts), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

        for start in range(0, len(order), self.chunk_size):
            idx = order[start : start + self.chunk_size]
            encoded = self._tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.inference_mode():
                probs = torch.softmax(self._model(**encoded).logits, dim=-1)
            top = probs.max(dim=-1)

            for position, i in enumerate(idx):
                score = float(top.values[position])
                label = id2label[int(top.indices[position])]
                scores[i] = score if label == "POSITIVE" else -score

        return scores
