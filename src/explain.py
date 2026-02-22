from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shap

from src.features import normalize_texts
from src.signals.base import BaseSignal


@dataclass
class TokenContribution:
    token: str
    weight: float


@dataclass
class Explanation:
    text: str
    signal_name: str
    score: float
    contributions: list[TokenContribution]


class Explainer:
    def __init__(
        self,
        signal: BaseSignal,
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        self.signal = signal
        self.tokenizer = tokenizer or str.split

    def _mask_fn(self, tokens: list[str]) -> Callable[[np.ndarray], np.ndarray]:

        def fn(masks: np.ndarray) -> np.ndarray:
            texts = []
            for mask in masks:
                kept = [t for t, m in zip(tokens, mask) if m]
                texts.append(" ".join(kept) if kept else "")
            return self.signal.score(texts)

        return fn

    def explain(self, text: str, top_n: int = 10) -> Explanation:
        (normalized,) = normalize_texts([text])
        tokens = self.tokenizer(normalized)

        if not tokens:
            score = float(self.signal.score([normalized])[0])
            return Explanation(
                text=text,
                signal_name=self.signal.name,
                score=score,
                contributions=[],
            )
        mask_fn = self._mask_fn(tokens)

        background = np.zeros((1, len(tokens)))
        explainer = shap.KernelExplainer(mask_fn, background)

        sample = np.ones((1, len(tokens)))
        shap_values = explainer.shap_values(sample, silent=True)

        values = shap_values[0]

        paired = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)

        contributions = [TokenContribution(token=tok, weight=float(val)) for tok, val in paired[:top_n]]

        score = float(self.signal.score([normalized])[0])

        return Explanation(text=text, signal_name=self.signal.name, score=score, contributions=contributions)

    def explain_batch(self, texts: list[str], top_n: int = 10) -> list[Explanation]:
        return [self.explain(t, top_n=top_n) for t in texts]
