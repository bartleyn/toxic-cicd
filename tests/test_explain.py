import numpy as np

from src.explain import Explainer, Explanation
from src.signals.base import BaseSignal


class FakeSignal(BaseSignal):
    name = "fake"

    def score(self, texts: list[str]) -> np.ndarray:
        return np.array([1.0 if "bad" in t.split() else 0.0 for t in texts])


class TestExplainer:
    def setup_method(self):
        self.signal = FakeSignal()
        self.explainer = Explainer(self.signal)

    def test_explain_returns_explanation(self):
        result = self.explainer.explain("this is bad text")
        assert isinstance(result, Explanation)
        assert result.signal_name == "fake"
        assert result.score == 1.0
        assert len(result.contributions) > 0

    def test_top_contribution_is_bad(self):
        result = self.explainer.explain("this is bad text")
        top_token = result.contributions[0].token
        assert top_token == "bad"

    def test_top_n_limits_output(self):
        result = self.explainer.explain("This is bad text with many words", top_n=2)
        assert len(result.contributions) <= 2

    def test_contributions_are_sorted_by_magnitude(self):
        result = self.explainer.explain("this is bad text")
        weights = [abs(c.weight) for c in result.contributions]
        assert weights == sorted(weights, reverse=True)

    def test_empty_text(self):
        result = self.explainer.explain("")
        assert result.contributions == []

    def test_explain_batch(self):
        results = self.explainer.explain_batch(["bad text", "good text"], top_n=3)
        assert len(results) == 2
        assert all(isinstance(r, Explanation) for r in results)

    def test_custom_tokenizer(self):
        explainer = Explainer(self.signal, tokenizer=lambda t: t.split(","))
        result = explainer.explain("good,bad,okay")
        assert any(c.token == "bad" for c in result.contributions)
