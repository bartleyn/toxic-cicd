from __future__ import annotations

import pytest

from src.schemas import ItemResult
from src.utils import validate_texts


class TestValidateTexts:
    def test_valid_input(self):
        validate_texts(["hello", "world"])  # should not raise

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_texts([])

    def test_non_list_raises(self):
        with pytest.raises(TypeError, match="list"):
            validate_texts("not a list")

    def test_non_string_element_raises(self):
        with pytest.raises(TypeError, match="string"):
            validate_texts(["hello", 123])

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_texts(["hello", "   "])


class TestItemResultSchema:
    def test_valid_item(self):
        item = ItemResult(label=1, scores={"toxicity": 0.9, "sentiment": 0.3})

        assert item.label == 1
        assert item.scores["toxicity"] == 0.9

    def test_empty_scores(self):
        item = ItemResult(scores={}, label=0)
        assert item.scores == {}
