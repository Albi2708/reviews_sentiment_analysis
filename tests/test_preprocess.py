"""Tests for :func:`pipeline.preprocess` against the real ftfy + spaCy stack."""
from __future__ import annotations

import pipeline


def test_empty_input_returns_empty_fields() -> None:
    """Empty string yields empty fields and no sentences."""
    out = pipeline.preprocess("")
    assert set(out) == {"model_input", "annotated_text", "sentences"}
    assert out["model_input"] == ""
    assert out["annotated_text"] == ""
    assert out["sentences"] == []


def test_whitespace_only_input_is_trimmed_to_empty() -> None:
    """Whitespace-only input strips down to empty model input."""
    out = pipeline.preprocess("   \n\t  ")
    assert out["model_input"] == ""
    assert out["sentences"] == []


def test_emojis_are_preserved() -> None:
    """ftfy normalization keeps emoji characters in the model input."""
    out = pipeline.preprocess("This is great! 😍🔥")
    assert "😍" in out["model_input"]
    assert "🔥" in out["model_input"]
    assert out["sentences"]
    assert isinstance(out["annotated_text"], str)


def test_very_long_text_is_handled() -> None:
    """A long, many-sentence review segments without error."""
    sentence = "The product works well and I am happy with it. "
    long_text = sentence * 300
    out = pipeline.preprocess(long_text)
    assert isinstance(out["model_input"], str)
    assert len(out["sentences"]) > 1
    assert out["model_input"].startswith("The product works well")


def test_multi_sentence_segmentation() -> None:
    """The spaCy sentencer splits a two-sentence review into two segments."""
    out = pipeline.preprocess("First sentence here. Second sentence there.")
    assert len(out["sentences"]) == 2


def test_negation_is_marked_for_display_only() -> None:
    """Negation scopes are bracketed in the display text but not in model input."""
    out = pipeline.preprocess("I do not like this product.")
    assert "[NEG:" in out["annotated_text"]
    assert "[NEG:" not in out["model_input"]
    assert out["model_input"] == "I do not like this product."
