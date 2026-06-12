"""Unit tests for the validator branches in :func:`pipeline.analyze`.

The model wrappers are monkeypatched with deterministic fakes so the branch
logic (irony correction, last-clause-wins, low-confidence gate, and their
precedence) is exercised without the real transformers. Locked thresholds:
LOW_CONF=0.70, IRONY_HIGH_CONF=0.50, MULTIPOLARITY_TOP_CLASS=0.60.
"""
from __future__ import annotations

from typing import Any

import pytest

import pipeline


def _distribution(dist: dict[str, float]) -> dict[str, Any]:
    """Build a sentiment result (label = argmax) from a partial distribution."""
    full = {"negative": 0.0, "neutral": 0.0, "positive": 0.0, **dist}
    label = max(full, key=full.get)
    return {"label": label, "confidence": full[label], "distribution": full}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_input: str,
    sentences: list[str],
    sentiment: dict[str, dict[str, float]],
    irony_label: str = "non_irony",
    irony_conf: float = 0.99,
    annotated: str | None = None,
) -> None:
    """Patch the pipeline's model wrappers with text-keyed fakes.

    ``sentiment`` maps an exact input string to a partial distribution; a
    lookup miss raises ``KeyError`` so a mis-specified test fails loudly.
    """
    monkeypatch.setattr(
        pipeline,
        "preprocess",
        lambda text: {
            "model_input": model_input,
            "annotated_text": annotated if annotated is not None else model_input,
            "sentences": list(sentences),
        },
    )
    monkeypatch.setattr(
        pipeline, "_predict_sentiment", lambda text: _distribution(sentiment[text])
    )
    monkeypatch.setattr(
        pipeline,
        "_predict_irony",
        lambda text: {"label": irony_label, "confidence": irony_conf},
    )


def test_clear_positive_no_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """High-confidence single-polarity review: no flags, full schema present."""
    _install(
        monkeypatch,
        model_input="I love it.",
        sentences=["I love it."],
        sentiment={"I love it.": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}},
    )
    out = pipeline.analyze("I love it.")

    assert out["label"] == "positive"
    assert out["confidence"] == pytest.approx(0.95)
    assert out["flags"] == {
        "low_confidence": False,
        "model_agreement": False,
        "multipolarity": False,
    }
    assert out["segments"] is None

    assert set(out) == {
        "text", "preprocessed_text", "label", "confidence",
        "flags", "sentiment_raw", "irony", "segments",
    }
    assert out["text"] == "I love it."
    assert set(out["sentiment_raw"]) == {"label", "confidence", "distribution"}
    assert set(out["sentiment_raw"]["distribution"]) == {"negative", "neutral", "positive"}
    assert set(out["irony"]) == {"label", "confidence"}


def test_low_confidence_flag_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confidence in [0.60, 0.70): low-confidence flag fires, nothing else."""
    _install(
        monkeypatch,
        model_input="It is okay I guess.",
        sentences=["It is okay I guess."],
        sentiment={"It is okay I guess.": {"positive": 0.65, "neutral": 0.20, "negative": 0.15}},
    )
    out = pipeline.analyze("It is okay I guess.")

    assert out["label"] == "positive"
    assert out["flags"]["low_confidence"] is True
    assert out["flags"]["model_agreement"] is False
    assert out["flags"]["multipolarity"] is False
    assert out["segments"] is None


def test_irony_inverts_confident_positive_to_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """positive + irony + conf >= LOW_CONF → invert to negative."""
    _install(
        monkeypatch,
        model_input="Oh great, it broke again.",
        sentences=["Oh great, it broke again."],
        sentiment={"Oh great, it broke again.": {"positive": 0.85, "neutral": 0.10, "negative": 0.05}},
        irony_label="irony",
        irony_conf=0.80,
    )
    out = pipeline.analyze("Oh great, it broke again.")

    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.85)
    assert out["flags"]["model_agreement"] is True
    assert out["flags"]["multipolarity"] is False
    assert out["sentiment_raw"]["label"] == "positive"
    assert out["segments"] is None


def test_irony_downgrades_uncertain_positive_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    """positive + irony + conf < LOW_CONF → downgrade to neutral."""
    _install(
        monkeypatch,
        model_input="Sure, fantastic.",
        sentences=["Sure, fantastic."],
        sentiment={"Sure, fantastic.": {"positive": 0.65, "neutral": 0.20, "negative": 0.15}},
        irony_label="irony",
        irony_conf=0.70,
    )
    out = pipeline.analyze("Sure, fantastic.")

    assert out["label"] == "neutral"
    assert out["flags"]["model_agreement"] is True
    assert out["flags"]["multipolarity"] is False
    assert out["flags"]["low_confidence"] is True
    assert out["sentiment_raw"]["label"] == "positive"


def test_irony_keeps_negative_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """irony + an already-negative label leaves the label untouched."""
    _install(
        monkeypatch,
        model_input="This is awful and I hate it.",
        sentences=["This is awful and I hate it."],
        sentiment={"This is awful and I hate it.": {"negative": 0.90, "neutral": 0.07, "positive": 0.03}},
        irony_label="irony",
        irony_conf=0.80,
    )
    out = pipeline.analyze("This is awful and I hate it.")

    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.90)
    assert out["flags"]["model_agreement"] is True
    assert out["segments"] is None


def test_irony_keeps_neutral_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """irony + an already-neutral label leaves the label untouched."""
    _install(
        monkeypatch,
        model_input="It exists, more or less.",
        sentences=["It exists, more or less."],
        sentiment={"It exists, more or less.": {"neutral": 0.80, "positive": 0.12, "negative": 0.08}},
        irony_label="irony",
        irony_conf=0.65,
    )
    out = pipeline.analyze("It exists, more or less.")

    assert out["label"] == "neutral"
    assert out["flags"]["model_agreement"] is True
    assert out["segments"] is None


def test_irony_below_threshold_does_not_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    """irony label but conf < IRONY_HIGH_CONF: gate stays closed, no correction."""
    _install(
        monkeypatch,
        model_input="Oh great, it broke again.",
        sentences=["Oh great, it broke again."],
        sentiment={"Oh great, it broke again.": {"positive": 0.85, "neutral": 0.10, "negative": 0.05}},
        irony_label="irony",
        irony_conf=0.49,
    )
    out = pipeline.analyze("Oh great, it broke again.")

    assert out["label"] == "positive"
    assert out["flags"]["model_agreement"] is False


def test_multipolarity_last_clause_wins_multi_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low full-pass confidence + multi-sentence: last segment decides the label."""
    first = "The screen is gorgeous."
    last = "But the battery dies in an hour."
    full = first + " " + last
    _install(
        monkeypatch,
        model_input=full,
        sentences=[first, last],
        sentiment={
            full: {"positive": 0.45, "neutral": 0.33, "negative": 0.22},
            first: {"positive": 0.92, "neutral": 0.05, "negative": 0.03},
            last: {"negative": 0.88, "neutral": 0.08, "positive": 0.04},
        },
    )
    out = pipeline.analyze(full)

    assert out["flags"]["multipolarity"] is True
    assert out["flags"]["model_agreement"] is False
    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.88)
    assert out["segments"] is not None
    assert [s["label"] for s in out["segments"]] == ["positive", "negative"]
    assert out["segments"][-1]["label"] == out["label"]


def test_multipolarity_clause_fallback_single_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    """spaCy returns one sentence → the clause splitter rescues the segment view."""
    text = "Great materials but the balancing is a complete mess."
    clauses = pipeline._clause_split(text)
    assert len(clauses) == 2, "precondition: clause splitter yields two clauses"
    _install(
        monkeypatch,
        model_input=text,
        sentences=[text],
        sentiment={
            text: {"positive": 0.40, "neutral": 0.35, "negative": 0.25},
            clauses[0]: {"positive": 0.80, "neutral": 0.12, "negative": 0.08},
            clauses[1]: {"negative": 0.90, "neutral": 0.07, "positive": 0.03},
        },
    )
    out = pipeline.analyze(text)

    assert out["flags"]["multipolarity"] is True
    assert out["segments"] is not None
    assert len(out["segments"]) == 2
    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.90)


def test_both_flags_uses_matching_segment_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both flags fire: irony resolves the label; the matching last segment lends
    its in-context confidence in place of the diluted raw value."""
    first = "Looks nice."
    last = "Completely useless though."
    full = first + " " + last
    _install(
        monkeypatch,
        model_input=full,
        sentences=[first, last],
        sentiment={
            full: {"negative": 0.45, "neutral": 0.30, "positive": 0.25},
            first: {"positive": 0.70, "neutral": 0.20, "negative": 0.10},
            last: {"negative": 0.80, "neutral": 0.12, "positive": 0.08},
        },
        irony_label="irony",
        irony_conf=0.75,
    )
    out = pipeline.analyze(full)

    assert out["flags"]["model_agreement"] is True
    assert out["flags"]["multipolarity"] is True
    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.80)
    assert out["segments"] is not None
    assert len(out["segments"]) == 2


def test_both_flags_mismatched_segment_keeps_raw_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both flags fire but the last segment disagrees with the resolved label →
    the raw full-pass confidence is kept rather than an unrelated segment's."""
    first = "Hate the setup."
    last = "But wow, it works great now."
    full = first + " " + last
    _install(
        monkeypatch,
        model_input=full,
        sentences=[first, last],
        sentiment={
            full: {"negative": 0.45, "neutral": 0.30, "positive": 0.25},
            first: {"negative": 0.85, "neutral": 0.10, "positive": 0.05},
            last: {"positive": 0.92, "neutral": 0.05, "negative": 0.03},
        },
        irony_label="irony",
        irony_conf=0.75,
    )
    out = pipeline.analyze(full)

    assert out["flags"]["model_agreement"] is True
    assert out["flags"]["multipolarity"] is True
    assert out["label"] == "negative"
    assert out["confidence"] == pytest.approx(0.45)
    assert out["flags"]["low_confidence"] is True
    assert out["segments"][-1]["label"] == "positive"
