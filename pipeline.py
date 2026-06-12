"""End-to-end review sentiment analysis pipeline.

Composes preprocessing (ftfy, spaCy segmentation, negation marking) with the
two Cardiff NLP models (sentiment + irony) and the validator rules. Public
entry point: :func:`analyze`. Run ``python pipeline.py`` for a smoke demo.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Literal

import ftfy
import spacy
from spacy.tokens import Doc
from transformers import pipeline as hf_pipeline

# Validation thresholds. LOW_CONF also gates the irony correction: confident
# positive -> invert to negative; uncertain -> neutral.
LOW_CONF_THRESHOLD = 0.70
IRONY_HIGH_CONF_THRESHOLD = 0.50
MULTIPOLARITY_TOP_CLASS_THRESHOLD = 0.60

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
SPACY_MODEL = "en_core_web_sm"

_NEG_LEMMAS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor",
    "none", "nobody", "nothing", "without", "nowhere",
})
_SCOPE_BOUNDARY_LEMMAS: frozenset[str] = frozenset({
    "but", "however", "although", "yet", "and", "or",
})

# Clause fallback: used only when spaCy returns a single segment for a review
# already flagged multipolar.
_CLAUSE_SPLIT_PATTERN: re.Pattern[str] = re.compile(
    r"\s*(?:,|;|\bbut\b|\bhowever\b|\balthough\b|\bunfortunately\b|\byet\b)\s*",
    flags=re.IGNORECASE,
)

Label = Literal["negative", "neutral", "positive"]


def _annotate_negation(doc: Doc) -> str:
    """Wrap negation scopes in ``[NEG: ...]`` for display; models get raw text.

    A cue lemma in ``_NEG_LEMMAS`` extends forward until punctuation or a
    scope-boundary lemma.
    """
    text = doc.text
    spans: list[tuple[int, int]] = []
    tokens = list(doc)
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok.lemma_.lower() in _NEG_LEMMAS:
            j = i + 1
            while j < n:
                t = tokens[j]
                if t.is_punct or t.lemma_.lower() in _SCOPE_BOUNDARY_LEMMAS:
                    break
                j += 1
            start = tok.idx
            end = tokens[j - 1].idx + len(tokens[j - 1].text)
            spans.append((start, end))
            i = max(j, i + 1)
        else:
            i += 1
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        out.append(f"[NEG: {text[start:end]}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def preprocess(text: str) -> dict[str, Any]:
    """Normalize (ftfy) and segment (spaCy) the input.

    Returns a dict with ``model_input`` (text fed to the models),
    ``annotated_text`` (``[NEG: ...]`` markup for display), and ``sentences``
    (segments for the multipolarity branch).
    """
    fixed = ftfy.fix_text(text).strip()
    nlp = _get_nlp()
    doc = nlp(fixed)
    annotated = _annotate_negation(doc)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    return {
        "model_input": fixed,
        "annotated_text": annotated,
        "sentences": sentences,
    }


# Lazily-loaded, cached singletons so importing this module is cheap; FastAPI
# warms them at startup.

@functools.cache
def _get_sentiment_pipe():
    """Cached sentiment pipeline."""
    return hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None)


@functools.cache
def _get_irony_pipe():
    """Cached irony pipeline."""
    return hf_pipeline("text-classification", model=IRONY_MODEL)


@functools.cache
def _get_nlp():
    """Cached spaCy pipeline."""
    return spacy.load(SPACY_MODEL)


def _predict_sentiment(text: str) -> dict[str, Any]:
    """Return the sentiment model's top label, confidence, and 3-class distribution."""
    raw = _get_sentiment_pipe()(text, truncation=True, max_length=512)[0]
    distribution: dict[str, float] = {
        item["label"].lower(): float(item["score"]) for item in raw
    }
    for cls in ("negative", "neutral", "positive"):
        distribution.setdefault(cls, 0.0)
    top_label = max(distribution, key=distribution.get)
    return {
        "label": top_label,
        "confidence": distribution[top_label],
        "distribution": distribution,
    }


def _predict_irony(text: str) -> dict[str, Any]:
    """Return the irony model's top label and confidence."""
    out = _get_irony_pipe()(text, truncation=True, max_length=512)[0]
    return {"label": out["label"].lower(), "confidence": float(out["score"])}


def _clause_split(text: str) -> list[str]:
    """Split on commas, semicolons, and contrastive connectives."""
    return [p.strip() for p in _CLAUSE_SPLIT_PATTERN.split(text) if p.strip()]


def _score_segments(segments: list[str]) -> list[dict[str, Any]]:
    """Score each segment with the sentiment model (no per-segment distribution)."""
    results: list[dict[str, Any]] = []
    for seg in segments:
        s = _predict_sentiment(seg)
        results.append({"text": seg, "label": s["label"], "confidence": s["confidence"]})
    return results


def analyze(text: str) -> dict[str, Any]:
    """Run preprocessor -> both models -> validator; return the result dict.

    Top-level keys: ``text``, ``preprocessed_text``, ``label``, ``confidence``,
    ``flags`` (low_confidence/model_agreement/multipolarity), ``sentiment_raw``,
    ``irony``, ``segments`` (list of {text, label, confidence} or None). When
    both model_agreement and multipolarity fire, the irony branch resolves the
    label and ``segments`` stays informational; the inline branch documents the
    confidence convention.
    """
    prep = preprocess(text)
    model_input: str = prep["model_input"]
    sentiment_raw = _predict_sentiment(model_input)
    irony_pred = _predict_irony(model_input)

    multipolarity = (
        sentiment_raw["confidence"] < MULTIPOLARITY_TOP_CLASS_THRESHOLD
    )
    model_agreement = (
        irony_pred["label"] == "irony"
        and irony_pred["confidence"] >= IRONY_HIGH_CONF_THRESHOLD
    )

    # Per-segment view is built whenever multipolarity fires, for the UI.
    segments_out: list[dict[str, Any]] | None
    if multipolarity:
        segments = prep["sentences"]
        if len(segments) <= 1:
            fallback = _clause_split(model_input)
            if len(fallback) > 1:
                segments = fallback
        segments_out = _score_segments(segments)
    else:
        segments_out = None

    # Irony correction takes precedence over multipolarity.
    if model_agreement:
        if sentiment_raw["label"] == "positive":
            if sentiment_raw["confidence"] >= LOW_CONF_THRESHOLD:
                resolved_label = "negative"
            else:
                resolved_label = "neutral"
        else:
            resolved_label = sentiment_raw["label"]
        resolved_confidence = sentiment_raw["confidence"]
        # Prefer the last segment's in-context confidence when it agrees.
        if segments_out is not None and segments_out[-1]["label"] == resolved_label:
            resolved_confidence = segments_out[-1]["confidence"]
    elif multipolarity:
        # Last-clause wins: the final segment sets the aggregate label.
        assert segments_out is not None
        last = segments_out[-1]
        resolved_label = last["label"]
        resolved_confidence = last["confidence"]
    else:
        resolved_label = sentiment_raw["label"]
        resolved_confidence = sentiment_raw["confidence"]

    low_confidence = resolved_confidence < LOW_CONF_THRESHOLD

    return {
        "text": text,
        "preprocessed_text": prep["annotated_text"],
        "label": resolved_label,
        "confidence": resolved_confidence,
        "flags": {
            "low_confidence": low_confidence,
            "model_agreement": model_agreement,
            "multipolarity": multipolarity,
        },
        "sentiment_raw": sentiment_raw,
        "irony": irony_pred,
        "segments": segments_out,
    }


def _demo() -> None:
    """Print pipeline output for a handful of hand-picked reviews."""
    samples = [
        # clear positive
        "Honestly the best laptop I've ever owned. Battery lasts all day, the screen is gorgeous, and it boots in under five seconds. Worth every penny.",
        # structured sarcasm (known to fool both models)
        "This food supplement is amazing! If you are looking for constant stomach and headaches, with no benefits, this is the product for you! 10/10.",
        # multipolar, two sentences (spaCy sentencer)
        "Great materials and shape, very modern. Unfortunately, the balancing, which is the most important feature, is a complete mess.",
        # multipolar, single sentence (clause fallback)
        "Great materials but the balancing is a complete mess.",
        # blunt sarcasm (irony inverts positive -> negative)
        "Oh fantastic, another pair of headphones that broke after a week. Truly the legendary build quality the ad promised.",
        # explicit negation
        "I wouldn't say this vacuum is bad, but it's nothing I'd recommend either - suction drops on carpet and the canister is a pain to empty.",
    ]
    for i, text in enumerate(samples, 1):
        print(f"\n=== [{i}] {text[:90]}{'...' if len(text) > 90 else ''}")
        result = analyze(text)
        print(f"  preprocessed: {result['preprocessed_text']}")
        print(f"  label:        {result['label']}  (conf {result['confidence']:.3f})")
        print(f"  flags:        {result['flags']}")
        raw = result["sentiment_raw"]
        print(f"  raw sentiment {raw['label']:<8s} (conf {raw['confidence']:.3f})  dist {raw['distribution']}")
        print(f"  irony:        {result['irony']}")
        if result["segments"]:
            print(f"  segments ({len(result['segments'])}):")
            for s in result["segments"]:
                snippet = s["text"][:80] + ("..." if len(s["text"]) > 80 else "")
                print(f"    [{s['label']:<8s} {s['confidence']:.2f}] {snippet}")


if __name__ == "__main__":
    _demo()
