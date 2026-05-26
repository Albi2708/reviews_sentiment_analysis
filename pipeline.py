"""End-to-end review sentiment analysis pipeline.

Composes preprocessing (ftfy unicode normalization, spaCy sentence segmentation,
negation-scope marking) with the two Cardiff NLP transformer models from the
concept document (sentiment + irony) and the validator rules locked in at
roadmap item 4. Public entry point: :func:`analyze`.

Run ``python pipeline.py`` to execute a small smoke demo on a handful of
hand-picked reviews.
"""
from __future__ import annotations

import functools
import re
from typing import Any, Literal

import ftfy
import spacy
from spacy.tokens import Doc
from transformers import pipeline as hf_pipeline

# --- Validation thresholds (locked at roadmap item 4) ----------------------

# Below this, the aggregate prediction is flagged low-confidence. Also gates
# the irony correction: positive + irony fires + sentiment conf >= this →
# invert to negative; otherwise → downgrade to neutral.
LOW_CONF_THRESHOLD = 0.60

# Irony detector confidence at or above this counts as the detector firing.
IRONY_HIGH_CONF_THRESHOLD = 0.70

# Top-class sentiment probability below this on the full-review pass triggers
# the multipolarity branch (segment + last-clause-wins aggregate).
MULTIPOLARITY_TOP_CLASS_THRESHOLD = 0.45

# --- Model identifiers (concept doc §2 / §5) -------------------------------

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
SPACY_MODEL = "en_core_web_sm"

# --- Negation marking ------------------------------------------------------

_NEG_LEMMAS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor",
    "none", "nobody", "nothing", "without", "nowhere",
})
_SCOPE_BOUNDARY_LEMMAS: frozenset[str] = frozenset({
    "but", "however", "although", "yet", "and", "or",
})

# --- Multipolarity clause fallback (item 5 decision: option B) -------------
# Used only when spaCy's sentencer returns a single segment for a review that
# has already been flagged multipolar by the full-review confidence check.

_CLAUSE_SPLIT_PATTERN: re.Pattern[str] = re.compile(
    r"\s*(?:,|;|\bbut\b|\bhowever\b|\balthough\b|\bunfortunately\b|\byet\b)\s*",
    flags=re.IGNORECASE,
)

Label = Literal["negative", "neutral", "positive"]


# --- Preprocessor ----------------------------------------------------------

def _annotate_negation(doc: Doc) -> str:
    """Wrap negation scopes in ``[NEG: ...]`` brackets for transparency.

    Scopes are identified lexically: a cue token whose lemma is in
    ``_NEG_LEMMAS`` extends forward until a punctuation token or a
    scope-boundary lemma (``and``/``or``/``but``/...). The annotated string
    is for human display only — the models receive the unannotated
    ``model_input``.
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
    """Normalize input and segment it for downstream models.

    Args:
        text: Raw review text from the user.

    Returns:
        Dict with three keys:
            ``model_input`` — ftfy-normalized, whitespace-trimmed text fed to
            the transformers.
            ``annotated_text`` — the same text with ``[NEG: ...]`` markers,
            for UI display.
            ``sentences`` — list of sentence-level segments (spaCy sentencer),
            used by the multipolarity branch.
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


# --- Model wrappers --------------------------------------------------------
# Lazily loaded singletons so importing this module is cheap. FastAPI (item 6)
# can eagerly warm them at startup by calling preprocess + analyze once.

@functools.cache
def _get_sentiment_pipe():
    return hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, top_k=None)


@functools.cache
def _get_irony_pipe():
    return hf_pipeline("text-classification", model=IRONY_MODEL)


@functools.cache
def _get_nlp():
    return spacy.load(SPACY_MODEL)


def _predict_sentiment(text: str) -> dict[str, Any]:
    """Run the sentiment model and return top label + full 3-class distribution."""
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
    """Run the irony model and return its top label + confidence."""
    out = _get_irony_pipe()(text, truncation=True, max_length=512)[0]
    return {"label": out["label"].lower(), "confidence": float(out["score"])}


# --- Multipolarity branch helpers ------------------------------------------

def _clause_split(text: str) -> list[str]:
    """Clause-level fallback splitter (item 5 decision: option B).

    Used only when the spaCy sentencer collapses a multipolar review into a
    single sentence — splits on commas, semicolons, and a small set of
    contrastive connectives so the per-segment view is still meaningful.
    """
    return [p.strip() for p in _CLAUSE_SPLIT_PATTERN.split(text) if p.strip()]


def _score_segments(segments: list[str]) -> list[dict[str, Any]]:
    """Score each segment with the sentiment model; drop the full distribution
    from the per-segment view (only the aggregate keeps it)."""
    results: list[dict[str, Any]] = []
    for seg in segments:
        s = _predict_sentiment(seg)
        results.append({"text": seg, "label": s["label"], "confidence": s["confidence"]})
    return results


# --- Public entry point ----------------------------------------------------

def analyze(text: str) -> dict[str, Any]:
    """Run preprocessor → both models → validator and return the output dict.

    Output schema (item 5)::

        {
          "text": str,                          # original input
          "preprocessed_text": str,             # annotated with [NEG: ...]
          "label": "negative"|"neutral"|"positive",  # resolved label
          "confidence": float,                  # see note below
          "flags": {
            "low_confidence": bool,
            "model_agreement": bool,            # irony fires at high confidence
            "multipolarity": bool,
          },
          "sentiment_raw": {                    # pre-correction model output
            "label": "...",
            "confidence": float,
            "distribution": {"negative": ..., "neutral": ..., "positive": ...},
          },
          "irony": {"label": "irony"|"non_irony", "confidence": float},
          "segments": list[{text, label, confidence}] | None,
        }

    Branch precedence (item 5 decision): when both ``flags.model_agreement``
    and ``flags.multipolarity`` are True, the irony correction branch
    resolves the label; the per-segment view is still populated and
    surfaced as ``segments`` for the UI.

    Confidence convention: ``confidence`` carries the sentiment model's
    confidence on its raw full-review prediction in the irony and
    trust-raw branches, and the last segment's confidence in the
    multipolarity branch. When both flags fire and the last segment's
    label matches the resolved label, the segment's confidence is
    reported instead of the raw full-pass value — it reflects the
    in-context evidence for the chosen label. When
    ``flags.model_agreement`` is True the top-level ``label`` may have
    been overridden by the irony detector — see ``sentiment_raw.label``
    for the original prediction and ``irony.confidence`` for the
    override-signal strength.

    ``segments`` is populated whenever ``flags.multipolarity`` is True,
    regardless of which branch resolved the label.
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

    # Per-segment view is always built when the multipolarity flag is set, so
    # the UI can show it regardless of which branch resolved the label.
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

    # Resolve label and confidence. Irony correction takes precedence over
    # multipolarity (item 5 decision): when both flags fire, the irony branch
    # determines the label and the per-segment view is informational.
    if model_agreement:
        if sentiment_raw["label"] == "positive":
            if sentiment_raw["confidence"] >= LOW_CONF_THRESHOLD:
                resolved_label = "negative"
            else:
                resolved_label = "neutral"
        else:
            resolved_label = sentiment_raw["label"]
        resolved_confidence = sentiment_raw["confidence"]
        # When multipolarity also fired, the last segment carries in-context
        # evidence for the resolved label that the raw full-pass softmax
        # diluted. Use it iff its label matches the resolved label —
        # otherwise we'd be reporting an unrelated segment's confidence.
        if segments_out is not None and segments_out[-1]["label"] == resolved_label:
            resolved_confidence = segments_out[-1]["confidence"]
    elif multipolarity:
        # Last-clause wins (CLAUDE.md "Design decisions").
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


# --- Smoke demo ------------------------------------------------------------

def _demo() -> None:
    """Compact terminal demo across a handful of hand-picked reviews covering
    the phenomenon categories the pipeline is expected to handle."""
    samples = [
        # Clear positive
        "Honestly the best laptop I've ever owned. Battery lasts all day, the screen is gorgeous, and it boots in under five seconds. Worth every penny.",
        # Structured sarcasm (item 1: this one is known to fool both models)
        "This food supplement is amazing! If you are looking for constant stomach and headaches, with no benefits, this is the product for you! 10/10.",
        # Multipolar, two sentences — handled by the spaCy sentencer
        "Great materials and shape, very modern. Unfortunately, the balancing, which is the most important feature, is a complete mess.",
        # Multipolar, single sentence — exercises the clause fallback
        "Great materials but the balancing is a complete mess.",
        # Short blunt sarcasm — irony detector should fire and invert positive → negative
        "Oh fantastic, another pair of headphones that broke after a week. Truly the legendary build quality the ad promised.",
        # Explicit negation
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
