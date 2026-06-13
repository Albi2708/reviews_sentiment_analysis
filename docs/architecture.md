# Architecture

## Purpose and scope

The system classifies customer reviews into three sentiment classes — **positive**,
**neutral**, **negative** — and explicitly handles the linguistic phenomena that distort
surface-level sentiment: **irony, sarcasm, negation, and multipolarity**. Every prediction
carries a confidence score and a set of validation flags, so its reliability can be assessed
both per-prediction and across the system as a whole.

This document describes the system's structure, the request lifecycle, the validation logic,
the output contract, and the deployment model. The major design choices are recorded
separately as Architecture Decision Records (ADRs) under [`adr/`](adr/) and linked at the end.

## Architectural principles

The system is split into three layers so that each can evolve independently:

- **Presentation** — a thin Streamlit UI that holds no analysis logic.
- **Service** — a stateless FastAPI application exposing a small REST contract.
- **Model** — the analysis pipeline: preprocessing, two transformer models, and a rule-based
  validator.

The defining consequence of this split is **swappability**: the sentiment model can be replaced
with a fine-tuned variant without any change to the REST contract or the UI. The service layer
is stateless; the only persistent state is an append-only prediction log, which sits off the
critical path.

## Component overview

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                     Service (FastAPI)                      │
   Browser       │                                                           │
     │           │   POST /analyze ─┐                                        │
     ▼           │   GET  /health   │                                        │
┌──────────┐  HTTP│                  ▼                                        │
│ Streamlit │────▶│            ┌───────────┐   ┌──────────────────────────┐  │
│    UI     │ JSON│            │  Pipeline │──▶│ Preprocessor (ftfy+spaCy) │  │
│  (ui.py)  │◀────│            │(pipeline. │   ├──────────────────────────┤  │
└──────────┘      │            │   py)     │──▶│ Sentiment model (Cardiff) │  │
                  │            │           │──▶│ Irony model (Cardiff)     │  │
                  │            │           │──▶│ Validator (rules)         │  │
                  │            └─────┬─────┘   └──────────────────────────┘  │
                  │                  │                                        │
                  │                  ▼                                        │
                  │            ┌───────────┐                                  │
                  │            │  SQLite   │  (db.py — best-effort log)       │
                  │            └───────────┘                                  │
                  └──────────────────────────────────────────────────────────┘

   Evaluation harness (evaluate.py, sweep.py) calls the pipeline directly, offline.
```

| Component | Module | Responsibility |
| --- | --- | --- |
| UI | `ui.py` | Single-review and batch-CSV front end; calls the API over HTTP. Holds no model logic. |
| Service | `api.py` | `POST /analyze`, `GET /health`; Pydantic request/response models; warms the pipeline at startup; logs each prediction. |
| Pipeline | `pipeline.py` | Orchestrates preprocessing → both models → validator; exposes `analyze(text)`. |
| Preprocessor | `pipeline.py` | Unicode normalization (ftfy), sentence segmentation and negation marking (spaCy). |
| Sentiment model | `pipeline.py` | `cardiffnlp/twitter-roberta-base-sentiment-latest`; three-class probability distribution. |
| Irony model | `pipeline.py` | `cardiffnlp/twitter-roberta-base-irony`; binary irony / non-irony. |
| Validator | `pipeline.py` | Threshold-based flags, irony correction, multipolarity segmentation, label resolution. |
| Storage | `db.py` | SQLite prediction log (one row per `/analyze` call). |
| Evaluation | `evaluate.py`, `sweep.py` | Offline metrics on a labeled benchmark and a phenomenon test set; threshold tuning. |

## Request lifecycle

A single call to `analyze(text)` proceeds as follows:

1. **Preprocess.** `ftfy` repairs mojibake and normalizes Unicode; the text is stripped. spaCy
   segments the text into sentences and marks negation scopes as `[NEG: ...]` for display. The
   models receive the **unannotated** text — negation marking is for transparency only, since
   the transformer has already learned negation patterns implicitly.
2. **Sentiment.** The sentiment model returns a three-class distribution over
   negative / neutral / positive; the top class becomes the raw label and its probability the
   raw confidence.
3. **Irony.** The irony model runs independently and returns irony / non-irony with a
   confidence.
4. **Validate and resolve.** Two flags are computed, a per-segment view is built when needed,
   and a single aggregate label is resolved (see below).
5. **Return.** A result dictionary is assembled (see *Output contract*). The API layer persists
   one log row and returns the dictionary as JSON.

## Validation logic

Validation operates at three layers — per-prediction (flags and confidence), system-level
(benchmark metrics), and phenomenon-specific (a curated edge-case set). The per-prediction
logic lives in the validator and is driven by three thresholds, tuned on the evaluation sets
and locked at:

| Threshold | Value | Role |
| --- | --- | --- |
| `LOW_CONF_THRESHOLD` | 0.70 | Below this resolved confidence, the prediction is flagged low-confidence. Also gates the irony correction (see below). |
| `IRONY_HIGH_CONF_THRESHOLD` | 0.50 | The irony detector must reach this confidence (and label the text *irony*) for the model-agreement flag to fire. |
| `MULTIPOLARITY_TOP_CLASS_THRESHOLD` | 0.60 | If the sentiment top-class probability is below this, the distribution is treated as near-uniform and the multipolarity flag fires. |

**Flags.**

- `multipolarity` — fires when the sentiment top-class probability `< 0.60`. A low top-class
  probability is used as a tractable proxy for the "near-uniform distribution" that signals a
  review spanning multiple polarities.
- `model_agreement` — fires when the irony detector labels the text *irony* with confidence
  `≥ 0.50`, i.e. the irony detector contradicts a straight reading of the sentiment.
- `low_confidence` — fires when the **resolved** confidence (after any correction) is `< 0.70`.

**Per-segment view.** Whenever `multipolarity` fires, the review is split into segments and each
is scored independently. Segmentation uses spaCy's sentence boundaries; if that yields a single
segment, a clause-level fallback splits on commas, semicolons, and contrastive connectives
(*but, however, although, unfortunately, yet*). The per-segment scores are always reported for
transparency, even when they do not drive the aggregate label.

**Label resolution (branch precedence).** The aggregate label is resolved in this order:

1. **Irony correction takes precedence.** If `model_agreement` fired and the raw sentiment is
   *positive*, the label is **inverted to negative** when the model is confident (`≥ 0.70`),
   otherwise **downgraded to neutral**. If `model_agreement` fired but the raw sentiment is
   already negative or neutral, the label is kept (no correction needed).
2. **Otherwise, if `multipolarity` fired**, the aggregate follows the **final segment**
   ("last-clause wins"). This matches how readers interpret reviews such as *"X is great, but Y
   is broken"* — the closing clause dominates the takeaway (a recency effect).
3. **Otherwise**, the raw sentiment label stands.

**Confidence convention.** The resolved confidence normally carries the sentiment model's raw
top-class probability. When a correction or aggregation routes through a branch and the final
segment's label matches the resolved label, that segment's in-context confidence is reported
instead — keeping the displayed confidence aligned with the evidence that actually decided the
label.

## Output contract

`analyze()` returns, and the API mirrors 1:1 as `AnalyzeResponse`, the following structure:

```jsonc
{
  "text": "...",                 // original input
  "preprocessed_text": "...",    // normalized text with [NEG: ...] markup
  "label": "negative",           // resolved aggregate label
  "confidence": 0.83,            // resolved confidence
  "flags": {
    "low_confidence": false,
    "model_agreement": false,    // irony detector contradicts sentiment
    "multipolarity": false
  },
  "sentiment_raw": {             // pre-correction sentiment output (provenance)
    "label": "negative",
    "confidence": 0.83,
    "distribution": { "negative": 0.83, "neutral": 0.10, "positive": 0.07 }
  },
  "irony": { "label": "non_irony", "confidence": 0.95 },
  "segments": null               // list of {text, label, confidence} when multipolarity fires
}
```

`sentiment_raw` and `irony` are always present so a consumer can see the original model outputs
behind any correction. `segments` is `null` unless the multipolarity branch built a per-segment
view.

## Technology stack

| Concern | Tool | Rationale |
| --- | --- | --- |
| UI / demo front end | Streamlit | Fast prototyping of an interactive ML front end. |
| Service layer | FastAPI | Type-safe REST contract with auto-generated OpenAPI docs. |
| Sentiment & irony engine | Hugging Face Transformers + PyTorch | Industry-standard runtime; provides the Cardiff models off the shelf. |
| Preprocessing | spaCy + ftfy | Sentence segmentation, negation-scope detection, Unicode normalization. |
| Storage / logging | SQLite | Lightweight, zero-configuration request and prediction logging. |
| Evaluation | scikit-learn + pandas + matplotlib | Standard metrics, confusion matrices, phenomenon reports. |
| Packaging | Docker | Reproducible, one-command deployment of the full stack. |

## Deployment

The full stack runs with a single command (`docker compose up --build`). One image is built and
run as two services: the **API** (`uvicorn` on port 8000) and the **UI** (`streamlit` on port
8501, configured to reach the API over the internal network). The two Cardiff models and the
spaCy model are pre-cached into the image at build time so the first request is fast; a
CPU-only build of PyTorch keeps the image lean. The API exposes a health check that reports
readiness once the models are warm, and the UI waits on it before accepting traffic. The
prediction log is kept on a named volume so it survives container restarts. A local
virtual-environment path is also supported for development.

## Evaluation and known limitations

The pipeline is benchmarked on two sets: a labeled public dataset (Amazon Reviews 2023,
Electronics; n = 498 stratified across the three classes) and a curated phenomenon set
(n = 45, covering irony, sarcasm, negation, and multipolarity). Headline results on the locked
configuration are accuracy 0.568 / macro-F1 0.542 on the benchmark and accuracy 0.667 on the
phenomenon set (sarcasm handled at 10/10). Full per-class and per-category breakdowns live under
`results/`.

Two limitations are accepted and documented:

- **Neutral-class collapse on long-form reviews.** Neutral recall on the benchmark is ~0.25;
  the sentiment model is trained on short, informal text and rarely emits a neutral label on
  review-length prose. This is a distribution mismatch in the off-the-shelf model rather than a
  threshold artifact, so it cannot be tuned away. See
  [ADR 0002](adr/0002-off-the-shelf-pretrained-models.md).
- **Confidence-gated multipolarity branch.** The multipolarity branch engages only when the
  sentiment model is uncertain, so it cannot correct reviews where the model is *confidently
  wrong*. See [ADR 0003](adr/0003-two-specialized-models.md).

## Decision records

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-three-class-sentiment-scheme.md) | Three-class sentiment scheme (positive / neutral / negative). |
| [0002](adr/0002-off-the-shelf-pretrained-models.md) | Use the Cardiff models off the shelf, without fine-tuning. |
| [0003](adr/0003-two-specialized-models.md) | Run two specialized models (sentiment + irony) in parallel. |
| [0004](adr/0004-fastapi-service-layer.md) | FastAPI as the stateless service layer. |
| [0005](adr/0005-sqlite-prediction-logging.md) | SQLite for prediction logging. |
