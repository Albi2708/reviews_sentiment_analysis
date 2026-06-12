"""FastAPI wrapper around the sentiment analysis pipeline.

Exposes ``POST /analyze`` (runs :func:`pipeline.analyze` on one review) and
``GET /health`` (liveness + model-warmup status). Pydantic models mirror the
pipeline's output schema so the OpenAPI docs at ``/docs`` are self-describing.
Run with ``make run`` or ``uvicorn api:app --reload``.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

import db
from pipeline import Label, analyze

IronyLabel = Literal["irony", "non_irony"]


class AnalyzeRequest(BaseModel):
    """Payload for ``POST /analyze``."""

    text: str = Field(
        ...,
        min_length=1,
        description="The raw review text to analyze.",
        examples=["Great materials but the balancing is a complete mess."],
    )


class Flags(BaseModel):
    """Boolean validation flags; see :func:`pipeline.analyze` for semantics."""

    low_confidence: bool
    model_agreement: bool
    multipolarity: bool


class SentimentDistribution(BaseModel):
    """Full 3-class softmax over the sentiment model."""

    negative: float
    neutral: float
    positive: float


class SentimentRaw(BaseModel):
    """Pre-correction sentiment-model output (the label before any irony override)."""

    label: Label
    confidence: float
    distribution: SentimentDistribution


class IronyPrediction(BaseModel):
    """Irony detector's raw output."""

    label: IronyLabel
    confidence: float


class Segment(BaseModel):
    """Per-segment sentiment; populated when ``flags.multipolarity`` fires."""

    text: str
    label: Label
    confidence: float


class AnalyzeResponse(BaseModel):
    """Response for ``POST /analyze``; mirrors :func:`pipeline.analyze`'s return value."""

    text: str
    preprocessed_text: str
    label: Label
    confidence: float
    flags: Flags
    sentiment_raw: SentimentRaw
    irony: IronyPrediction
    segments: list[Segment] | None


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: Literal["ok"]
    models_warm: bool = Field(
        ...,
        description=(
            "True once both Cardiff models and the spaCy pipeline have been "
            "loaded; flipped during the FastAPI startup lifespan."
        ),
    )


_models_warm = False


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create the prediction-log schema, then warm the pipeline.

    ``db.init_schema()`` runs first so a broken DB fails startup loudly. The
    warmup ``analyze("warmup")`` forces the cached model singletons to load
    before traffic arrives; that call is not logged.
    """
    global _models_warm
    db.init_schema()
    analyze("warmup")
    _models_warm = True
    yield


app = FastAPI(
    title="Customer Review Sentiment Analysis",
    description=(
        "Two-stage sentiment + irony analysis over customer reviews. "
        "See the pipeline source for branch precedence and threshold "
        "rationale."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a single customer review.",
)
def analyze_endpoint(payload: AnalyzeRequest) -> dict:
    """Run the pipeline on ``payload.text`` and return the full result.

    Sync handler on purpose: HuggingFace inference is blocking and FastAPI
    dispatches sync endpoints to a threadpool. Persists one row via
    :func:`db.log_prediction` (latency is the pipeline-only ``analyze()``
    duration in ms); logging failures do not propagate to the client.
    """
    start = time.perf_counter()
    result = analyze(payload.text)
    latency_ms = (time.perf_counter() - start) * 1000.0
    db.log_prediction(result, latency_ms)
    return result


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe.",
)
def health_endpoint() -> dict:
    """Return service status and whether the models have been warmed."""
    return {"status": "ok", "models_warm": _models_warm}
