"""FastAPI wrapper around the sentiment analysis pipeline.

Exposes two endpoints:

* ``POST /analyze`` — runs :func:`pipeline.analyze` on a single review and
  returns the structured result documented in that function's docstring.
* ``GET /health`` — liveness probe; also reports whether the underlying
  models have finished warming up.

Pydantic models mirror the pipeline's output schema so the auto-generated
OpenAPI docs at ``/docs`` and ``/redoc`` are self-describing.

Run with ``make run`` or ``uvicorn api:app --reload``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pipeline import Label, analyze

# --- Pydantic schemas ------------------------------------------------------
# Schemas mirror the dict returned by :func:`pipeline.analyze` one-to-one so
# the OpenAPI surface and the in-process Python surface stay in lockstep.

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
    """Boolean validation flags raised by the pipeline.

    Mirrors :func:`pipeline.analyze`'s ``flags`` block; see that function's
    docstring for the precise semantics and branch precedence rules.
    """

    low_confidence: bool
    model_agreement: bool
    multipolarity: bool


class SentimentDistribution(BaseModel):
    """Full 3-class softmax over the sentiment model."""

    negative: float
    neutral: float
    positive: float


class SentimentRaw(BaseModel):
    """Pre-correction sentiment-model output.

    Useful when ``flags.model_agreement`` is True and the top-level
    ``label`` may have been overridden by the irony branch.
    """

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
    """Response model for ``POST /analyze``.

    Identical in shape to :func:`pipeline.analyze`'s return value.
    """

    text: str
    preprocessed_text: str
    label: Label
    confidence: float
    flags: Flags
    sentiment_raw: SentimentRaw
    irony: IronyPrediction
    segments: list[Segment] | None


class HealthResponse(BaseModel):
    """Response model for ``GET /health``."""

    status: Literal["ok"]
    models_warm: bool = Field(
        ...,
        description=(
            "True once both Cardiff models and the spaCy pipeline have been "
            "loaded; flipped during the FastAPI startup lifespan."
        ),
    )


# --- App setup -------------------------------------------------------------

_models_warm = False


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Eagerly warm the pipeline so the first real request isn't slow.

    Calling :func:`pipeline.analyze` once at startup forces the three
    lazy ``functools.cache`` singletons in ``pipeline.py`` (sentiment,
    irony, spaCy) to load before the server starts accepting traffic.
    """
    global _models_warm
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


# --- Endpoints -------------------------------------------------------------

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a single customer review.",
)
def analyze_endpoint(payload: AnalyzeRequest) -> dict:
    """Run the pipeline on ``payload.text`` and return the full result.

    Declared as a sync handler on purpose: HuggingFace inference is
    blocking, and FastAPI dispatches sync endpoints to a threadpool, so
    the event loop is not stalled while a request is in flight.
    """
    return analyze(payload.text)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe.",
)
def health_endpoint() -> dict:
    """Return service status and whether the models have been warmed."""
    return {"status": "ok", "models_warm": _models_warm}
