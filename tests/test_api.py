"""Integration tests for the FastAPI ``/analyze`` and ``/health`` endpoints.

Runs the real pipeline end-to-end via ``TestClient``: the lifespan warms the
actual Cardiff + spaCy models, so the first request is slow and the suite needs
those models available. The prediction log is redirected to a temporary SQLite
file so the real ``data/predictions.db`` is left untouched.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api
import db


@pytest.fixture(scope="module")
def client_and_db(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[TestClient, Path]]:
    """Boot the app once (module-scoped warmup) with the log pointed at a temp DB."""
    db_file = tmp_path_factory.mktemp("data") / "predictions.db"
    original = db.DB_PATH
    db.DB_PATH = db_file
    try:
        with TestClient(api.app) as client:
            yield client, db_file
    finally:
        db.DB_PATH = original


def test_health_reports_models_warm(client_and_db: tuple[TestClient, Path]) -> None:
    """Once the lifespan has run, /health reports the models as warm."""
    client, _ = client_and_db
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["models_warm"] is True


_KNOWN_POLARITY = [
    (
        "Absolutely fantastic product. It works flawlessly, exceeded my "
        "expectations, and I would happily buy it again.",
        "positive",
    ),
    (
        "Terrible experience. It broke on the first day, the support team was "
        "useless, and it was a complete waste of money.",
        "negative",
    ),
    (
        "Excellent quality and great value. Reliable, well built, and a "
        "pleasure to use every single day.",
        "positive",
    ),
]


@pytest.mark.parametrize("text, expected_label", _KNOWN_POLARITY)
def test_analyze_known_polarity(
    client_and_db: tuple[TestClient, Path], text: str, expected_label: str
) -> None:
    """/analyze returns the expected label and a well-formed body for clear reviews."""
    client, _ = client_and_db
    resp = client.post("/analyze", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == expected_label
    assert body["text"] == text
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["flags"]) == {"low_confidence", "model_agreement", "multipolarity"}


def test_analyze_logs_one_row_per_request_and_skips_warmup(
    client_and_db: tuple[TestClient, Path]
) -> None:
    """Each successful /analyze persists exactly one row; the warmup call is not logged."""
    client, db_file = client_and_db

    def row_count() -> int:
        with sqlite3.connect(db_file) as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    before = row_count()
    for text, _ in _KNOWN_POLARITY:
        assert client.post("/analyze", json={"text": text}).status_code == 200
    assert row_count() - before == len(_KNOWN_POLARITY)

    with sqlite3.connect(db_file) as conn:
        warmups = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE input_text = 'warmup'"
        ).fetchone()[0]
    assert warmups == 0


def test_analyze_rejects_empty_text(client_and_db: tuple[TestClient, Path]) -> None:
    """Empty text violates the request model's min_length and is rejected."""
    client, _ = client_and_db
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422


def test_analyze_rejects_missing_body(client_and_db: tuple[TestClient, Path]) -> None:
    """A payload without the required text field is rejected."""
    client, _ = client_and_db
    resp = client.post("/analyze", json={})
    assert resp.status_code == 422
