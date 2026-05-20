"""Streamlit UI on top of the FastAPI sentiment analysis endpoint.

Two tabs:

* **Single review** — free-text area; submitting calls ``POST /analyze`` and
  renders the resolved label, confidence, flags, per-segment breakdown, and
  raw model provenance.
* **Batch CSV** — upload a CSV with a ``text`` column (other columns
  ignored); each row is sent through ``/analyze``; results are shown as a
  flat table and offered as a CSV download.

The UI talks to the API over HTTP — configure with the ``API_URL`` env var
(default ``http://localhost:8000``). Start the API with ``make run`` and
the UI with ``make ui`` (or ``streamlit run ui.py``).
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

# --- Config ----------------------------------------------------------------

API_URL: str = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ANALYZE_ENDPOINT: str = f"{API_URL}/analyze"
HEALTH_ENDPOINT: str = f"{API_URL}/health"

# Generous timeout: covers the worst-case first request when the API is
# still warming. After lifespan completes, real latency is sub-second.
REQUEST_TIMEOUT_SECONDS: float = 30.0

# Display colors for the resolved 3-class label.
LABEL_COLOR: dict[str, str] = {
    "positive": "#1f9e3a",
    "neutral": "#6b7280",
    "negative": "#c41e3a",
}


# --- HTTP wrappers ---------------------------------------------------------

def analyze_review(text: str) -> dict[str, Any]:
    """Call ``POST /analyze`` and return the parsed JSON body.

    Raises ``requests.RequestException`` on connection failure or non-2xx
    response.
    """
    response = requests.post(
        ANALYZE_ENDPOINT,
        json={"text": text},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def check_health() -> tuple[bool, str]:
    """Probe ``GET /health``; return ``(reachable, status_string)``."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
    except requests.RequestException as exc:
        return False, f"unreachable ({type(exc).__name__})"
    if not response.ok:
        return False, f"HTTP {response.status_code}"
    warm = bool(response.json().get("models_warm", False))
    return True, "models warm" if warm else "models warming"


# --- Rendering helpers -----------------------------------------------------

def _render_flags(flags: dict[str, bool]) -> None:
    """Render the boolean flag block as Streamlit colored-text markdown."""
    parts: list[str] = []
    if flags["low_confidence"]:
        parts.append(":orange[**low confidence**]")
    if flags["model_agreement"]:
        parts.append(":red[**irony detected**]")
    if flags["multipolarity"]:
        parts.append(":blue[**multipolar**]")
    if parts:
        st.markdown(" &nbsp;·&nbsp; ".join(parts))
    else:
        st.caption("no flags raised")


def render_result(result: dict[str, Any]) -> None:
    """Render an ``/analyze`` response into the single-review results panel."""
    label = result["label"]
    color = LABEL_COLOR.get(label, "#000")
    confidence = float(result["confidence"])

    st.markdown(
        f"<h2 style='color:{color};margin:0 0 0.25em 0'>"
        f"{label.upper()}"
        f" <span style='font-size:0.6em;color:#444'>({confidence:.1%})</span>"
        f"</h2>",
        unsafe_allow_html=True,
    )
    _render_flags(result["flags"])

    with st.expander("Preprocessed text (normalized + negation markup)"):
        st.text(result["preprocessed_text"])

    if result.get("segments"):
        st.subheader("Per-segment breakdown")
        seg_df = pd.DataFrame(result["segments"])[["text", "label", "confidence"]]
        seg_df["confidence"] = seg_df["confidence"].map(lambda c: f"{c:.1%}")
        st.dataframe(seg_df, hide_index=True, use_container_width=True)

    with st.expander("Model provenance"):
        sentiment_raw = result["sentiment_raw"]
        st.markdown(
            f"**Raw sentiment** (pre-correction) — `{sentiment_raw['label']}` "
            f"at {float(sentiment_raw['confidence']):.1%}"
        )
        dist_df = pd.DataFrame(
            {"probability": sentiment_raw["distribution"]},
        ).reindex(["negative", "neutral", "positive"])
        st.bar_chart(dist_df, horizontal=True)

        irony = result["irony"]
        st.markdown(
            f"**Irony detector** — `{irony['label']}` at "
            f"{float(irony['confidence']):.1%}"
        )


def flatten_for_csv(result: dict[str, Any]) -> dict[str, Any]:
    """Project an ``/analyze`` response onto a single row for the batch table."""
    return {
        "text": result["text"],
        "label": result["label"],
        "confidence": result["confidence"],
        "flag_low_confidence": result["flags"]["low_confidence"],
        "flag_model_agreement": result["flags"]["model_agreement"],
        "flag_multipolarity": result["flags"]["multipolarity"],
        "sentiment_raw_label": result["sentiment_raw"]["label"],
        "sentiment_raw_confidence": result["sentiment_raw"]["confidence"],
        "irony_label": result["irony"]["label"],
        "irony_confidence": result["irony"]["confidence"],
        "n_segments": len(result["segments"]) if result["segments"] else 0,
    }


# --- Layout ----------------------------------------------------------------

st.set_page_config(page_title="Review Sentiment Analysis", layout="wide")
st.title("Customer Review Sentiment Analysis")
st.caption(
    "Two-stage analysis: sentiment + irony, with multipolarity segmentation. "
    "See the API at `/docs` for the full response schema."
)

with st.sidebar:
    st.subheader("API")
    st.caption(f"`{API_URL}`")
    reachable, status = check_health()
    if reachable:
        st.success(f"reachable — {status}")
    else:
        st.error(f"unreachable — {status}")
        st.caption("Start the API with `make run`.")

single_tab, batch_tab = st.tabs(["Single review", "Batch CSV"])

# --- Single review tab -----------------------------------------------------

with single_tab:
    with st.form("single_review_form", clear_on_submit=False):
        review_text = st.text_area(
            "Review text",
            height=160,
            placeholder="Paste a customer review here...",
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not review_text.strip():
            st.warning("Please enter a review.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    st.session_state["single_result"] = analyze_review(review_text)
                st.session_state["single_error"] = None
            except requests.RequestException as exc:
                st.session_state["single_error"] = str(exc)
                st.session_state["single_result"] = None

    if st.session_state.get("single_error"):
        st.error(f"API call failed: {st.session_state['single_error']}")
    elif st.session_state.get("single_result"):
        render_result(st.session_state["single_result"])

# --- Batch CSV tab ---------------------------------------------------------

with batch_tab:
    st.write(
        "Upload a CSV with a `text` column. Other columns are ignored. "
        "Empty rows are skipped."
    )
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            input_df = None

        if input_df is not None:
            if "text" not in input_df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                texts = (
                    input_df["text"]
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                    .loc[lambda s: s.str.len() > 0]
                    .tolist()
                )
                st.info(f"{len(texts)} non-empty review(s) ready to analyze.")

                if texts and st.button("Run batch", type="primary"):
                    progress = st.progress(0.0, text="Analyzing...")
                    rows: list[dict[str, Any]] = []
                    failed = 0
                    for index, text in enumerate(texts, start=1):
                        try:
                            rows.append(flatten_for_csv(analyze_review(text)))
                        except requests.RequestException as exc:
                            failed += 1
                            rows.append(
                                {"text": text, "label": "ERROR", "error": str(exc)}
                            )
                        progress.progress(
                            index / len(texts),
                            text=f"Analyzing... {index}/{len(texts)}",
                        )
                    progress.empty()
                    st.session_state["batch_results"] = pd.DataFrame(rows)
                    st.session_state["batch_failed"] = failed

    results_df = st.session_state.get("batch_results")
    if results_df is not None:
        failed = st.session_state.get("batch_failed", 0)
        if failed:
            st.warning(f"{failed} request(s) failed — see the `error` column.")
        st.dataframe(results_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )
