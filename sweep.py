"""Threshold sweep harness for the review sentiment pipeline.

Caches the expensive per-review model outputs once (full-review sentiment +
irony + per-segment sentiment) and replays the validator logic across varying
threshold settings, avoiding a re-run of the transformers per setting. Sweeps
each of the three locked thresholds one at a time:

* ``LOW_CONF_THRESHOLD``        0.40 .. 0.90 step 0.10
* ``IRONY_HIGH_CONF_THRESHOLD`` 0.40 .. 0.90 step 0.10
* ``MULTIPOLARITY_TOP_CLASS_THRESHOLD`` 0.30 .. 0.60 step 0.05

Writes per-threshold CSVs and a ``report.txt`` under ``--out-dir`` (default
``results/sweep/``), plus a JSON cache under ``cache/``. A parity self-check
asserts the replay reproduces ``pipeline.analyze`` at the locked values before
sweeping; a mismatch aborts the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from evaluate import (
    LABELS,
    PHENOMENON_CSV,
    load_phenomenon,
    sample_amazon,
)
from pipeline import (
    IRONY_HIGH_CONF_THRESHOLD,
    LOW_CONF_THRESHOLD,
    MULTIPOLARITY_TOP_CLASS_THRESHOLD,
    _clause_split,
    _predict_irony,
    _predict_sentiment,
    _score_segments,
    analyze,
    preprocess,
)

LOW_CONF_GRID: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
IRONY_HIGH_GRID: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MULTIPOLARITY_GRID: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def _build_entry(text: str) -> dict[str, Any]:
    """Run preprocessor + both models + segment scoring on a single review.

    Segments are always computed (same clause-split fallback as
    ``pipeline.analyze``) so the cache can replay any multipolarity threshold.
    """
    prep = preprocess(text)
    model_input = prep["model_input"]
    sentiment_raw = _predict_sentiment(model_input)
    irony_pred = _predict_irony(model_input)

    segments = prep["sentences"]
    if len(segments) <= 1:
        fallback = _clause_split(model_input)
        if len(fallback) > 1:
            segments = fallback
    segments_scored = _score_segments(segments) if segments else []

    return {
        "sentiment_raw": sentiment_raw,
        "irony": irony_pred,
        "segments": segments_scored,
    }


def build_cache(rows: list[dict[str, str]], label: str) -> list[dict[str, Any]]:
    """Run inference on every row and bundle predictions with metadata."""
    total = len(rows)
    print(f"  building {label} cache: {total} reviews ...")
    cache: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        entry = _build_entry(row["text"])
        entry["text"] = row["text"]
        entry["expected_label"] = row["expected_label"]
        if "category" in row:
            entry["category"] = row["category"]
        cache.append(entry)
        if i % 25 == 0 or i == total:
            print(f"    cached {i}/{total}")
    return cache


def save_cache(cache: list[dict[str, Any]], path: Path, meta: dict[str, Any]) -> None:
    """Write the cache entries plus their ``meta`` to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "entries": cache}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cache(path: Path, expected_meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return cached entries iff the on-disk meta matches ``expected_meta``."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("meta") != expected_meta:
        print(f"  cache at {path} has stale meta, rebuilding")
        return None
    return payload.get("entries")


def resolve(
    entry: dict[str, Any],
    low_conf: float,
    irony_high: float,
    multi_thr: float,
) -> dict[str, Any]:
    """Replay the validator branch logic given cached predictions.

    Mirrors the label-resolution code in ``pipeline.analyze`` 1:1; the parity
    self-check in ``_self_check`` enforces the invariant.
    """
    sentiment_raw = entry["sentiment_raw"]
    irony_pred = entry["irony"]
    segments = entry["segments"]

    multipolarity = sentiment_raw["confidence"] < multi_thr
    model_agreement = (
        irony_pred["label"] == "irony"
        and irony_pred["confidence"] >= irony_high
    )

    segments_out = segments if multipolarity else None

    if model_agreement:
        if sentiment_raw["label"] == "positive":
            if sentiment_raw["confidence"] >= low_conf:
                resolved_label = "negative"
            else:
                resolved_label = "neutral"
        else:
            resolved_label = sentiment_raw["label"]
        resolved_confidence = sentiment_raw["confidence"]
        if (
            segments_out
            and segments_out[-1]["label"] == resolved_label
        ):
            resolved_confidence = segments_out[-1]["confidence"]
    elif multipolarity:
        assert segments_out, "multipolarity flagged but no segments cached"
        last = segments_out[-1]
        resolved_label = last["label"]
        resolved_confidence = last["confidence"]
    else:
        resolved_label = sentiment_raw["label"]
        resolved_confidence = sentiment_raw["confidence"]

    low_confidence = resolved_confidence < low_conf

    return {
        "label": resolved_label,
        "confidence": resolved_confidence,
        "flags": {
            "low_confidence": low_confidence,
            "model_agreement": model_agreement,
            "multipolarity": multipolarity,
        },
    }


def _self_check(cache: list[dict[str, Any]], n_samples: int = 5) -> None:
    """Assert replay at locked thresholds equals ``analyze`` on a sample, else abort."""
    print(f"  parity self-check ({min(n_samples, len(cache))} samples) ...")
    for i, entry in enumerate(cache[:n_samples]):
        replay = resolve(
            entry,
            low_conf=LOW_CONF_THRESHOLD,
            irony_high=IRONY_HIGH_CONF_THRESHOLD,
            multi_thr=MULTIPOLARITY_TOP_CLASS_THRESHOLD,
        )
        live = analyze(entry["text"])
        problems: list[str] = []
        if replay["label"] != live["label"]:
            problems.append(f"label: replay={replay['label']} live={live['label']}")
        if not math.isclose(
            replay["confidence"], live["confidence"], rel_tol=1e-6, abs_tol=1e-9
        ):
            problems.append(
                f"confidence: replay={replay['confidence']} live={live['confidence']}"
            )
        for key in ("low_confidence", "model_agreement", "multipolarity"):
            if replay["flags"][key] != live["flags"][key]:
                problems.append(
                    f"flags.{key}: replay={replay['flags'][key]} live={live['flags'][key]}"
                )
        if problems:
            print(f"  parity FAIL on sample {i}: {entry['text'][:80]}")
            for p in problems:
                print(f"    - {p}")
            raise RuntimeError(
                "Replay does not match pipeline.analyze at locked thresholds. "
                "Sweep aborted to avoid producing misleading results."
            )
    print("  parity OK")


def _split_errors(
    y_true: list[str], y_pred: list[str], mask: list[bool]
) -> tuple[float, float, int, int]:
    """Return (in_mask_error_rate, out_mask_error_rate, n_in, n_out)."""
    in_correct = in_total = out_correct = out_total = 0
    for t, p, m in zip(y_true, y_pred, mask):
        if m:
            in_total += 1
            in_correct += t == p
        else:
            out_total += 1
            out_correct += t == p
    in_err = (1.0 - in_correct / in_total) if in_total else 0.0
    out_err = (1.0 - out_correct / out_total) if out_total else 0.0
    return in_err, out_err, in_total, out_total


def sweep_low_conf(
    cache: list[dict[str, Any]],
    irony_high: float,
    multi_thr: float,
) -> list[dict[str, Any]]:
    """Vary LOW_CONF; keep the other thresholds at locked values."""
    y_true = [e["expected_label"] for e in cache]
    rows: list[dict[str, Any]] = []
    for thr in LOW_CONF_GRID:
        resolved = [resolve(e, thr, irony_high, multi_thr) for e in cache]
        y_pred = [r["label"] for r in resolved]
        flagged_mask = [r["flags"]["low_confidence"] for r in resolved]
        flagged_err, unflagged_err, n_flagged, n_unflagged = _split_errors(
            y_true, y_pred, flagged_mask
        )
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )
        rows.append({
            "low_conf": thr,
            "n": len(cache),
            "n_flagged": n_flagged,
            "flagged_rate": n_flagged / len(cache),
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "macro_f1": macro_f1,
            "flagged_error_rate": flagged_err,
            "unflagged_error_rate": unflagged_err,
            "unflagged_n": n_unflagged,
        })
    return rows


def sweep_irony_high(
    cache: list[dict[str, Any]],
    low_conf: float,
    multi_thr: float,
) -> list[dict[str, Any]]:
    """Vary IRONY_HIGH; report engaged-branch size and accuracy on it."""
    y_true = [e["expected_label"] for e in cache]
    rows: list[dict[str, Any]] = []
    for thr in IRONY_HIGH_GRID:
        resolved = [resolve(e, low_conf, thr, multi_thr) for e in cache]
        y_pred = [r["label"] for r in resolved]
        engaged_mask = [r["flags"]["model_agreement"] for r in resolved]
        engaged_err, _, n_engaged, _ = _split_errors(y_true, y_pred, engaged_mask)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )
        rows.append({
            "irony_high": thr,
            "n": len(cache),
            "n_engaged": n_engaged,
            "engaged_rate": n_engaged / len(cache),
            "engaged_accuracy": (1.0 - engaged_err) if n_engaged else 0.0,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "macro_f1": macro_f1,
        })
    return rows


def sweep_multipolarity(
    cache: list[dict[str, Any]],
    low_conf: float,
    irony_high: float,
) -> list[dict[str, Any]]:
    """Vary MULTIPOLARITY; add a negative-ending multipolar breakdown."""
    y_true = [e["expected_label"] for e in cache]
    is_neg_multi = [
        e.get("category") == "multipolarity" and e["expected_label"] == "negative"
        for e in cache
    ]
    n_neg_multi_total = sum(is_neg_multi)

    rows: list[dict[str, Any]] = []
    for thr in MULTIPOLARITY_GRID:
        resolved = [resolve(e, low_conf, irony_high, thr) for e in cache]
        y_pred = [r["label"] for r in resolved]
        engaged_mask = [r["flags"]["multipolarity"] for r in resolved]
        engaged_err, _, n_engaged, _ = _split_errors(y_true, y_pred, engaged_mask)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )

        neg_multi_engaged = sum(
            1 for engaged, neg in zip(engaged_mask, is_neg_multi) if engaged and neg
        )
        neg_multi_correct = sum(
            1
            for pred, true, neg in zip(y_pred, y_true, is_neg_multi)
            if neg and pred == true
        )

        rows.append({
            "multipolarity": thr,
            "n": len(cache),
            "n_engaged": n_engaged,
            "engaged_rate": n_engaged / len(cache),
            "engaged_accuracy": (1.0 - engaged_err) if n_engaged else 0.0,
            "accuracy": acc,
            "error_rate": 1.0 - acc,
            "macro_f1": macro_f1,
            "neg_multi_total": n_neg_multi_total,
            "neg_multi_engaged": neg_multi_engaged,
            "neg_multi_correct": neg_multi_correct,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` (list of dicts) as a CSV at ``path``; no-op if empty."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_table(
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
) -> str:
    """Render a list of dicts as an aligned ASCII table.

    ``columns`` is a list of (key, header, format_string); the format string
    is applied via ``format(value, fmt)`` (empty fmt prints as-is).
    """
    widths = [max(len(hdr), 6) for _, hdr, _ in columns]
    for r in rows:
        for i, (k, _, fmt) in enumerate(columns):
            cell = format(r[k], fmt) if fmt else str(r[k])
            widths[i] = max(widths[i], len(cell))
    lines = [title, "-" * len(title)]
    header = "  ".join(f"{hdr:>{w}}" for (_, hdr, _), w in zip(columns, widths))
    lines.append(header)
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for (k, _, fmt), w in zip(columns, widths):
            cell = format(r[k], fmt) if fmt else str(r[k])
            cells.append(f"{cell:>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    dataset_label: str,
    locked: dict[str, float],
    low_conf_rows: list[dict[str, Any]],
    irony_rows: list[dict[str, Any]],
    multi_rows: list[dict[str, Any]],
    include_neg_multi: bool,
) -> None:
    """Write the three per-threshold CSVs and a combined ASCII report under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "low_conf.csv", low_conf_rows)
    _write_csv(out_dir / "irony_high.csv", irony_rows)
    _write_csv(out_dir / "multipolarity.csv", multi_rows)

    sections: list[str] = []
    sections.append(f"Threshold sweep — {dataset_label}")
    sections.append("=" * (len(dataset_label) + 19))
    sections.append("")
    sections.append(
        f"Locked thresholds: LOW_CONF={locked['low_conf']:.2f}  "
        f"IRONY_HIGH={locked['irony_high']:.2f}  "
        f"MULTIPOLARITY={locked['multi']:.2f}"
    )
    sections.append("(Each sweep varies one threshold; the other two stay locked.)")
    sections.append("")

    sections.append(_fmt_table(
        "LOW_CONF sweep (gates low_confidence flag + irony invert/downgrade)",
        low_conf_rows,
        [
            ("low_conf",            "LOW_CONF",   ".2f"),
            ("n_flagged",           "flagged",    "d"),
            ("flagged_rate",        "flag_rate",  ".3f"),
            ("error_rate",          "err",        ".3f"),
            ("flagged_error_rate",  "err|flag",   ".3f"),
            ("unflagged_error_rate","err|unflag", ".3f"),
            ("accuracy",            "acc",        ".3f"),
            ("macro_f1",            "mF1",        ".3f"),
        ],
    ))
    sections.append("")

    sections.append(_fmt_table(
        "IRONY_HIGH sweep (gates the irony-correction branch)",
        irony_rows,
        [
            ("irony_high",          "IRONY_HIGH", ".2f"),
            ("n_engaged",           "engaged",    "d"),
            ("engaged_rate",        "eng_rate",   ".3f"),
            ("engaged_accuracy",    "acc|eng",    ".3f"),
            ("error_rate",          "err",        ".3f"),
            ("accuracy",            "acc",        ".3f"),
            ("macro_f1",            "mF1",        ".3f"),
        ],
    ))
    sections.append("")

    multi_cols = [
        ("multipolarity",       "MULTI",     ".2f"),
        ("n_engaged",           "engaged",   "d"),
        ("engaged_rate",        "eng_rate",  ".3f"),
        ("engaged_accuracy",    "acc|eng",   ".3f"),
        ("error_rate",          "err",       ".3f"),
        ("accuracy",            "acc",       ".3f"),
        ("macro_f1",            "mF1",       ".3f"),
    ]
    if include_neg_multi:
        multi_cols.extend([
            ("neg_multi_engaged",   "neg_eng/N",  "d"),
            ("neg_multi_correct",   "neg_ok/N",   "d"),
        ])
    sections.append(_fmt_table(
        "MULTIPOLARITY_TOP_CLASS sweep (gates the segment + last-clause branch)",
        multi_rows,
        multi_cols,
    ))
    if include_neg_multi and multi_rows:
        total = multi_rows[0]["neg_multi_total"]
        sections.append(
            f"\nneg_eng/N = engaged-branch count out of {total} negative-ending "
            "multipolar phenomenon rows;\n"
            f"neg_ok/N  = correctly-labeled count out of those {total} rows."
        )

    (out_dir / "report.txt").write_text("\n".join(sections) + "\n", encoding="utf-8")


def _ensure_cache_phenomenon(
    cache_path: Path, csv_path: Path, rebuild: bool
) -> list[dict[str, Any]]:
    """Load the phenomenon cache if its meta matches, else rebuild and save it."""
    meta = {"dataset": "phenomenon", "csv": str(csv_path)}
    if not rebuild:
        cached = load_cache(cache_path, meta)
        if cached is not None:
            print(f"  loaded phenomenon cache ({len(cached)} entries) from {cache_path}")
            return cached
    rows = load_phenomenon(csv_path)
    cache = build_cache(rows, "phenomenon")
    save_cache(cache, cache_path, meta)
    print(f"  saved phenomenon cache -> {cache_path}")
    return cache


def _ensure_cache_amazon(
    cache_path: Path,
    sample_size: int,
    category: str,
    seed: int,
    rebuild: bool,
) -> list[dict[str, Any]]:
    """Load the Amazon cache if its meta matches, else rebuild and save it."""
    meta = {
        "dataset": "amazon",
        "sample_size": sample_size,
        "category": category,
        "seed": seed,
    }
    if not rebuild:
        cached = load_cache(cache_path, meta)
        if cached is not None:
            print(f"  loaded amazon cache ({len(cached)} entries) from {cache_path}")
            return cached
    rows = sample_amazon(sample_size, category, seed)
    cache = build_cache(rows, "amazon")
    save_cache(cache, cache_path, meta)
    print(f"  saved amazon cache -> {cache_path}")
    return cache


def run_dataset(
    cache: list[dict[str, Any]],
    out_dir: Path,
    dataset_label: str,
    include_neg_multi: bool,
) -> None:
    """Self-check, run the three sweeps, and write the report for one dataset."""
    print(f"\n== Sweeping {dataset_label} ({len(cache)} reviews) ==")
    _self_check(cache)

    locked = {
        "low_conf": LOW_CONF_THRESHOLD,
        "irony_high": IRONY_HIGH_CONF_THRESHOLD,
        "multi": MULTIPOLARITY_TOP_CLASS_THRESHOLD,
    }
    low_conf_rows = sweep_low_conf(
        cache, irony_high=locked["irony_high"], multi_thr=locked["multi"]
    )
    irony_rows = sweep_irony_high(
        cache, low_conf=locked["low_conf"], multi_thr=locked["multi"]
    )
    multi_rows = sweep_multipolarity(
        cache, low_conf=locked["low_conf"], irony_high=locked["irony_high"]
    )

    write_report(
        out_dir, dataset_label, locked,
        low_conf_rows, irony_rows, multi_rows,
        include_neg_multi=include_neg_multi,
    )
    print(f"  -> results saved to {out_dir}")
    print()
    print((out_dir / "report.txt").read_text(encoding="utf-8"))


def main() -> None:
    """Parse CLI flags and run the requested threshold sweep(s)."""
    parser = argparse.ArgumentParser(
        description="Threshold sweep harness for the sentiment pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon", "phenomenon", "all"],
        default="all",
        help="Which dataset(s) to sweep (default: all).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Amazon stratified sample size (default: 500, matches evaluate.py).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the Amazon stratified shuffle (default: 42).",
    )
    parser.add_argument(
        "--amazon-category", default="Electronics",
        help="Amazon Reviews 2023 product category (default: Electronics).",
    )
    parser.add_argument(
        "--phenomenon-csv", type=Path, default=PHENOMENON_CSV,
        help=f"Phenomenon CSV path (default: {PHENOMENON_CSV}).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/sweep"),
        help="Output directory (default: results/sweep/).",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Ignore any existing inference cache and re-run the models.",
    )
    args = parser.parse_args()

    cache_dir = args.out_dir / "cache"

    if args.dataset in ("phenomenon", "all"):
        cache = _ensure_cache_phenomenon(
            cache_dir / "phenomenon.json", args.phenomenon_csv, args.rebuild_cache
        )
        run_dataset(
            cache, args.out_dir / "phenomenon",
            dataset_label="phenomenon", include_neg_multi=True,
        )

    if args.dataset in ("amazon", "all"):
        cache = _ensure_cache_amazon(
            cache_dir / "amazon.json",
            args.sample_size, args.amazon_category, args.seed,
            args.rebuild_cache,
        )
        run_dataset(
            cache, args.out_dir / "amazon",
            dataset_label=f"amazon / {args.amazon_category}",
            include_neg_multi=False,
        )


if __name__ == "__main__":
    main()
