"""Evaluation harness for the review sentiment pipeline.

Runs :func:`pipeline.analyze` against two test sets and writes accuracy,
macro-F1, per-class precision/recall/F1, and a confusion matrix for each.

Test sets (roadmap items 2 and 3):

* **Amazon Reviews 2023** (``McAuley-Lab/Amazon-Reviews-2023``) — the
  ``raw_review_Electronics`` config, streamed and stratified to a fixed
  3-class sample. Star ratings are mapped 1–2 → negative, 3 → neutral,
  4–5 → positive.
* **Phenomenon test set** — the hand-curated CSV at
  ``tests/fixtures/phenomenon_reviews.csv`` (irony, sarcasm, negation,
  multipolarity). The phenomenon report includes a per-category breakdown
  since that's what the set was built to surface.

Output layout (under ``--out-dir``, default ``results/``):

    results/
      amazon/
        metrics.json
        confusion_matrix.csv
        report.txt
      phenomenon/
        metrics.json
        confusion_matrix.csv
        report.txt

Run with ``make eval`` or ``python evaluate.py``. The script calls
``pipeline.analyze`` directly, so SQLite logging in ``db.py`` (wired only
into the FastAPI app) is not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from pipeline import analyze

LABELS: list[str] = ["negative", "neutral", "positive"]

AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
PHENOMENON_CSV = Path("tests/fixtures/phenomenon_reviews.csv")


# --- Label mapping ---------------------------------------------------------

def stars_to_label(rating: float) -> str:
    """Map an Amazon 1–5 star rating to a 3-class sentiment label."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


# --- Dataset loading -------------------------------------------------------

def sample_amazon(
    sample_size: int,
    category: str,
    seed: int,
    max_stream: int = 20_000,
) -> list[dict[str, str]]:
    """Stream Amazon Reviews 2023 and return a stratified 3-class sample.

    Streams (no full download) until each class has at least
    ``2 * per_class`` candidates, then shuffles with ``seed`` and keeps the
    first ``per_class`` per class. ``max_stream`` caps how far we read in
    case a class is rare (neutrals are ~5–8% of Amazon reviews).
    """
    config = f"raw_review_{category}"
    print(f"  streaming {AMAZON_REPO} / {config} ...")
    ds = load_dataset(
        AMAZON_REPO, config, trust_remote_code=True, streaming=True
    )["full"]

    per_class = sample_size // 3
    over_fill = per_class * 2
    pool: dict[str, list[dict[str, str]]] = {lbl: [] for lbl in LABELS}

    seen = 0
    for row in ds:
        seen += 1
        text = (row.get("text") or "").strip()
        rating = row.get("rating")
        if not text or rating is None:
            continue
        label = stars_to_label(float(rating))
        if len(pool[label]) < over_fill:
            pool[label].append({"text": text, "expected_label": label})
        if all(len(v) >= over_fill for v in pool.values()):
            break
        if seen >= max_stream:
            break

    print(f"  streamed {seen} rows; pool sizes: " +
          ", ".join(f"{lbl}={len(pool[lbl])}" for lbl in LABELS))

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in LABELS:
        items = pool[label]
        rng.shuffle(items)
        if len(items) < per_class:
            print(f"  WARN: only {len(items)} {label} samples available, wanted {per_class}")
        selected.extend(items[:per_class])
    rng.shuffle(selected)
    return selected


def load_phenomenon(path: Path) -> list[dict[str, str]]:
    """Read the phenomenon CSV into a list of ``{text, expected_label, category}`` dicts."""
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "text": row["text"],
                "expected_label": row["expected_label"],
                "category": row["category"],
            })
    return rows


# --- Prediction loop -------------------------------------------------------

def run_predictions(rows: list[dict[str, str]]) -> list[str]:
    """Call ``analyze`` on each row's text; return the resolved labels."""
    total = len(rows)
    preds: list[str] = []
    for i, row in enumerate(rows, start=1):
        preds.append(analyze(row["text"])["label"])
        if i % 50 == 0 or i == total:
            print(f"    predicted {i}/{total}")
    return preds


# --- Metrics + reporting ---------------------------------------------------

def _per_class_report(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    """``classification_report`` as plain-Python dict (json-serializable)."""
    raw = classification_report(
        y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True
    )
    return {
        label: {
            "precision": float(raw[label]["precision"]),
            "recall": float(raw[label]["recall"]),
            "f1": float(raw[label]["f1-score"]),
            "support": int(raw[label]["support"]),
        }
        for label in LABELS
    }


def _format_confusion_matrix(cm: list[list[int]]) -> list[str]:
    """Render the confusion matrix as aligned ASCII lines."""
    header = " " * 14 + " ".join(f"{l[:10]:>10s}" for l in LABELS)
    lines = ["Confusion matrix (rows=true, cols=pred):", header]
    for label, row in zip(LABELS, cm):
        lines.append(f"  true_{label:<8s}" + " ".join(f"{v:>10d}" for v in row))
    return lines


def write_results(
    out_dir: Path,
    y_true: list[str],
    y_pred: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write metrics.json, confusion_matrix.csv, and report.txt under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    )
    per_class = _per_class_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    metrics: dict[str, Any] = {
        "n": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": LABELS,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    if extra:
        metrics.update(extra)

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([""] + [f"pred_{l}" for l in LABELS])
        for label, row in zip(LABELS, cm):
            writer.writerow([f"true_{label}"] + row)

    lines: list[str] = []
    if "source" in (extra or {}):
        lines.append(f"source     = {extra['source']}")
    if "seed" in (extra or {}):
        lines.append(f"seed       = {extra['seed']}")
    lines.append(f"n          = {len(y_true)}")
    lines.append(f"accuracy   = {acc:.4f}")
    lines.append(f"macro-F1   = {macro_f1:.4f}")
    lines.append("")
    lines.append("Per-class metrics:")
    for label, m in per_class.items():
        lines.append(
            f"  {label:<10s} precision={m['precision']:.3f}  "
            f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  support={m['support']}"
        )
    lines.append("")
    lines.extend(_format_confusion_matrix(cm))

    per_cat = (extra or {}).get("per_category")
    if per_cat:
        lines.append("")
        lines.append("Per-category breakdown:")
        for cat, stats in per_cat.items():
            dist = stats["predicted"]
            dist_str = ", ".join(f"{lbl}={dist.get(lbl, 0)}" for lbl in LABELS)
            lines.append(
                f"  {cat:<14s} n={stats['n']:<3d} "
                f"accuracy={stats['accuracy']:.3f}  predicted: {dist_str}"
            )

    (out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Per-dataset drivers ---------------------------------------------------

def evaluate_amazon(
    out_dir: Path, sample_size: int, category: str, seed: int
) -> None:
    print(f"Evaluating Amazon Reviews 2023 / {category} "
          f"(sample_size={sample_size}, seed={seed})")
    rows = sample_amazon(sample_size, category, seed)
    y_true = [r["expected_label"] for r in rows]
    y_pred = run_predictions(rows)
    write_results(
        out_dir / "amazon",
        y_true,
        y_pred,
        extra={"source": f"{AMAZON_REPO} / raw_review_{category}", "seed": seed},
    )
    print(f"  -> results saved to {out_dir / 'amazon'}")


def evaluate_phenomenon(out_dir: Path, csv_path: Path) -> None:
    print(f"Evaluating phenomenon test set ({csv_path})")
    rows = load_phenomenon(csv_path)
    y_true = [r["expected_label"] for r in rows]
    y_pred = run_predictions(rows)

    by_cat: dict[str, dict[str, list[str]]] = {}
    for row, pred in zip(rows, y_pred):
        cat = row["category"]
        bucket = by_cat.setdefault(cat, {"y": [], "p": []})
        bucket["y"].append(row["expected_label"])
        bucket["p"].append(pred)
    per_category: dict[str, dict[str, Any]] = {}
    for cat, bucket in by_cat.items():
        per_category[cat] = {
            "n": len(bucket["y"]),
            "accuracy": float(accuracy_score(bucket["y"], bucket["p"])),
            "predicted": dict(Counter(bucket["p"])),
        }

    write_results(
        out_dir / "phenomenon",
        y_true,
        y_pred,
        extra={"source": str(csv_path), "per_category": per_category},
    )
    print(f"  -> results saved to {out_dir / 'phenomenon'}")


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline evaluation harness.")
    parser.add_argument(
        "--dataset",
        choices=["amazon", "phenomenon", "all"],
        default="all",
        help="Which dataset(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Amazon stratified sample size (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the Amazon stratified shuffle (default: 42).",
    )
    parser.add_argument(
        "--amazon-category",
        default="Electronics",
        help="Amazon Reviews 2023 product category (default: Electronics).",
    )
    parser.add_argument(
        "--phenomenon-csv",
        type=Path,
        default=PHENOMENON_CSV,
        help=f"Phenomenon CSV path (default: {PHENOMENON_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/).",
    )
    args = parser.parse_args()

    if args.dataset in ("phenomenon", "all"):
        evaluate_phenomenon(args.out_dir, args.phenomenon_csv)
    if args.dataset in ("amazon", "all"):
        evaluate_amazon(
            args.out_dir, args.sample_size, args.amazon_category, args.seed
        )


if __name__ == "__main__":
    main()
