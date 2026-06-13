# User Guide

This guide walks through using the Customer Review Sentiment Analysis tool, both from the web
interface and directly from the API. For installation and how to start the services, see the
[development guide](development.md).

## What the tool does

You give it a customer review; it returns a sentiment label — **positive**, **neutral**, or
**negative** — with a confidence score. It is built to cope with the things that fool naive
sentiment tools: **irony and sarcasm**, **negation**, and **multipolarity** (a review that is
positive about one thing and negative about another). When it spots one of these, it says so
through a flag, and for mixed reviews it breaks the text down segment by segment.

## The web interface

Once the stack is running, open **http://localhost:8501**. The sidebar shows the API address and
whether the models are ready ("reachable — models warm"); if it shows *unreachable*, the API
isn't up yet.

There are two tabs.

### Single review

1. Paste a review into the text box.
2. Click **Analyze** (or press `Ctrl+Enter`).

The results panel shows:

- **The label**, in colour, with the confidence as a percentage.
- **Flags**, if any (see [Reading the results](#reading-the-results)).
- **Preprocessed text** (expandable) — the normalized text with negation scopes wrapped as
  `[NEG: ...]`.
- **Per-segment breakdown** — a table of each segment's label and confidence, shown only when
  the review is multipolar.
- **Model provenance** (expandable) — the raw sentiment label and its full three-class
  probability distribution, plus the irony detector's reading.

### Batch CSV

1. Upload a CSV file that has a **`text`** column (other columns are ignored; empty rows are
   skipped).
2. Click **Run batch**.

A progress bar tracks the run. When it finishes you get a results table — one row per review with
its label, confidence, flags, and raw model outputs — and a **Download results CSV** button. If
any individual review fails, its row is marked `ERROR` and the rest still complete.

## Reading the results

**Label and confidence.** The label is the tool's overall verdict; the confidence is how sure it
is. Confidence below 70% raises the low-confidence flag.

**Flags** highlight cases worth a closer look:

| Flag | Meaning |
| --- | --- |
| **low confidence** | The final confidence is below the 70% threshold — treat the label as tentative. |
| **irony detected** | The irony detector contradicts a literal reading. The label may have been corrected (a confident ironic "positive" is flipped to negative, a less confident one downgraded to neutral). |
| **multipolar** | The review spans more than one polarity. The aggregate label follows the **final** part of the review, and the per-segment table shows the rest. |

**Per-segment breakdown.** For a review like *"Great materials, but the balancing is a complete
mess,"* the tool scores each part separately and reports the closing part as the headline — which
is how a reader would summarize it.

## Worked examples

| Review | Label | What to notice |
| --- | --- | --- |
| "Best laptop I've ever owned. Battery lasts all day." | positive | High confidence, no flags. |
| "Oh fantastic, another pair of headphones that broke after a week." | negative | **irony detected** — the surface-positive wording is corrected. |
| "Great materials, but the balancing is a complete mess." | negative | **multipolar** — per-segment table shows a positive opening and a negative close; the close wins. |

## Using the API directly

The web interface is a thin client over a REST API; you can call it yourself.

### Analyze a review

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Great materials, but the balancing is a complete mess."}'
```

The response includes the resolved `label` and `confidence`, the `flags` object, the per-segment
`segments` list (when multipolar), and the raw `sentiment_raw` and `irony` outputs for
transparency. The full field-by-field schema is in the [architecture doc](architecture.md#output-contract).

### Check health

```bash
curl http://localhost:8000/health
```

Returns `status` and a `models_warm` flag — useful for confirming the service is ready before
sending traffic.

### Interactive documentation

With the API running, browse the auto-generated documentation at **http://localhost:8000/docs**
(Swagger UI) or **http://localhost:8000/redoc**. You can try requests directly from the page.
