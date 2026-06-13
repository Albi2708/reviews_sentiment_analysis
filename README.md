# Customer Review Sentiment Analysis

A tool that classifies customer reviews into three sentiment classes — **positive**,
**neutral**, **negative** — and explicitly handles the linguistic phenomena that distort
surface-level sentiment: **irony, sarcasm, negation, and multipolarity**. Every prediction
carries a confidence score and validation flags so its reliability can be judged at a glance.

The system is split into a thin Streamlit UI, a stateless FastAPI service, and a model layer,
so any component can be swapped independently (for example, replacing the sentiment model with
a fine-tuned variant requires no UI or API change).

## Highlights

- **Three-class sentiment** from `cardiffnlp/twitter-roberta-base-sentiment-latest`, whose
  informal, opinionated training distribution matches how customers actually write.
- **Irony / sarcasm correction.** A second model, `cardiffnlp/twitter-roberta-base-irony`,
  runs in parallel; when it flags a review as ironic with high confidence, the sentiment label
  is inverted (confident positive → negative) or downgraded to neutral, and the conflict is
  surfaced as a flag.
- **Negation transparency.** A spaCy preprocessing step wraps negation scopes as `[NEG: ...]`
  for display; the models still receive the unannotated text.
- **Multipolarity handling.** When the sentiment distribution is near-uniform, the review is
  split into segments, each scored independently; both per-segment and aggregate labels are
  reported (the aggregate follows the final clause).
- **Per-prediction validation.** Each result includes a confidence score, a low-confidence
  flag, a model-agreement (irony) flag, and a multipolarity flag.
- **Prediction logging.** Every `/analyze` call is logged to SQLite (timestamp, input, label,
  confidence, flags, latency).
- **Evaluation harness.** Standard metrics (accuracy, macro-F1, confusion matrices) on both a
  labeled public dataset and a curated phenomenon test set.

## Architecture

```
        Browser
           │
      Streamlit UI  ──REST/JSON──▶  FastAPI service ──▶  Preprocessor (ftfy + spaCy)
        (ui.py)                       (api.py)        ──▶  Sentiment model ┐
                                          │                Irony model     ┘──▶ Validator
                                          │                                       │
                                          └────────────────────────────────▶  SQLite log
```

The pipeline (`pipeline.py`) composes preprocessing → both models → validator and exposes a
single `analyze(text)` entry point. The API wraps it; the UI calls the API over HTTP.

## Quickstart

### Option A — Docker (recommended, one command)

No local Python setup required. From the repository root:

```bash
docker compose up --build
```

This builds the image (installing dependencies, the `en_core_web_sm` spaCy model, and
pre-caching the two Cardiff models so the first request is fast), then starts both services:

- **UI** → http://localhost:8501
- **API** → http://localhost:8000 (interactive docs at http://localhost:8000/docs)

The UI is pre-configured to reach the API service. Stop with `Ctrl+C`; on later runs you can
drop `--build`.

### Option B — Local (virtual environment)

Requires **Python 3.10+** (developed and tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # pip install -r requirements.txt + spaCy model download
```

Then start the two services in separate terminals:

```bash
make run                         # FastAPI on http://localhost:8000
make ui                          # Streamlit on http://localhost:8501
```

The UI reads the API location from the `API_URL` environment variable
(default `http://localhost:8000`).

## Usage

### Web UI

Open http://localhost:8501. The **Single review** tab analyzes one review and shows the label,
confidence, flags, a per-segment breakdown (when multipolarity fires), and model provenance.
The **Batch CSV** tab accepts a CSV with a `text` column, analyzes every non-empty row, and
offers the results as a downloadable CSV.

### API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Best laptop I have ever owned. Battery lasts all day and it boots in seconds."}'
```

Response (numeric values illustrative):

```json
{
  "text": "Best laptop I have ever owned. Battery lasts all day and it boots in seconds.",
  "preprocessed_text": "Best laptop I have ever owned. Battery lasts all day and it boots in seconds.",
  "label": "positive",
  "confidence": 0.98,
  "flags": {
    "low_confidence": false,
    "model_agreement": false,
    "multipolarity": false
  },
  "sentiment_raw": {
    "label": "positive",
    "confidence": 0.98,
    "distribution": { "negative": 0.01, "neutral": 0.01, "positive": 0.98 }
  },
  "irony": { "label": "non_irony", "confidence": 0.95 },
  "segments": null
}
```

Field notes:

- `label` / `confidence` — the resolved sentiment after any irony correction or multipolarity
  aggregation.
- `preprocessed_text` — Unicode-normalized text with negation scopes wrapped as `[NEG: ...]`.
- `flags.model_agreement` — `true` when the irony detector contradicts the sentiment model; in
  that case `label` may differ from `sentiment_raw.label` (e.g. a confident positive inverted
  to negative).
- `flags.multipolarity` — `true` when the review spans multiple polarities; `segments` is then
  a list of per-segment `{text, label, confidence}` objects (otherwise `null`).
- `sentiment_raw` / `irony` — raw model outputs, kept for provenance.

`GET /health` returns service status and a `models_warm` flag (the models are warmed eagerly at
startup so the first real request isn't slow).

### Python

```python
from pipeline import analyze

result = analyze("Great materials but the balancing is a complete mess.")
print(result["label"], result["confidence"])
```

## API reference

With the API running, the auto-generated OpenAPI documentation is available at:

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

## Evaluation

Run the evaluation harness:

```bash
make eval                        # python evaluate.py
```

It writes `metrics.json`, `confusion_matrix.csv`, `confusion_matrix.png`, and a human-readable
`report.txt` per dataset under `results/`. Headline results on the locked configuration:

| Test set                              |   n | Accuracy | Macro-F1 |
| ------------------------------------- | --: | -------: | -------: |
| Amazon Reviews 2023 (Electronics)     | 498 |    0.568 |    0.542 |
| Phenomenon set (irony/sarcasm/…)      |  45 |    0.667 |    0.470 |

On the phenomenon set, sarcasm is handled at 10/10 and negation at 7/10. See `results/` for the
full per-class and per-category breakdowns.

## Known limitations & future improvements

- **Neutral-class collapse on long-form reviews.** On the Amazon Electronics sample neutral
  recall is ~0.25 — most true neutrals are predicted positive or negative. The sentiment model
  is Twitter-trained and rarely emits a neutral label on review-length prose. This is a
  model-distribution issue rather than a threshold one, so it cannot be tuned away. *Future
  work:* fine-tune or swap the sentiment model on review-length data — the layered architecture
  makes this a model-layer change with no UI or API impact.

- **Multipolarity branch engages only on uncertain predictions.** The multipolarity branch is
  gated on the full-pass sentiment confidence being low, so it cannot reach reviews where the
  model is *confidently wrong* (for instance, a high-confidence positive on a review that ends
  negatively). *Future work:* replace the confidence trigger with a per-segment polarity-
  disagreement signal — engage the branch whenever segments disagree, regardless of full-pass
  confidence.

## Project layout

```
.
├── api.py            FastAPI service: POST /analyze, GET /health
├── ui.py             Streamlit front end (single review + batch CSV)
├── pipeline.py       Preprocess → sentiment + irony models → validator
├── db.py             SQLite prediction logging
├── evaluate.py       Evaluation harness (Amazon + phenomenon sets)
├── sweep.py          Threshold-sweep utility
├── requirements.txt
├── Makefile          install / run / ui / test / eval / sweep
├── Dockerfile        Container image for the stack
├── docker-compose.yml  One-command API + UI run
├── results/          Saved evaluation metrics and confusion matrices
└── tests/            pytest suite + phenomenon fixture
```

## Datasets & models

This project uses one labeled benchmark dataset and two off-the-shelf pretrained models. Cited
in APA 7:

**Dataset**

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging language and
> items for retrieval and recommendation* (arXiv:2403.03952). arXiv.
> https://arxiv.org/abs/2403.03952

**Models**

> Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., & Camacho-Collados, J. (2022).
> TimeLMs: Diachronic language models from Twitter. In *Proceedings of the 60th Annual Meeting
> of the Association for Computational Linguistics: System Demonstrations* (pp. 251–260).
> Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.acl-demo.25

> Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified
> benchmark and comparative evaluation for tweet classification. In *Findings of the
> Association for Computational Linguistics: EMNLP 2020* (pp. 1644–1650). Association for
> Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.148

The Amazon Reviews 2023 dataset is released for non-commercial, academic use; it is streamed at
evaluation time (no full download) and is not redistributed with this repository.

## License

Released under the MIT License. See [LICENSE](LICENSE).
