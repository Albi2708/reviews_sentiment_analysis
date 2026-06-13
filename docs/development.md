# Development & Setup Guide

How to run, test, and develop the Customer Review Sentiment Analysis tool. For using the running
application, see the [user guide](user-guide.md); for the system design, see the
[architecture doc](architecture.md).

## Prerequisites

- **Docker** (with Compose v2) for the one-command run, **or**
- **Python 3.10+** (developed and tested on 3.12) for local development.

The first run downloads the two Cardiff models and the spaCy model (several hundred MB). With
Docker these are baked into the image at build time; locally they are fetched on first use.

## Option A — Docker (recommended)

From the repository root:

```bash
docker compose up --build
```

This builds a single image (dependencies, the `en_core_web_sm` spaCy model, and both Cardiff
models pre-cached) and starts two services:

- **API** → http://localhost:8000 (docs at `/docs`)
- **UI** → http://localhost:8501

The UI is configured to reach the API over the internal Docker network. The prediction log is
stored on a named volume, so it survives restarts. Stop with `Ctrl+C`; on later runs you can drop
`--build`. To stop and remove the containers:

```bash
docker compose down
```

## Option B — Local virtual environment

```bash
git clone <repository-url>
cd reviews_sentiment_analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

make install                       # pip install -r requirements.txt + spaCy model download
```

Then run the two services in separate terminals:

```bash
make run                           # FastAPI on http://localhost:8000 (uvicorn --reload)
make ui                            # Streamlit on http://localhost:8501
```

The UI reads the API location from the `API_URL` environment variable (default
`http://localhost:8000`).

## Running the tests

```bash
make test                          # pytest
```

The suite covers the preprocessor, every validator branch (against lightweight model stubs), and
an end-to-end `/analyze` integration test. The integration test loads the real Cardiff models, so
the first run is slower while they download and warm.

## Evaluation and threshold tuning

```bash
make eval                          # python evaluate.py — benchmark + phenomenon metrics
make sweep                         # python sweep.py — one-at-a-time threshold sweep
```

`evaluate.py` writes `metrics.json`, confusion matrices, and a readable `report.txt` per dataset
under `results/`. It streams the Amazon Reviews 2023 benchmark (no full download) and reads the
phenomenon set from `tests/fixtures/phenomenon_reviews.csv`. Both accept CLI flags (`--dataset`,
`--sample-size`, `--seed`, …); run with `--help` for the full list.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_URL` | `http://localhost:8000` | Where the UI sends requests. Set automatically in Docker. |
| `HF_HOME` | (Hugging Face default) | Model cache location. Set to a fixed path in the image. |

The SQLite prediction log lives at `data/predictions.db` (created automatically; on a mounted
volume under Docker).

## Make targets

| Target | Action |
| --- | --- |
| `make install` | Install dependencies and the spaCy model. |
| `make run` | Start the FastAPI service. |
| `make ui` | Start the Streamlit UI. |
| `make test` | Run the test suite. |
| `make eval` | Run the evaluation harness. |
| `make sweep` | Run the threshold sweep. |

## Project layout

```
.
├── api.py            FastAPI service: POST /analyze, GET /health
├── ui.py             Streamlit front end (single review + batch CSV)
├── pipeline.py       Preprocess → sentiment + irony models → validator
├── db.py             SQLite prediction logging
├── evaluate.py       Evaluation harness (benchmark + phenomenon sets)
├── sweep.py          Threshold-sweep utility
├── requirements.txt  Pinned dependencies
├── Makefile          install / run / ui / test / eval / sweep
├── Dockerfile        Container image for the stack
├── docker-compose.yml  One-command API + UI run
├── docs/             Architecture, ADRs, user & development guides
├── results/          Saved evaluation metrics and confusion matrices
└── tests/            Test suite + phenomenon fixture
```
