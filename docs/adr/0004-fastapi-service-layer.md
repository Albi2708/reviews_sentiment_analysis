# ADR 0004: FastAPI as the service layer

- **Status:** Accepted

## Context

The UI and the model core need a clean contract so each can be developed, tested, and deployed
independently. The service layer should be type-safe, self-documenting, and impose minimal
overhead. Model inference is CPU-bound and blocking, which constrains how request handlers
should be written.

## Decision

Use **FastAPI** with Pydantic request and response models. The response model mirrors the
pipeline's output one-to-one. Handlers are declared **synchronous** so FastAPI dispatches them
to a worker threadpool — the correct shape for blocking inference, where an `async` handler
would stall the event loop. The models are warmed eagerly in a **startup lifespan** so the first
real request does not pay the load cost.

## Consequences

- **Self-documenting** — OpenAPI docs are generated automatically at `/docs` and `/redoc`, and
  stay in sync with the Pydantic schema.
- **Clean decoupling** — the UI depends only on the REST contract, so the model layer can change
  without touching the front end (the central goal of the layered design).
- **Input validation for free** — Pydantic rejects malformed requests (for example, empty text)
  with a `422` before any model runs.
- **Statelessness** — the service holds no per-request state; cross-request state (the prediction
  log) is delegated to the storage layer (see [ADR 0005](0005-sqlite-prediction-logging.md)).
- **One network hop** between UI and model — acceptable, as both run within the same deployment
  network.
