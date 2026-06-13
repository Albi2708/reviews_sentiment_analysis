# ADR 0005: SQLite for prediction logging

- **Status:** Accepted

## Context

The system should log requests and predictions so they can be inspected later — auditing
behavior, spotting low-confidence or flagged cases, and tracking latency — without standing up a
separate database service or adding operational overhead.

## Decision

Log each successful `/analyze` call to a single **SQLite** file. The schema (timestamp, input
text, label, confidence, the three validation flags as separate integer columns, and latency in
milliseconds) is created on startup. Logging is **best-effort**: any storage error is caught and
warned to stderr so it can never break a successful response. The startup schema creation, by
contrast, fails loudly, since a broken database at boot indicates a configuration problem.
Latency records pipeline time only, excluding HTTP parsing and serialization.

## Consequences

- **Zero configuration** — a single file, no extra service to run or manage.
- **Directly queryable** — the three flags are stored as separate columns, so flagged or
  low-confidence predictions can be filtered with plain SQL.
- **Off the critical path** — logging failures degrade observability, not correctness; a
  successful analysis is always returned to the client.
- **Single-writer (accepted limitation)** — SQLite is not suited to high write concurrency. This
  is adequate for a single-instance service or demo, but not for horizontal scale.
- **Local file** — in a container the database lives on a mounted volume so it survives restarts.
  Replacing SQLite with a networked database would be a storage-layer change behind the same
  logging call.
