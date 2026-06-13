# ADR 0003: Two specialized models in parallel

- **Status:** Accepted

## Context

A single sentiment model systematically misclassifies ironic and sarcastic reviews: it reads
surface positivity literally and misses the intended negative meaning. The requirement
explicitly mandates handling irony and sarcasm, so a single general-purpose sentiment model is
insufficient on its own.

## Decision

Run a dedicated **irony detector in parallel** with the sentiment model on the same input. When
the irony detector labels a review as ironic with high confidence, the sentiment prediction is
**corrected**:

- a confident *positive* is **inverted to negative**;
- a less confident *positive* is **downgraded to neutral**;
- an already negative or neutral reading is kept.

The conflict is surfaced through the `model_agreement` flag, and both raw model outputs are
retained in the response. Multipolar reviews are split into segments, each scored by the same
sentiment model, with the aggregate label following the final segment ("last-clause wins").

## Consequences

- **Addresses the irony/sarcasm requirement directly** — on the phenomenon set, sarcasm is
  handled at 10/10.
- **Transparent correction** — the pre-correction sentiment label, the irony output, and the
  per-segment scores are all preserved, so a reviewer can see why a label changed.
- **Independently swappable** — sentiment and irony models can be replaced separately.
- **Cost** — two model loads (memory and latency) instead of one.
- **Confidence-gated branches (accepted limitation).** The multipolarity branch is triggered by
  *low* full-pass sentiment confidence, so it cannot engage on reviews where the model is
  **confidently wrong** — for example a high-confidence positive on a review that ends
  negatively. A more robust trigger would engage on per-segment polarity disagreement regardless
  of full-pass confidence; this is recorded as future work.
