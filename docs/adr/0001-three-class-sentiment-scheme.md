# ADR 0001: Three-class sentiment scheme

- **Status:** Accepted

## Context

The product requirement is to identify how customers feel about a product from their reviews,
distinguishing **at least three levels of valence**. The label scheme therefore has to (a) meet
that minimum bar, (b) align with the output of the chosen sentiment model, and (c) map cleanly
onto star ratings, which serve as a noisy auxiliary ground truth where they accompany a review.

## Decision

Adopt a uniform three-class scheme — **negative, neutral, positive** — across the entire system:
the pipeline output, the API contract, the evaluation labels, and the star-rating mapping. Stars
map `1–2 → negative`, `3 → neutral`, `4–5 → positive`.

Mixed-polarity reviews are *not* given a separate class; they are handled by segmentation and a
single aggregate label (see [ADR 0003](0003-two-specialized-models.md)).

## Consequences

- **Meets the mandatory bar** of three valence levels.
- **No label remapping at inference** — the sentiment model emits exactly these three classes,
  so its output is used directly.
- **Clean cross-checking** — the star-rating mapping is unambiguous, so star ratings can be used
  as auxiliary ground truth for benchmarking.
- **Coarse granularity (accepted)** — the scheme carries no intensity or fine-grained sentiment,
  and no dedicated "mixed" label; multipolarity is expressed through per-segment scores plus an
  aggregate rather than a fourth class.
- **Neutral is the hardest class** on long-form text, which surfaces as a recall limitation
  rooted in the model rather than the scheme — see
  [ADR 0002](0002-off-the-shelf-pretrained-models.md).
