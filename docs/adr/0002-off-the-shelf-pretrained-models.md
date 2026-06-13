# ADR 0002: Off-the-shelf pretrained models

- **Status:** Accepted

## Context

The system needs a sentiment model and an irony detector. The Cardiff NLP models
(`twitter-roberta-base-sentiment-latest` and `twitter-roberta-base-irony`) are trained on
informal, opinionated, often ironic text — a distribution close to how customers write reviews —
and are available as public checkpoints. The alternative, fine-tuning, would require labeled
in-domain data, a training pipeline, and the time and reproducibility risk that come with it.

## Decision

Use both Cardiff models **as-is, without fine-tuning**. Adapt the system to the models' behavior
through preprocessing and the validation layer rather than by retraining. The architecture keeps
the model layer behind a stable interface so a fine-tuned model can be substituted later.

## Consequences

- **No training overhead** — no labeled training data, training infrastructure, or training-time
  compute; the system is reproducible from public checkpoints.
- **Substitutable** — because the model sits behind the pipeline interface, a fine-tuned variant
  can replace it with no change to the API contract or the UI.
- **Neutral-class collapse on long-form reviews (accepted trade-off).** The sentiment model is
  trained on short text and rarely emits a neutral label on review-length prose. On the benchmark
  (Amazon Electronics, n = 498), neutral recall is **0.25**: of 166 true neutrals, 41 are
  predicted neutral, 76 negative, and 49 positive. This is a property of the model's training
  distribution, not a tunable parameter — a threshold sweep confirmed it cannot be recovered by
  re-tuning. It is accepted as a known limitation of using the model off the shelf.
- **Mitigation / future work** — fine-tune or swap in a model trained on review-length data; the
  layered design makes this a model-layer change with no downstream impact.
