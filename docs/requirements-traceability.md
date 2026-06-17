# Requirements Traceability

This document maps each mandatory requirement to the part of the system that satisfies it and to
the evidence that demonstrates it. It is the bridge between the original brief and the shipped
code: every requirement is traced to a concrete mechanism (in `pipeline.py`) and to measured
results (under `results/`).

## Source requirements

The tool must identify how customers feel about a product from their reviews. Two capabilities are
mandatory:

- **R1 — Valence levels.** Distinguish **at least three levels of valence**: positive, neutral,
  and negative.
- **R2 — Linguistic phenomena.** Correctly handle the constructions that distort surface-level
  sentiment: **irony, sarcasm, negation, and multipolarity**.

R2 is treated below as four sub-requirements (R2a–R2d) so each phenomenon can be traced to its own
mechanism and evidence.

## Traceability matrix

| ID | Requirement | Mechanism | Evidence |
| --- | --- | --- | --- |
| R1 | Three valence levels | Three-class sentiment model; single resolved `label` ∈ {negative, neutral, positive} | `results/amazon/report.txt` (all three classes scored) |
| R2a | Irony | Parallel irony detector + irony-correction branch | `results/phenomenon/report.txt` — irony category |
| R2b | Sarcasm | Same irony detector + correction branch | `results/phenomenon/report.txt` — sarcasm category |
| R2c | Negation | Model handles it implicitly; spaCy negation-scope markup for transparency | `results/phenomenon/report.txt` — negation category |
| R2d | Multipolarity | Multipolarity flag → per-segment scoring → last-clause aggregation | `results/phenomenon/report.txt` — multipolarity category |

The two benchmarks behind the evidence column are the labeled **Amazon Reviews 2023** sample
(Electronics, n = 498, stratified across the three classes) and the curated **phenomenon set**
(`tests/fixtures/phenomenon_reviews.csv`, n = 45), which contains hand-written examples in each of
the four phenomenon categories. The phenomenon set exists specifically to exercise R2.

---

## R1 — Three valence levels

**How it is met.** The sentiment stage uses
`cardiffnlp/twitter-roberta-base-sentiment-latest`, which emits a full probability distribution
over the three classes `negative`, `neutral`, and `positive`. The pipeline (`_predict_sentiment`
in `pipeline.py`) takes the top class as the raw label and carries the full distribution through to
the response as `sentiment_raw.distribution`. Every `analyze()` result therefore resolves to
exactly one of the three required valence levels, and the underlying three-way distribution is
always visible to a consumer. The choice of a three-class scheme — rather than binary or
finer-grained — is recorded in [ADR 0001](adr/0001-three-class-sentiment-scheme.md).

**Evidence.** On the Amazon Reviews 2023 benchmark (n = 498, ~166 per class), all three classes are
predicted and scored; headline accuracy is 0.568 with macro-F1 0.542. Per-class F1 is 0.620
(negative), 0.345 (neutral), and 0.661 (positive). Full per-class precision/recall and the
confusion matrix are in `results/amazon/`.

**Honest limit.** Neutral recall is low (0.247): on review-length prose the sentiment model rarely
emits a neutral label, so most true neutrals are predicted positive or negative. This is a property
of the model's training distribution (short, informal text), not a tunable threshold — a sweep
confirmed it cannot be recovered by re-tuning. The three-class capability is met; its accuracy on
the neutral class is the principal limitation, accepted and documented in
[ADR 0002](adr/0002-off-the-shelf-pretrained-models.md).

## R2a / R2b — Irony and sarcasm

**How it is met.** A second model, `cardiffnlp/twitter-roberta-base-irony`, runs in parallel with
the sentiment model on the same text (`_predict_irony` in `pipeline.py`). When it labels a review
*irony* with confidence ≥ `IRONY_HIGH_CONF_THRESHOLD` (0.50), the `model_agreement` flag fires,
signalling that a literal sentiment reading is contradicted. The validator then **corrects** the
label: a confident positive (≥ 0.70) is inverted to **negative**, a less confident positive is
downgraded to **neutral**, and an already-negative or neutral reading is kept. The pre-correction
sentiment label, the irony output, and the resolved label are all retained in the response, so the
correction is transparent. Both the parallel-model design and the correction rules are recorded in
[ADR 0003](adr/0003-two-specialized-models.md). Sarcasm is handled by the same mechanism — it is a
form of irony as far as the detector is concerned — so it is traced to the same branch but
evidenced separately.

**Evidence** (`results/phenomenon/report.txt`, per-category breakdown):

- **Sarcasm:** 10/10 (accuracy 1.000) — every sarcastic example is resolved to its intended
  negative meaning.
- **Irony:** 6/10 (accuracy 0.600) — the majority of ironic examples are corrected.

**Honest limit.** The irony branch is gated on the irony detector firing. Some ironic phrasings are
read literally by the sentiment model with high confidence and are not flagged ironic, so they slip
through uncorrected; this is the same confident-but-wrong failure mode discussed under R2d. Sarcasm
(blunter, more lexically marked) is caught reliably; subtle irony is the weaker case.

## R2c — Negation

**How it is met.** Negation is handled on two levels. First, the sentiment model has learned
negation patterns implicitly from its training data and receives the **unannotated** text, so
phrases like *"not bad"* or *"doesn't work"* are scored by a model that has seen such constructions.
Second, for transparency, a spaCy preprocessing step (`_annotate_negation` in `pipeline.py`) marks
negation scopes in the displayed text: a negation cue (*not, no, never, without, …*) extends
forward to the next punctuation mark or contrastive boundary (*but, however, …*) and the span is
wrapped as `[NEG: …]`. This markup is **display-only** — it documents where the system sees negation
without altering the text the model classifies.

**Evidence.** On the phenomenon negation category (n = 10), accuracy is 0.700
(`results/phenomenon/report.txt`). The set deliberately includes double negatives and
negated-but-neutral cases (*"Not bad, not great. Just exists."*) to probe the harder constructions.

**Honest limit.** Because correction is the model's responsibility and the markup is purely
informational, the system has no rule-based override for negation; the failures are cases where the
model itself misreads a negated construction. The markup makes those cases legible but does not fix
them.

## R2d — Multipolarity

**How it is met.** When a review mixes polarities (*"great screen, but the keyboard is mushy"*), the
sentiment distribution tends to flatten. The pipeline uses a low top-class probability as a
tractable proxy: when it falls below `MULTIPOLARITY_TOP_CLASS_THRESHOLD` (0.60), the `multipolarity`
flag fires. The review is then split into segments — spaCy sentence boundaries first, with a
clause-level fallback (splitting on commas, semicolons, and contrastive connectives) when the text
is a single sentence — and **each segment is scored independently**. The per-segment scores are
always reported. The single aggregate label follows the **final** segment ("last-clause wins"),
matching how a reader weights the closing clause of a mixed review (a recency effect). The
segmentation and aggregation rules are recorded in
[ADR 0003](adr/0003-two-specialized-models.md).

**Evidence.** On the phenomenon multipolarity category (n = 15, ten negative-ending and five
positive-ending), accuracy is 0.467 (`results/phenomenon/report.txt`). The per-segment view is
exercised by the integration tests, which assert the last-clause-wins aggregation and the
single-sentence clause fallback.

**Honest limit.** The branch is *confidence-gated*: it engages only when the full-pass sentiment is
uncertain, so it cannot reach reviews where the model is **confidently wrong** — for example a
high-confidence positive on a review that ends negatively. A more robust trigger would engage on
per-segment polarity *disagreement* regardless of full-pass confidence. This is the largest single
weakness on the phenomenon set and is recorded as future work in
[ADR 0003](adr/0003-two-specialized-models.md) and the
[reflections doc](reflections-and-future-work.md).

---

## Summary

| Requirement | Status | Where it can fail |
| --- | --- | --- |
| R1 — three valence levels | Met | Neutral recall low on long-form prose (model distribution) |
| R2a — irony | Met | Subtle irony read confidently-literally slips the gate |
| R2b — sarcasm | Met (10/10) | — |
| R2c — negation | Met | No rule-based override; relies on the model |
| R2d — multipolarity | Met | Confidence gate misses confidently-wrong reviews |

All mandatory requirements are satisfied with a working mechanism and measured evidence. The
limitations are real and documented rather than hidden: they cluster on the neutral class and on
reviews the off-the-shelf model gets wrong with high confidence. For the underlying design
rationale see the [ADRs](adr/); for the evaluation
methodology and full numbers see `results/`.
