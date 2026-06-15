# Reflections and Future Work

A short look back at building this tool: what worked, what was harder than expected, the
trade-offs I made deliberately, and where I would take it next.

## What worked

**The layered design paid off.** Splitting the system into a thin UI, a stateless API, and a model
pipeline behind a single `analyze()` entry point kept every concern in one place. The clearest
payoff is that the sentiment model is swappable: if I ever fine-tune or replace it, nothing in the
API contract or the UI has to change. That is not an accident of the design — it was the point of
it.

**Two specialized models beat one general one.** Running a dedicated irony detector alongside the
sentiment model, and letting it correct a confident-but-ironic "positive" into a negative, is the
single mechanism that does the most work against the brief. Sarcasm, the most clearly
marked case, comes out at 10/10 on my phenomenon set. That result alone convinced me the
two-model approach was the right shape.

**Building the phenomenon set early.** Writing my own examples of irony, sarcasm, negation, and
multipolarity, in language from my own domain, gave me a target I could actually feel. Aggregate
accuracy on a public benchmark tells you very little about whether you handle sarcasm; a set built
to provoke each phenomenon tells you exactly where you stand, category by category.

**Validation as a first-class output.** Every prediction carries a confidence score and three
flags (low-confidence, irony/model-disagreement, multipolarity), and mixed reviews are broken down
segment by segment. Surfacing the system's uncertainty turned out to be more useful than chasing a
slightly higher headline number.

## What was hard, and the trade-offs I made

**Off-the-shelf models, on purpose.** I chose to use the Cardiff models as they are, without
fine-tuning. This was a deliberate decision: it keeps the system reproducible from
public checkpoints, needs no labeled training data or training infrastructure, and uses models
trained on exactly the kind of informal, opinionated text that customer reviews resemble. The
trade-off I accepted in return is that I adapt the system *to* the model, through preprocessing
and the validation laye, rather than adapting the models to the domain. For this project that was
the right call, and the layered design leaves the door open to revisit it later.

**The neutral class was the genuine surprise.** The sentiment model is trained on short, informal
text and rarely commits to a neutral label on review-length prose. On the Amazon benchmark, neutral
recall sits around 0.25, most true neutrals get pushed to positive or negative. I spent time
trying to tune my way out of this before accepting what the evaluation was telling me: it is a
property of the model's training distribution, not a threshold I can move. Recognizing the
difference between *a knob I can turn* and *a limit of the model I chose* was one of the more
useful lessons of the project.

**Confidence-gating is a blunt trigger.** My multipolarity branch engages only when the sentiment
model is uncertain (a low top-class probability). That is tractable and it works on genuinely mixed
reviews — but it has a structural blind spot: it cannot reach a review where the model is
*confidently wrong*, for example a high-confidence positive on a review that clearly ends on a
negative note. A confidence gate, by definition, can only fire when the model already doubts
itself. Seeing this clearly in the threshold sweep was the moment I understood the limitation was
in the *design of the trigger*, not in the thresholds.

## What I would do next

For the future, two changes will matter most, both come straight from what the evaluation
surfaced:

1. **Fix the neutral class at the model layer.** Fine-tune the sentiment model, or swap in one
   trained on review-length data, so it is willing to call a review neutral. Because the model sits
   behind the pipeline interface, this is a model-layer change with no impact on the API or the UI,
   exactly the kind of substitution the architecture was built to allow. This is the highest-value
   next step for overall accuracy.

2. **Replace the confidence gate with a polarity-disagreement signal.** Instead of triggering the
   multipolarity branch on low full-pass confidence, score the segments first and engage the branch
   whenever they *disagree* in polarity (one segment positive, another negative) regardless of how
   confident the full-pass reading was. This directly addresses the confidently-wrong reviews the
   current gate cannot reach, and it would lift the multipolarity category, which is today the
   weakest on the phenomenon set.

## Closing thought

The project taught me as much about *knowing the limits of a chosen tool* as about building one.
The system meets the brief, and where it falls short it does so for reasons I can name precisely and
point to in the numbers. That honesty, a tool that tells you when it is unsure, backed by
documentation that tells you where it is weak, is the outcome I am most satisfied with.
