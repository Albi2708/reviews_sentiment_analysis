"""Validate the Cardiff NLP sentiment and irony models on a sample set of reviews.

One-off script for roadmap item 1: confirms that both transformer models from
the project concept load and produce sensible predictions on a small, varied
set of customer reviews covering the phenomenon categories we care about
(clear positive / negative / neutral, sarcasm, explicit negation, and
multipolarity).

Run with: python validate_models.py
First run will download both models (~500 MB) into ~/.cache/huggingface/.
"""
from transformers import pipeline

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"

SAMPLE_REVIEWS: list[dict[str, str]] = [
    {
        "category": "clear positive",
        "text": "Honestly the best laptop I've ever owned. Battery lasts all day, the screen is gorgeous, and it boots in under five seconds. Worth every penny.",
    },
    {
        "category": "clear negative",
        "text": "Arrived dented, three buttons don't work, and customer support hung up on me twice. Already filing for a refund.",
    },
    {
        "category": "clear neutral",
        "text": "It's a phone case. Fits the phone, cut-outs are in the right places. The \"black\" is more of a charcoal grey under sunlight.",
    },
    {
        "category": "sarcastic",
        "text": "Oh fantastic, another pair of headphones that broke after a week. Truly the legendary build quality the ad promised.",
    },
    {
        "category": "negated",
        "text": "I wouldn't say this vacuum is bad, but it's nothing I'd recommend either — suction drops on carpet and the canister is a pain to empty.",
    },
    {
        "category": "negated",
        "text": "Doesn't work as described. Battery life isn't even close to 12 hours — more like 4 — and the charger isn't even the right one for my region.",
    },
    {
        "category": "clear positive",
        "text": "This food supplement changed my life! I totally feel better now, I have more energy during the day and sleep better at night. Recommended!",
    },
    {
        "category": "clear negative",
        "text": "The noise isolation feature doesn't work as expected, and the bluetooth connection is bad too. I'll request a complete refund.",
    },
    {
        "category": "clear neutral",
        "text": "This charger is compatible with USB type C ports. The default model is white, but it's provided in black too.",
    },
    {
        "category": "sarcastic",
        "text": "This food supplement is amazing! If you are looking for constant stomach and headaches, with no benefits, this is the product for you! 10/10.",
    },
    {
        "category": "multipolarity",
        "text": "Great materials and shape, very modern. Unfortunately, the balancing, which is the most important feature, is a complete mess.",
    },
]


def main() -> None:
    """Load both Cardiff NLP models and print predictions for each sample review."""
    print(f"Loading sentiment model: {SENTIMENT_MODEL}")
    sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    print(f"Loading irony model:     {IRONY_MODEL}")
    irony = pipeline("text-classification", model=IRONY_MODEL)
    print()

    for i, review in enumerate(SAMPLE_REVIEWS, start=1):
        s = sentiment(review["text"])[0]
        ir = irony(review["text"])[0]
        print(f"[{i:>2}] expected: {review['category']}")
        print(f"     text:      {review['text']}")
        print(f"     sentiment: {s['label']:<10s}  (conf {s['score']:.3f})")
        print(f"     irony:     {ir['label']:<10s}  (conf {ir['score']:.3f})")
        print()


if __name__ == "__main__":
    main()
