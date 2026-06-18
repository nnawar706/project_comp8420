"""
Sentiment analysis.

Two routes:
  * VADER  -> fast, no training, the baseline.
  * Transformer (optional, lazy) -> HF pipeline for the neural comparison.

Sentiment feeds prioritisation: a strongly negative customer raises a priority
flag and nudges the escalation decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ML_SENTIMENT_PATH = config.MODELS_DIR / "ml_sentiment.joblib"


def _compound_to_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


class SentimentAnalyzer:
    """VADER by default; switch to a transformer with backend='transformer',
    or a traditional-ML model (TF-IDF + LogisticRegression) with backend='ml'."""

    def __init__(self, backend: str = "vader"):
        self.backend = backend
        self._vader = None
        self._pipe = None
        self._ml = None

    # --- lazy loaders ------------------------------------------------------ #
    @property
    def vader(self):
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    @property
    def pipe(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("sentiment-analysis", model=config.HF_SENTIMENT_MODEL)
        return self._pipe

    @property
    def ml(self):
        if self._ml is None:
            import joblib
            if not ML_SENTIMENT_PATH.exists():
                raise FileNotFoundError(
                    f"{ML_SENTIMENT_PATH} not found. Train it first with "
                    "sentiment.train_ml_sentiment(texts)."
                )
            self._ml = joblib.load(ML_SENTIMENT_PATH)
        return self._ml

    # --- inference --------------------------------------------------------- #
    def analyze(self, text: str) -> dict:
        if self.backend == "transformer":
            res = self.pipe(text[:512])[0]
            label = res["label"].lower()          # POSITIVE / NEGATIVE
            score = float(res["score"])
            compound = score if label == "positive" else -score
            label = "positive" if label == "positive" else "negative"
        elif self.backend == "ml":
            proba = self.ml.predict_proba([text])[0]
            classes = list(self.ml.classes_)
            idx = int(proba.argmax())
            label = str(classes[idx])
            score = float(proba[idx])
            # signed pseudo-compound for a consistent escalation signal
            pos = proba[classes.index("positive")] if "positive" in classes else 0.0
            neg = proba[classes.index("negative")] if "negative" in classes else 0.0
            compound = float(pos - neg)
        else:
            scores = self.vader.polarity_scores(text)
            compound = float(scores["compound"])
            label = _compound_to_label(compound)
            score = abs(compound)

        return {
            "label": label,
            "score": round(score, 4),
            "compound": round(compound, 4),
            "priority": compound <= config.SENTIMENT_ESCALATE_COMPOUND,
            "backend": self.backend,
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]


def train_ml_sentiment(texts: list[str], labels: list[str] | None = None):
    """
    Train a traditional-ML sentiment classifier (TF-IDF + LogisticRegression).

    If `labels` is None we *weak-label* the corpus with VADER (a form of
    distillation) so we can demonstrate a trained ML model even though Bitext has
    no gold sentiment column. For a stronger result, pass a small hand-labelled
    gold set as `labels`.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if labels is None:
        v = SentimentAnalyzer("vader")
        labels = [v.analyze(t)["label"] for t in texts]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(texts, labels)
    joblib.dump(model, ML_SENTIMENT_PATH)
    print(f"[sentiment] trained ML model ({len(set(labels))} classes) -> {ML_SENTIMENT_PATH}")
    return model


def agreement(texts: list[str]) -> dict:
    """VADER-vs-transformer agreement, for the comparison table."""
    from sklearn.metrics import cohen_kappa_score

    v = SentimentAnalyzer("vader")
    t = SentimentAnalyzer("transformer")
    # transformer model is 2-way; collapse VADER neutral->nearest for fair compare.
    vl, tl = [], []
    for txt in texts:
        a = v.analyze(txt)["label"]
        b = t.analyze(txt)["label"]
        vl.append("positive" if a == "positive" else "negative")
        tl.append(b)
    agree = sum(x == y for x, y in zip(vl, tl)) / max(len(texts), 1)
    return {"agreement": agree, "cohen_kappa": cohen_kappa_score(vl, tl)}


if __name__ == "__main__":
    sa = SentimentAnalyzer()
    for m in ["This is unacceptable, I'm furious!", "Thanks, that was super helpful!", "Where is my order?"]:
        print(m, "->", sa.analyze(m))
