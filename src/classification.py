"""
Traditional text classification: TF-IDF + linear classifier.

This is the classical baseline the LLM pipeline must beat. We use Logistic
Regression by default because it gives calibrated-ish probabilities, which we
reuse as a *confidence* signal for escalation gating in the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import clean_for_classical  # noqa: E402


def build_pipeline(kind: str = "logreg"):
    """TfidfVectorizer (uni+bigrams, min_df=2) -> linear classifier.

    kind: 'logreg' (default, gives probabilities), 'svm' (LinearSVC), or
    'nb' (Multinomial Naive Bayes - the classic fast text-classification baseline).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    vectorizer = TfidfVectorizer(
        preprocessor=clean_for_classical,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    if kind == "svm":
        clf = LinearSVC(C=1.0)
    elif kind == "nb":
        clf = MultinomialNB()
    elif kind == "mlp":
        # A genuine feed-forward neural network (one hidden layer) trained on the
        # TF-IDF features. This is the deep-learning baseline we can train end-to-
        # end on CPU here; it also exposes predict_proba for the calibration study.
        clf = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                            alpha=1e-4, max_iter=25, n_iter_no_change=4,
                            random_state=42)
    else:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


class TextClassifier:
    """Thin wrapper that always exposes a (label, confidence) interface."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.classes_ = None if pipeline is None else getattr(pipeline, "classes_", None)

    # --- training / persistence ------------------------------------------- #
    def fit(self, X, y, kind: str = "logreg") -> "TextClassifier":
        self.pipeline = build_pipeline(kind)
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def save(self, path: Path = config.CLASSIFIER_PATH) -> None:
        joblib.dump(self.pipeline, path)
        print(f"[clf] saved -> {path}")

    @classmethod
    def load(cls, path: Path = config.CLASSIFIER_PATH) -> "TextClassifier":
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/train_classifier.py first.")
        return cls(joblib.load(path))

    # --- inference --------------------------------------------------------- #
    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted_category, confidence in [0,1])."""
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba([text])[0]
            idx = int(np.argmax(probs))
            return str(self.pipeline.classes_[idx]), float(probs[idx])
        # LinearSVC: convert decision margins to a softmax-like confidence.
        scores = self.pipeline.decision_function([text])[0]
        scores = np.atleast_1d(scores)
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()
        idx = int(np.argmax(probs))
        return str(self.pipeline.classes_[idx]), float(probs[idx])

    def predict_batch(self, texts: list[str]) -> list[str]:
        return list(self.pipeline.predict(texts))

    # --- interpretability -------------------------------------------------- #
    def top_features(self, n: int = 15) -> dict[str, list[str]]:
        """Top-n TF-IDF features per class (only for linear models with coef_)."""
        clf = self.pipeline.named_steps["clf"]
        vocab = np.array(self.pipeline.named_steps["tfidf"].get_feature_names_out())
        coefs = getattr(clf, "coef_", None)
        if coefs is None:
            return {}
        out = {}
        for i, cls in enumerate(self.pipeline.classes_):
            row = coefs[i] if coefs.shape[0] > 1 else coefs[0]
            out[str(cls)] = list(vocab[np.argsort(row)[-n:][::-1]])
        return out


def evaluate(clf: TextClassifier, X_test, y_test) -> dict:
    """Return accuracy, macro-F1, full per-class report and confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    preds = clf.predict_batch(list(X_test))
    labels = sorted(pd.unique(y_test))
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "report_text": classification_report(y_test, preds, zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
        "y_true": list(y_test),
        "y_pred": preds,
    }


# --------------------------------------------------------------------------- #
# Optional: fine-tuned DistilBERT baseline (stretch goal; lazily imported)
# --------------------------------------------------------------------------- #
def train_transformer_baseline(train_df, val_df, epochs: int = 2, out_dir: str | None = None):
    """
    Optional neural baseline to show the classical-vs-neural gap.
    Requires `transformers`, `torch`, `datasets`. Heavy - run on a GPU/Colab.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    labels = sorted(train_df[config.CATEGORY_COL].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(df):
        ds = Dataset.from_dict({
            "text": df[config.TEXT_COL].tolist(),
            "label": [label2id[l] for l in df[config.CATEGORY_COL]],
        })
        return ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                                    max_length=128), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels),
        id2label={i: l for l, i in label2id.items()}, label2id=label2id,
    )
    out_dir = out_dir or str(config.MODELS_DIR / "distilbert_classifier")
    args = TrainingArguments(output_dir=out_dir, num_train_epochs=epochs,
                             per_device_train_batch_size=16, eval_strategy="epoch",
                             logging_steps=50, report_to=[])
    trainer = Trainer(model=model, args=args,
                      train_dataset=encode(train_df), eval_dataset=encode(val_df))
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[clf] transformer baseline saved -> {out_dir}")
    return out_dir
