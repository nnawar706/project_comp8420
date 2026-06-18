"""
Fine-tuned transformer text classifier (DistilBERT) - the neural counterpart to
the classical TF-IDF + Logistic Regression baseline in `classification.py`.

WHY THIS EXISTS
---------------
A fine-tuned transformer is a stronger baseline than just calling a pretrained
model off the shelf, and it lets us compare a contextual model against the
classical bag-of-words approach. This module fine-tunes DistilBERT on the
Bitext `category` task and exposes the SAME (label, confidence) inference
interface as `TextClassifier`, so the evaluation notebook can score both models
with identical code (accuracy, macro-F1, calibration, escalation).

WHAT TO EXPECT
--------------
On Bitext the classical baseline already hits ~99.8% because the data is
template-generated, so a fine-tuned DistilBERT will *match* it on the in-domain
test set rather than beat it - the meaningful comparison is on the robustness /
out-of-scope splits, where the contextual model should degrade more gracefully
than bag-of-words. That contrast is the analytical point worth writing up.

RUNTIME
-------
Fine-tuning needs `torch`, `transformers`, `datasets` and downloads the
pretrained DistilBERT weights from HuggingFace. Run it on a GPU (Google Colab is
fine) - a couple of epochs over the Bitext splits takes only a few minutes there.
Everything degrades gracefully: if the dependencies or a trained model are
missing, the helpers raise a clear, actionable error instead of crashing a
notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MODEL_DIR = config.MODELS_DIR / "distilbert_classifier"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128


def dependencies_available() -> bool:
    """True iff torch + transformers + datasets can be imported."""
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("torch", "transformers", "datasets"))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(train_df, val_df, epochs: int = 3, batch_size: int = 16,
          lr: float = 5e-5, out_dir: Path = MODEL_DIR, max_len: int = MAX_LEN) -> Path:
    """Fine-tune DistilBERT on the `category` task and save model + label map.

    train_df / val_df: DataFrames with config.TEXT_COL and config.CATEGORY_COL.
    Returns the directory the fine-tuned model was saved to.
    """
    if not dependencies_available():
        raise RuntimeError(
            "transformer deps missing. Install with:\n"
            "  pip install torch transformers datasets\n"
            "and run on a GPU (Colab recommended)."
        )
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    labels = sorted(train_df[config.CATEGORY_COL].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode(df):
        ds = Dataset.from_dict({
            "text": df[config.TEXT_COL].astype(str).tolist(),
            "label": [label2id[l] for l in df[config.CATEGORY_COL]],
        })
        return ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length",
                                    max_length=max_len), batched=True)

    train_ds, val_ds = encode(train_df), encode(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(gold, preds),
                "macro_f1": f1_score(gold, preds, average="macro")}

    out_dir = Path(out_dir)
    # `eval_strategy` is the current name; fall back to `evaluation_strategy`
    # on older transformers versions.
    try:
        args = TrainingArguments(
            output_dir=str(out_dir), num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
            learning_rate=lr, eval_strategy="epoch", save_strategy="no",
            logging_steps=50, report_to=[], seed=config.RANDOM_STATE)
    except TypeError:
        args = TrainingArguments(
            output_dir=str(out_dir), num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=32,
            learning_rate=lr, evaluation_strategy="epoch", save_strategy="no",
            logging_steps=50, report_to=[], seed=config.RANDOM_STATE)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(f"[distilbert] val metrics: {metrics}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "label_map.json").write_text(json.dumps(id2label, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[distilbert] saved -> {out_dir}")
    return out_dir


# --------------------------------------------------------------------------- #
# Inference wrapper - mirrors classification.TextClassifier
# --------------------------------------------------------------------------- #
class TransformerClassifier:
    """Loads a fine-tuned DistilBERT and exposes predict / predict_batch like the
    classical TextClassifier, so evaluation code is model-agnostic."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        if not dependencies_available():
            raise RuntimeError("transformer deps missing (pip install torch transformers).")
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"{model_dir} not found. Train it first:\n"
                "  python scripts/train_transformer.py")
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        self.model.eval()
        id2label = json.loads((Path(model_dir) / "label_map.json").read_text())
        self.classes_ = [id2label[str(i)] for i in range(len(id2label))]

    def _logits(self, texts: list[str]):
        enc = self.tok(texts, truncation=True, padding=True, max_length=MAX_LEN,
                       return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model(**enc).logits

    def predict(self, text: str):
        probs = self.torch.softmax(self._logits([text]), dim=1)[0]
        idx = int(probs.argmax())
        return self.classes_[idx], float(probs[idx])

    def predict_batch(self, texts: list[str], batch_size: int = 64) -> list[str]:
        out = []
        for i in range(0, len(texts), batch_size):
            logits = self._logits(list(texts[i:i + batch_size]))
            out.extend(self.classes_[int(j)] for j in logits.argmax(1))
        return out

    def predict_proba(self, texts: list[str], batch_size: int = 64):
        import numpy as np
        rows = []
        for i in range(0, len(texts), batch_size):
            logits = self._logits(list(texts[i:i + batch_size]))
            rows.append(self.torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(rows)


if __name__ == "__main__":
    print("transformer deps available:", dependencies_available())
    print("To fine-tune: python scripts/train_transformer.py")
