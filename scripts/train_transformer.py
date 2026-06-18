"""
Fine-tune DistilBERT on the Bitext `category` task and compare it head-to-head
with the classical TF-IDF + Logistic Regression baseline.

Run on a GPU (Colab recommended):
    pip install torch transformers datasets
    python scripts/train_transformer.py --epochs 3

Outputs:
    models/distilbert_classifier/        fine-tuned model + tokenizer + label map
    evaluation/transformer_vs_classical.json   side-by-side metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src import transformer_classifier as TC
from src.classification import TextClassifier, evaluate
from src.data_loader import load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--subset", type=int, default=0,
                    help="train on only N examples (smoke test on CPU); 0 = full")
    args = ap.parse_args()

    if not TC.dependencies_available():
        sys.exit("Install torch transformers datasets first (and use a GPU).")

    train_df, val_df, test_df = load_splits()
    if args.subset:
        train_df = train_df.sample(args.subset, random_state=config.RANDOM_STATE)

    # 1) fine-tune
    TC.train(train_df, val_df, epochs=args.epochs, batch_size=args.batch_size)

    # 2) evaluate both models on the SAME held-out test set
    X_test = test_df[config.TEXT_COL].astype(str).tolist()
    y_test = test_df[config.CATEGORY_COL].tolist()

    neural = TC.TransformerClassifier()
    classical = TextClassifier.load()

    res = {
        "classical_tfidf_logreg": {k: evaluate(classical, X_test, y_test)[k]
                                   for k in ("accuracy", "macro_f1")},
        "distilbert_finetuned": {k: evaluate(neural, X_test, y_test)[k]
                                 for k in ("accuracy", "macro_f1")},
    }
    out = config.EVAL_DIR / "transformer_vs_classical.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[compare] saved -> {out}")


if __name__ == "__main__":
    main()
