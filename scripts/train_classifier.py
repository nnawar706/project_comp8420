#!/usr/bin/env python
"""Train the TF-IDF + Logistic Regression classifier; save model, metrics and plots."""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data_loader, evaluation  # noqa: E402
from src.classification import TextClassifier, evaluate  # noqa: E402


def main(kind: str = "logreg") -> None:
    train, val, test = data_loader.load_splits()
    X_tr, y_tr = train[config.TEXT_COL], train[config.CATEGORY_COL]

    clf = TextClassifier().fit(X_tr, y_tr, kind=kind)
    clf.save()

    metrics = evaluate(clf, test[config.TEXT_COL], test[config.CATEGORY_COL])
    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}\n")
    print(metrics["report_text"])

    evaluation.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"])
    evaluation.plot_per_class_f1(metrics["report"])

    out = config.EVAL_DIR / "classification_metrics.json"
    out.write_text(json.dumps(
        {k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")}, indent=2))
    print(f"\nSaved metrics -> {out}")

    top = clf.top_features(n=12)
    if top:
        print("\nTop TF-IDF features per class (first 3 classes):")
        for cls in list(top)[:3]:
            print(f"  {cls}: {', '.join(top[cls][:8])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logreg")
