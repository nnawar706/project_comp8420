#!/usr/bin/env python
"""
Run the evaluation suite and write tables/plots to evaluation/.

By default it runs the offline-safe parts (classification, NER, sentiment).
Add --llm to also run the RAG and prompt-variant ablations (needs Ollama running).

Usage:
  python scripts/evaluate.py            # classical metrics only
  python scripts/evaluate.py --llm      # + RAG/prompt ablations via Ollama
  python scripts/evaluate.py --llm 40   # limit LLM eval to 40 messages
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import data_loader, evaluation  # noqa: E402


def main() -> None:
    do_llm = "--llm" in sys.argv
    n_llm = next((int(a) for a in sys.argv if a.isdigit()), 30)

    train, _, test = data_loader.load_splits()
    results: dict = {}

    # --- Classification ---------------------------------------------------- #
    from src.classification import TextClassifier, evaluate as eval_clf

    clf = TextClassifier.load()
    cm = eval_clf(clf, test[config.TEXT_COL], test[config.CATEGORY_COL])
    results["classification"] = {"accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"]}
    evaluation.plot_confusion_matrix(cm["confusion_matrix"], cm["labels"])
    evaluation.plot_per_class_f1(cm["report"])
    print(f"[eval] classification  acc={cm['accuracy']:.3f}  macroF1={cm['macro_f1']:.3f}")

    # --- NER (silver labels from placeholders) ----------------------------- #
    try:
        from src.ner import EntityExtractor, evaluate_against_placeholders

        ner = EntityExtractor()
        ner_m = evaluate_against_placeholders(ner, test, config.TEXT_COL, limit=400)
        results["ner"] = ner_m
        print(f"[eval] NER  P={ner_m['precision']:.3f}  R={ner_m['recall']:.3f}  F1={ner_m['f1']:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] NER skipped: {exc}")

    # --- Sentiment distribution by category -------------------------------- #
    try:
        from src.sentiment import SentimentAnalyzer

        sa = SentimentAnalyzer("vader")
        sample = test.sample(min(500, len(test)), random_state=config.RANDOM_STATE).copy()
        sample["sentiment"] = [sa.analyze(t)["label"] for t in sample[config.TEXT_COL]]
        evaluation.plot_sentiment_by_category(sample, "sentiment", config.CATEGORY_COL)
        results["sentiment_distribution"] = (
            sample.groupby(config.CATEGORY_COL)["sentiment"].value_counts().unstack(fill_value=0).to_dict()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] sentiment plot skipped: {exc}")

    # --- LLM / RAG ablations (optional) ------------------------------------ #
    if do_llm:
        from src.generation import ollama_available
        from src.rag import RAGStore

        if not ollama_available():
            print("[eval] --llm requested but Ollama is offline; skipping LLM eval.")
        else:
            store = RAGStore()
            msgs = test[config.TEXT_COL].sample(n_llm, random_state=config.RANDOM_STATE).tolist()
            from src.preprocessing import clean_for_llm
            msgs = [clean_for_llm(m) for m in msgs]

            print(f"[eval] running RAG ablation on {len(msgs)} messages (LLM judge) ...")
            abl = evaluation.rag_ablation(msgs, store)
            results["rag_ablation"] = {k: v for k, v in abl.items() if k != "rows"}
            evaluation.plot_rag_ablation(abl)
            print("       RAG-on :", results["rag_ablation"]["rag_on"])
            print("       RAG-off:", results["rag_ablation"]["rag_off"])

            print("[eval] running prompt-variant ablation ...")
            results["prompt_ablation"] = evaluation.prompt_variant_ablation(msgs[:max(10, n_llm // 2)], store)
            for v, m in results["prompt_ablation"].items():
                print(f"       {v}: {m}")

    out = config.EVAL_DIR / "evaluation_summary.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[eval] summary -> {out}")
    print("[eval] figures -> evaluation/figures/")


if __name__ == "__main__":
    main()
