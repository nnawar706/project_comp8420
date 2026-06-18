"""
Evaluation layer.

Produces the quantitative results the report needs:
  * Classification: accuracy, macro-F1, per-class report, confusion-matrix plot.
  * NER: entity-level P/R/F1 against Bitext placeholder silver labels.
  * Sentiment: distribution-by-category plot (+ optional gold-set scoring).
  * RAG/LLM: schema-validity rate, RAG-on vs RAG-off ablation, prompt-variant
    ablation, and LLM-as-judge faithfulness/answer-relevancy scores.

The LLM-as-judge uses the same local Ollama model, so the whole thing runs free
and offline. (RAGAS is documented in the README as an optional drop-in.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import generation  # noqa: E402


# --------------------------------------------------------------------------- #
# Classification plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, labels, path: Path = config.FIG_DIR / "confusion_matrix.png"):
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7),) * 2)
    cm = np.array(cm)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[eval] saved {path}")
    return path


def plot_per_class_f1(report: dict, path: Path = config.FIG_DIR / "per_class_f1.png"):
    classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    f1s = [report[c]["f1-score"] for c in classes]
    order = sorted(range(len(classes)), key=lambda i: f1s[i])
    fig, ax = plt.subplots(figsize=(7, max(3, len(classes) * 0.35)))
    ax.barh([classes[i] for i in order], [f1s[i] for i in order], color="#2a6f97")
    ax.set_xlabel("F1"); ax.set_xlim(0, 1); ax.set_title("Per-class F1")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[eval] saved {path}")
    return path


def plot_sentiment_by_category(df, sentiment_col: str, category_col: str,
                               path: Path = config.FIG_DIR / "sentiment_by_category.png"):
    import pandas as pd

    ct = pd.crosstab(df[category_col], df[sentiment_col], normalize="index")
    ax = ct.plot(kind="bar", stacked=True, figsize=(9, 5),
                 color={"negative": "#d62828", "neutral": "#bbbbbb", "positive": "#2a9d8f"})
    ax.set_ylabel("proportion"); ax.set_title("Sentiment distribution by category")
    ax.legend(title="sentiment", bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"[eval] saved {path}")
    return path


# --------------------------------------------------------------------------- #
# LLM-as-judge (offline, via Ollama)
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are a strict evaluator. Given a customer question, the company knowledge
that was retrieved, and an assistant's answer, rate the answer on two axes from 0 to 1:
- faithfulness: is every claim in the answer supported by the company knowledge? (1 = fully grounded, 0 = hallucinated)
- answer_relevancy: does the answer actually address the customer's question? (1 = directly, 0 = off-topic)

Question: {q}
Company knowledge: {ctx}
Answer: {ans}

Respond with ONLY JSON: {{"faithfulness": <0..1>, "answer_relevancy": <0..1>}}"""


def llm_judge(question: str, context: str, answer: str) -> dict:
    prompt = _JUDGE_PROMPT.format(q=question, ctx=context, ans=answer)
    try:
        raw = generation.call_ollama(prompt, temperature=0.0)
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return {"faithfulness": float(data.get("faithfulness", 0.0)),
                "answer_relevancy": float(data.get("answer_relevancy", 0.0))}
    except Exception:  # noqa: BLE001
        return {"faithfulness": None, "answer_relevancy": None}


# --------------------------------------------------------------------------- #
# RAG-on vs RAG-off ablation + prompt-variant ablation
# --------------------------------------------------------------------------- #
def rag_ablation(messages: list[str], rag_store, n: int | None = None) -> dict:
    """Compare grounded (RAG-on) vs ungrounded (RAG-off) answers with the LLM judge."""
    from src.rag import format_context

    messages = messages[: n] if n else messages
    rows = []
    for msg in messages:
        chunks = rag_store.retrieve(msg) if rag_store else []
        ctx_on = format_context(chunks)
        ans_on = generation.generate_response(msg, ctx_on, variant="few_shot")["reply"]
        ans_off = generation.generate_response(msg, "(no company knowledge provided)",
                                               variant="few_shot")["reply"]
        rows.append({
            "message": msg,
            "rag_on": llm_judge(msg, ctx_on, ans_on),
            "rag_off": llm_judge(msg, ctx_on, ans_off),  # judged against true context
        })
    return _summarise_ablation(rows)


def _summarise_ablation(rows: list[dict]) -> dict:
    def avg(side, key):
        vals = [r[side][key] for r in rows if r[side].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": len(rows),
        "rag_on": {"faithfulness": avg("rag_on", "faithfulness"),
                   "answer_relevancy": avg("rag_on", "answer_relevancy")},
        "rag_off": {"faithfulness": avg("rag_off", "faithfulness"),
                    "answer_relevancy": avg("rag_off", "answer_relevancy")},
        "rows": rows,
    }


def prompt_variant_ablation(messages: list[str], rag_store) -> dict:
    """Compare zero-shot / few-shot / few-shot+CoT on schema-validity + judge scores."""
    from src.rag import format_context

    variants = ["zero_shot", "few_shot", "few_shot_cot"]
    out = {}
    for v in variants:
        valid, faith, rel = 0, [], []
        for msg in messages:
            ctx = format_context(rag_store.retrieve(msg)) if rag_store else "(none)"
            r = generation.generate_response(msg, ctx, variant=v)
            valid += int(r.get("schema_valid", False))
            j = llm_judge(msg, ctx, r["reply"])
            if j["faithfulness"] is not None:
                faith.append(j["faithfulness"]); rel.append(j["answer_relevancy"])
        out[v] = {
            "schema_valid_rate": round(valid / max(len(messages), 1), 3),
            "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "answer_relevancy": round(sum(rel) / len(rel), 4) if rel else None,
        }
    return out


def plot_rag_ablation(summary: dict, path: Path = config.FIG_DIR / "rag_ablation.png"):
    metrics = ["faithfulness", "answer_relevancy"]
    on = [summary["rag_on"][m] or 0 for m in metrics]
    off = [summary["rag_off"][m] or 0 for m in metrics]
    import numpy as np
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, off, w, label="RAG-off", color="#e76f51")
    ax.bar(x + w / 2, on, w, label="RAG-on", color="#2a9d8f")
    ax.set_xticks(x, metrics); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("RAG ablation (LLM-as-judge)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[eval] saved {path}")
    return path


# --------------------------------------------------------------------------- #
# Confidence calibration  (justifies the escalation threshold quantitatively)
# --------------------------------------------------------------------------- #
# The classifier scores ~99.8% on Bitext's held-out test set. That number is
# inflated because Bitext is template-generated, so the *interesting* question is
# not "is it accurate?" but "is its CONFIDENCE trustworthy enough to gate
# automated escalation?". The functions below answer that: a reliability diagram
# + Expected Calibration Error (ECE) measure whether a stated confidence of, say,
# 0.6 really corresponds to a 60% chance of being right -- which is exactly the
# assumption CLASSIFIER_CONF_THRESHOLD relies on.
def expected_calibration_error(confidences, correct, n_bins: int = 10) -> dict:
    """Compute ECE, MCE and per-bin accuracy/confidence (equal-width bins)."""
    import numpy as np

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            in_bin = (confidences > lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi + 1e-9)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "acc": None, "conf": None, "gap": None})
            continue
        acc = float(correct[in_bin].mean())
        conf = float(confidences[in_bin].mean())
        gap = abs(acc - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append({"lo": float(lo), "hi": float(hi), "count": count,
                     "acc": acc, "conf": conf, "gap": gap})
    return {"ece": round(ece, 4), "mce": round(mce, 4), "n_bins": n_bins, "bins": rows}


def confidence_report(clf, X_test, y_test, n_bins: int = 10) -> dict:
    """Run the classifier over the test set and quantify how calibrated it is."""
    import numpy as np

    pipe = getattr(clf, "pipeline", clf)
    classes = list(pipe.classes_)
    probs = pipe.predict_proba(list(X_test))
    pred_idx = probs.argmax(1)
    preds = [classes[i] for i in pred_idx]
    conf = probs.max(1)
    y_true = list(y_test)
    correct = np.array([int(p == t) for p, t in zip(preds, y_true)])

    onehot = np.zeros_like(probs)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    for r, t in enumerate(y_true):
        if t in cls_to_i:
            onehot[r, cls_to_i[t]] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    cal = expected_calibration_error(conf, correct, n_bins=n_bins)
    return {
        "accuracy": float(correct.mean()),
        "brier": round(brier, 4),
        "ece": cal["ece"],
        "mce": cal["mce"],
        "calibration": cal,
        "confidences": conf.tolist(),
        "correct": correct.tolist(),
        "mean_confidence": round(float(conf.mean()), 4),
    }


def plot_reliability_diagram(cal: dict, path: Path = config.FIG_DIR / "calibration_reliability.png"):
    """Reliability diagram: bin accuracy vs bin confidence (the diagonal = perfect)."""
    bins = [b for b in cal["calibration"]["bins"] if b["count"] > 0]
    confs = [b["conf"] for b in bins]
    accs = [b["acc"] for b in bins]
    counts = [b["count"] for b in bins]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6),
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    ax.plot(confs, accs, "o-", color="#2a6f97", label="classifier")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability diagram  (ECE={cal['ece']}, Brier={cal.get('brier','-')})")
    ax.legend(loc="upper left", fontsize=8)
    ax2.bar(confs, counts, width=0.06, color="#bbbbbb")
    ax2.set_xlim(0, 1); ax2.set_xlabel("confidence"); ax2.set_ylabel("# examples")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[eval] saved {path}")
    return path


# --------------------------------------------------------------------------- #
# Escalation-decision evaluation  (does low confidence catch out-of-scope msgs?)
# --------------------------------------------------------------------------- #
def _default_ood_probes() -> list:
    """Out-of-scope messages a retail support bot should *not* try to answer."""
    return [
        "What's the weather forecast in Paris this weekend?",
        "Can you help me write a poem about the ocean?",
        "Which stock should I invest my savings in right now?",
        "My doctor prescribed a new medication, is it safe to take with alcohol?",
        "I'd like to dispute this charge through my lawyer and take legal action.",
        "Translate this paragraph into German for my school essay.",
        "What is the meaning of life?",
        "Can you recommend a good restaurant near the Eiffel Tower?",
        "I think one of your employees was extremely rude and I want them fired.",
        "Help me solve this calculus integral for my homework.",
        "What are your thoughts on the upcoming election?",
        "Can I get your CEO's personal phone number?",
        "How do I hack into someone else's account?",
        "Write me a cover letter for a marketing job.",
        "Is it going to rain tomorrow where I live?",
    ]


def escalation_eval(clf, X_in_dist, threshold: float = config.CLASSIFIER_CONF_THRESHOLD,
                    ood_probes=None, sample_in_dist: int = 400) -> dict:
    """Score the confidence-based escalation gate as a binary OOD detector."""
    import numpy as np
    from sklearn.metrics import (average_precision_score, f1_score,
                                  precision_score, recall_score, roc_auc_score)

    pipe = getattr(clf, "pipeline", clf)
    ood = ood_probes or _default_ood_probes()

    X_in = list(X_in_dist)
    if sample_in_dist and len(X_in) > sample_in_dist:
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(len(X_in), size=sample_in_dist, replace=False)
        X_in = [X_in[i] for i in idx]

    conf_in = pipe.predict_proba(X_in).max(1)
    conf_ood = pipe.predict_proba(ood).max(1)

    confidences = np.concatenate([conf_in, conf_ood])
    scores = 1.0 - confidences
    y_true = np.array([0] * len(conf_in) + [1] * len(conf_ood))
    y_pred = (confidences < threshold).astype(int)

    return {
        "threshold": threshold,
        "n_in_dist": len(conf_in),
        "n_ood": len(conf_ood),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "average_precision": round(float(average_precision_score(y_true, scores)), 4),
        "precision_at_threshold": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_at_threshold": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mean_conf_in_dist": round(float(conf_in.mean()), 4),
        "mean_conf_ood": round(float(conf_ood.mean()), 4),
        "conf_in_dist": conf_in.tolist(),
        "conf_ood": conf_ood.tolist(),
        "y_true": y_true.tolist(),
        "scores": scores.tolist(),
    }


def plot_escalation_analysis(esc: dict, path: Path = config.FIG_DIR / "escalation_analysis.png"):
    """Two panels: confidence distributions (in-dist vs OOD) and the ROC curve."""
    import numpy as np
    from sklearn.metrics import roc_curve

    fig, (axh, axr) = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 21)
    axh.hist(esc["conf_in_dist"], bins=bins, alpha=0.7, color="#2a9d8f",
             label=f"in-domain (mean {esc['mean_conf_in_dist']})")
    axh.hist(esc["conf_ood"], bins=bins, alpha=0.7, color="#e76f51",
             label=f"out-of-scope (mean {esc['mean_conf_ood']})")
    axh.axvline(esc["threshold"], color="black", ls="--",
                label=f"escalate < {esc['threshold']}")
    axh.set_xlabel("classifier confidence"); axh.set_ylabel("# messages")
    axh.set_title("Confidence separates in-domain from out-of-scope")
    axh.legend(fontsize=8)

    fpr, tpr, _ = roc_curve(esc["y_true"], esc["scores"])
    axr.plot(fpr, tpr, color="#2a6f97", label=f"ROC (AUC={esc['roc_auc']})")
    axr.plot([0, 1], [0, 1], "--", color="#888")
    axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
    axr.set_title("Escalation gate as an OOD detector"); axr.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[eval] saved {path}")
    return path
