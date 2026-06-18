"""
Robustness / generalisation stress-test.

Bitext is template-generated, so the in-domain test accuracy (~99.8%) overstates
how the model will behave on messages real customers actually type - full of
typos, missing punctuation, shouting, and paraphrase. This module perturbs the
test messages with increasing amounts of realistic noise and measures how the
classifier's accuracy, macro-F1 and *calibration* degrade. The gap between clean
and noisy performance is the honest measure of real-world readiness, and it is a
core piece of "deep analysis" for the report.

No external dependencies beyond numpy / scikit-learn.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

_KEYBOARD_NEIGHBOURS = {
    "a": "sq", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh",
    "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm",
    "o": "ip", "p": "o", "q": "wa", "r": "et", "s": "ad", "t": "ry", "u": "yi",
    "v": "cb", "w": "qe", "x": "zc", "y": "tu", "z": "x",
}


def _typo_word(word: str, rng: random.Random) -> str:
    """Apply one random character-level corruption to a word."""
    if len(word) < 3:
        return word
    op = rng.choice(["swap", "drop", "dup", "key"])
    i = rng.randrange(len(word))
    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    if op == "key":
        c = word[i].lower()
        if c in _KEYBOARD_NEIGHBOURS:
            repl = rng.choice(_KEYBOARD_NEIGHBOURS[c])
            return word[:i] + repl + word[i + 1:]
    return word


def perturb(text: str, level: float, seed: int = 0) -> str:
    """Corrupt `text` with intensity `level` in [0,1].

    Combines per-word typos (probability ~ level), random casing, and dropped
    punctuation - the kinds of noise a bag-of-words model is most brittle to.
    """
    rng = random.Random(hash((text, seed)) & 0xFFFFFFFF)
    words = text.split()
    out = []
    for w in words:
        if rng.random() < level:
            w = _typo_word(w, rng)
        if rng.random() < level * 0.5:
            w = w.upper() if rng.random() < 0.5 else w.lower()
        out.append(w)
    s = " ".join(out)
    if level > 0:
        s = re.sub(r"[.,!?]", lambda m: "" if rng.random() < level else m.group(0), s)
    return s


def evaluate_robustness(clf, X_test, y_test, levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                        seed: int = 0) -> dict:
    """Accuracy + macro-F1 of `clf` as noise increases. `clf` exposes predict_batch."""
    from sklearn.metrics import accuracy_score, f1_score

    X_test = list(X_test); y_test = list(y_test)
    rows = []
    for lv in levels:
        Xp = [perturb(t, lv, seed=seed) for t in X_test] if lv > 0 else X_test
        preds = clf.predict_batch(Xp)
        rows.append({"level": lv,
                     "accuracy": round(accuracy_score(y_test, preds), 4),
                     "macro_f1": round(f1_score(y_test, preds, average="macro"), 4)})
    return {"levels": list(levels), "curve": rows}


def compare_models_robustness(models: dict, X_test, y_test,
                              levels=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5), seed: int = 0) -> dict:
    """Run evaluate_robustness for several named classifiers -> {name: curve}."""
    return {name: evaluate_robustness(clf, X_test, y_test, levels, seed)["curve"]
            for name, clf in models.items()}


def plot_robustness(results: dict, metric: str = "accuracy",
                    path: Path = config.FIG_DIR / "robustness_curve.png"):
    """Line plot of `metric` vs noise level for each model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in results.items():
        xs = [r["level"] for r in curve]
        ys = [r[metric] for r in curve]
        ax.plot(xs, ys, "o-", label=name)
    ax.set_xlabel("noise level (fraction of words corrupted)")
    ax.set_ylabel(metric); ax.set_ylim(0, 1.02)
    ax.set_title(f"Robustness to input noise - {metric}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[robustness] saved {path}")
    return path
