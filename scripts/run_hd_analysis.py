"""
HD analysis driver: trains a neural (MLP) model, compares all classifiers under
input noise (robustness), and runs the harder 27-way intent task. Saves metrics
to evaluation/ and figures to evaluation/figures/. Stage-selectable so it fits
modest machines:  python scripts/run_hd_analysis.py [train|robust|intent|all]
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import config
from src.classification import TextClassifier, build_pipeline, evaluate
from src import robustness as R

TRAIN_SUB = 9000   # subsample for fast CPU training (full set also works, slower)
MODELS = {}

def _load_splits():
    tr = pd.read_parquet(config.TRAIN_PATH); te = pd.read_parquet(config.TEST_PATH)
    return tr, te

def stage_train():
    tr, te = _load_splits()
    sub = tr.sample(min(TRAIN_SUB, len(tr)), random_state=config.RANDOM_STATE)
    Xtr, ytr = sub[config.TEXT_COL].astype(str).tolist(), sub[config.CATEGORY_COL].tolist()
    Xte, yte = te[config.TEXT_COL].astype(str).tolist(), te[config.CATEGORY_COL].tolist()
    clean = {}
    for kind in ["logreg", "nb", "svm", "mlp"]:
        t0 = time.time()
        clf = TextClassifier().fit(Xtr, ytr, kind=kind)
        ev = evaluate(clf, Xte, yte)
        clean[kind] = {"accuracy": round(ev["accuracy"], 4), "macro_f1": round(ev["macro_f1"], 4)}
        import joblib; joblib.dump(clf.pipeline, config.MODELS_DIR / f"clf_{kind}.joblib")
        print(f"  trained {kind:7s} acc={clean[kind]['accuracy']} f1={clean[kind]['macro_f1']} ({time.time()-t0:.1f}s)")
    (config.EVAL_DIR / "clean_model_comparison.json").write_text(json.dumps(clean, indent=2))
    print("[train] saved clean_model_comparison.json")

def stage_robust():
    import joblib
    te = pd.read_parquet(config.TEST_PATH)
    Xte, yte = te[config.TEXT_COL].astype(str).tolist(), te[config.CATEGORY_COL].tolist()
    models = {}
    for name, kind in [("LogReg", "logreg"), ("NaiveBayes", "nb"), ("LinearSVM", "svm"), ("MLP (neural)", "mlp")]:
        pth = config.MODELS_DIR / f"clf_{kind}.joblib"
        if pth.exists():
            models[name] = TextClassifier(joblib.load(pth))
    res = R.compare_models_robustness(models, Xte, yte)
    (config.EVAL_DIR / "robustness_results.json").write_text(json.dumps(res, indent=2))
    R.plot_robustness(res, "accuracy", config.FIG_DIR / "robustness_accuracy.png")
    R.plot_robustness(res, "macro_f1", config.FIG_DIR / "robustness_macro_f1.png")
    print("[robust] saved robustness_results.json + 2 figures")
    for name, curve in res.items():
        print(f"  {name:14s} acc@0.0={curve[0]['accuracy']}  acc@0.3={curve[3]['accuracy']}  acc@0.5={curve[5]['accuracy']}")

def stage_intent():
    tr, te = _load_splits()
    sub = tr.sample(min(TRAIN_SUB, len(tr)), random_state=config.RANDOM_STATE)
    Xtr, ytr = sub[config.TEXT_COL].astype(str).tolist(), sub[config.INTENT_COL].tolist()
    Xte, yte = te[config.TEXT_COL].astype(str).tolist(), te[config.INTENT_COL].tolist()
    clf = TextClassifier().fit(Xtr, ytr, kind="logreg")
    ev = evaluate(clf, Xte, yte)
    out = {"task": "27-way intent (fine label)",
           "n_classes": int(te[config.INTENT_COL].nunique()),
           "accuracy": round(ev["accuracy"], 4),
           "macro_f1": round(ev["macro_f1"], 4)}
    (config.EVAL_DIR / "intent_27way_metrics.json").write_text(
        json.dumps({**out, "report": ev["report"]}, indent=2))
    # per-class F1 figure (sorted)
    from src.evaluation import plot_per_class_f1
    plot_per_class_f1(ev["report"], config.FIG_DIR / "intent_per_class_f1.png")
    print(f"[intent] {out['n_classes']}-way intent: acc={out['accuracy']} macro_f1={out['macro_f1']}")
    print("[intent] saved intent_27way_metrics.json + intent_per_class_f1.png")

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("train", "all"): stage_train()
    if stage in ("robust", "all"): stage_robust()
    if stage in ("intent", "all"): stage_intent()
