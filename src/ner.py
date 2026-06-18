"""
Named Entity Recognition - a hybrid of spaCy's statistical NER (PERSON, ORG,
PRODUCT, GPE, DATE, MONEY...) and rule-based extraction for domain entities
(ORDER_NUMBER, INVOICE_NUMBER, TRACKING_NUMBER, EMAIL, PHONE).

The hybrid is both accurate and explainable, and the extracted facts directly
populate the RAG query and the response template.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import PATTERNS, find_pii  # noqa: E402

# EntityRuler patterns for entities spaCy won't know about.
_RULER_PATTERNS = [
    {"label": "ORDER_NUMBER", "pattern": [{"LOWER": {"IN": ["order", "ord"]}},
                                          {"TEXT": {"REGEX": "^[#:]?$"}, "OP": "?"},
                                          {"TEXT": {"REGEX": r"^[A-Za-z]{0,3}\d{3,12}$"}}]},
    {"label": "INVOICE_NUMBER", "pattern": [{"LOWER": {"IN": ["invoice", "inv"]}},
                                            {"TEXT": {"REGEX": "^[#:]?$"}, "OP": "?"},
                                            {"TEXT": {"REGEX": r"^[A-Za-z]{0,3}\d{3,12}$"}}]},
    {"label": "ISSUE_TYPE", "pattern": [{"LOWER": {"IN": ["delayed", "late", "missing", "broken",
                                                          "damaged", "wrong", "refund", "cancel"]}},
                                        {"LOWER": {"IN": ["delivery", "order", "item", "product",
                                                          "package", "payment"]}, "OP": "?"}]},
]


class EntityExtractor:
    def __init__(self, model: str = config.SPACY_MODEL):
        import spacy

        try:
            self.nlp = spacy.load(model, disable=["lemmatizer"])
        except OSError as exc:  # noqa: BLE001
            raise OSError(
                f"spaCy model '{model}' not installed. Run:\n"
                f"  python -m spacy download {model}"
            ) from exc

        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(_RULER_PATTERNS)

    def extract(self, text: str) -> list[dict]:
        """Return merged, de-duplicated entities from spaCy + regex."""
        doc = self.nlp(text)
        ents = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]

        # Add regex matches for high-value domain entities (regex is authoritative
        # for these; it catches formats the statistical model misses).
        for span in find_pii(text):
            ents.append({"text": span.value, "label": span.label,
                         "start": span.start, "end": span.end})

        # De-duplicate by character span, preferring domain labels over generic ones.
        priority = {"ORDER_NUMBER": 3, "INVOICE_NUMBER": 3, "TRACKING_NUMBER": 3,
                    "EMAIL": 3, "PHONE": 3, "ISSUE_TYPE": 2}
        ents.sort(key=lambda e: (e["start"], -priority.get(e["label"], 1)))
        out, seen = [], []
        for e in ents:
            if any(not (e["end"] <= a or e["start"] >= b) for a, b in seen):
                continue
            out.append(e)
            seen.append((e["start"], e["end"]))
        return out

    def render_html(self, text: str) -> str:
        """displaCy HTML for the report / Streamlit (great screenshots)."""
        from spacy import displacy

        doc = self.nlp(text)
        return displacy.render(doc, style="ent", page=False, jupyter=False)


def evaluate_against_placeholders(extractor: "EntityExtractor", df, text_col: str,
                                  limit: int = 300) -> dict:
    """
    Entity-level P/R/F1 using Bitext {{placeholder}} spans as silver labels.
    We compare label-agnostic span overlap (the demo cares about *finding* the fact).
    """
    from src.preprocessing import extract_placeholder_entities, strip_placeholders

    tp = fp = fn = 0
    rows = df[text_col].head(limit).tolist()
    for raw in rows:
        gold = extract_placeholder_entities(str(raw))
        if not gold:
            continue
        clean = strip_placeholders(str(raw))
        pred = extractor.extract(clean)
        gold_spans = [(g["start"], g["end"]) for g in gold]
        pred_spans = [(p["start"], p["end"]) for p in pred]
        matched = set()
        for gs, ge in gold_spans:
            hit = next((i for i, (ps, pe) in enumerate(pred_spans)
                        if i not in matched and not (ge <= ps or gs >= pe)), None)
            if hit is not None:
                tp += 1
                matched.add(hit)
            else:
                fn += 1
        fp += len(pred_spans) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    ex = EntityExtractor()
    msg = "Hi, I'm Alex Morgan. My order 48213 of wireless headphones is a delayed delivery."
    for e in ex.extract(msg):
        print(f"  {e['label']:<15} {e['text']}")
