"""
Part-of-Speech tagging - "understanding query structure" (Use Case 1, basic techniques).

We use spaCy's POS tagger to expose the syntactic shape of a customer message:
the POS sequence, the main verbs (what the customer wants done) and noun chunks
(what they are talking about). These are useful, explainable signals - e.g. an
imperative verb like "cancel"/"refund" is a strong intent cue that complements
the TF-IDF classifier.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class POSTagger:
    def __init__(self, model: str = config.SPACY_MODEL):
        import spacy

        try:
            self.nlp = spacy.load(model, disable=["ner"])
        except OSError as exc:  # noqa: BLE001
            raise OSError(
                f"spaCy model '{model}' not installed. Run:\n"
                f"  python -m spacy download {model}"
            ) from exc

    def tag(self, text: str) -> list[dict]:
        """Token-level tags: (token, POS, fine tag, dependency)."""
        doc = self.nlp(text)
        return [{"token": t.text, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_}
                for t in doc if not t.is_space]

    def structure(self, text: str) -> dict:
        """Higher-level 'query structure' summary used as a classification signal."""
        doc = self.nlp(text)
        verbs = [t.lemma_.lower() for t in doc if t.pos_ == "VERB"]
        nouns = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
        noun_chunks = [c.text for c in doc.noun_chunks]
        is_question = text.strip().endswith("?") or (doc[0].tag_ in {"WP", "WRB", "WDT"} if len(doc) else False)
        is_imperative = bool(doc) and doc[0].pos_ == "VERB" and doc[0].dep_ == "ROOT"
        return {
            "main_verbs": verbs,
            "key_nouns": nouns,
            "noun_chunks": noun_chunks,
            "is_question": is_question,
            "is_imperative": is_imperative,
        }

    def pos_distribution(self, texts: list[str]) -> Counter:
        """Corpus-level POS counts (for an EDA / structure plot)."""
        counts: Counter = Counter()
        for doc in self.nlp.pipe(texts, batch_size=64):
            counts.update(t.pos_ for t in doc if not t.is_space)
        return counts

    def render_html(self, text: str):
        from spacy import displacy

        return displacy.render(self.nlp(text), style="dep", page=False, jupyter=False,
                               options={"compact": True, "distance": 90})


if __name__ == "__main__":
    pt = POSTagger()
    msg = "Please cancel my order and refund the payment to my card."
    print(pt.structure(msg))
