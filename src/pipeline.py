"""
End-to-end orchestration: one customer message -> the five outputs.

  1. category + confidence        (classification)
  2. sentiment + score            (sentiment)
  3. extracted entities           (NER)
  4. grounded generated response  (RAG + LLM)
  5. escalate / auto-resolve      (confidence-gating across all signals)

The classical signals feed and *gate* the LLM step: the predicted category can
route retrieval, and low classifier confidence / strong negative sentiment / low
LLM self-confidence each push the final decision toward escalation. That coupling
is the main design idea here.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import generation  # noqa: E402
from src.classification import TextClassifier  # noqa: E402
from src.ner import EntityExtractor  # noqa: E402
from src.preprocessing import clean_for_llm  # noqa: E402
from src.rag import RAGStore, format_context  # noqa: E402
from src.sentiment import SentimentAnalyzer  # noqa: E402


@dataclass
class PipelineResult:
    message: str
    category: str
    category_confidence: float
    sentiment: dict
    entities: list[dict]
    retrieved: list[dict]
    response: str
    response_confidence: float
    should_escalate: bool
    escalation_reasons: list[str] = field(default_factory=list)
    schema_valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CustomerServicePipeline:
    """Loads every component once; call .process(message) per request."""

    def __init__(self, use_rag: bool = True, prompt_variant: str = "few_shot_cot",
                 sentiment_backend: str = "vader"):
        self.use_rag = use_rag
        self.prompt_variant = prompt_variant
        print("[pipeline] loading components ...")
        self.classifier = self._safe(TextClassifier.load, "classifier")
        self.sentiment = SentimentAnalyzer(sentiment_backend)
        self.ner = self._safe(EntityExtractor, "NER")
        self.rag = self._safe(RAGStore, "RAG store") if use_rag else None
        print("[pipeline] ready.")

    @staticmethod
    def _safe(factory, name):
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] WARNING: {name} unavailable ({exc}). Continuing without it.")
            return None

    def process(self, message: str) -> PipelineResult:
        message = str(message).strip()
        llm_text = clean_for_llm(message)

        # 1) Classification -------------------------------------------------- #
        if self.classifier:
            category, cat_conf = self.classifier.predict(message)
        else:
            category, cat_conf = "unknown", 0.0

        # 2) Sentiment ------------------------------------------------------- #
        sentiment = self.sentiment.analyze(message)

        # 3) NER ------------------------------------------------------------- #
        entities = self.ner.extract(llm_text) if self.ner else []

        # 4) Retrieval (routed/enriched by category + entities) -------------- #
        retrieved = []
        if self.use_rag and self.rag:
            ent_hint = " ".join(e["text"] for e in entities)
            query = f"{category} {llm_text} {ent_hint}".strip()
            retrieved = self.rag.retrieve(query)
        context = format_context(retrieved)

        # 5) Generation ------------------------------------------------------ #
        gen = generation.generate_response(
            message=llm_text, context=context, entities=entities,
            category=category, variant=self.prompt_variant,
        )

        # Final escalation decision: combine all signals. ------------------- #
        reasons = []
        if gen.get("should_escalate"):
            reasons.append("LLM flagged the request for escalation")
        if cat_conf < config.CLASSIFIER_CONF_THRESHOLD:
            reasons.append(f"low classifier confidence ({cat_conf:.2f})")
        if gen.get("confidence", 1.0) < config.LLM_CONF_THRESHOLD:
            reasons.append(f"low response confidence ({gen.get('confidence', 0):.2f})")
        if sentiment["compound"] <= config.SENTIMENT_ESCALATE_COMPOUND:
            reasons.append(f"strongly negative sentiment ({sentiment['compound']:.2f})")
        if not retrieved and self.use_rag:
            reasons.append("no supporting company knowledge retrieved")

        should_escalate = len(reasons) > 0

        return PipelineResult(
            message=message,
            category=category,
            category_confidence=round(cat_conf, 4),
            sentiment=sentiment,
            entities=entities,
            retrieved=retrieved,
            response=gen.get("reply", ""),
            response_confidence=round(float(gen.get("confidence", 0.0)), 4),
            should_escalate=should_escalate,
            escalation_reasons=reasons,
            schema_valid=gen.get("schema_valid", False),
        )


if __name__ == "__main__":
    import json

    pipe = CustomerServicePipeline()
    res = pipe.process("My order #48213 still hasn't arrived after two weeks and nobody is replying. This is unacceptable.")
    print(json.dumps(res.to_dict(), indent=2)[:2000])
