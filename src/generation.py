"""
Response generation via a local LLM served by Ollama.

We talk to Ollama over its HTTP API (no extra dependency, fully transparent and
easy to debug). Prompts are engineered with: a role, retrieved company context,
the extracted entities, optional few-shot examples, optional chain-of-thought,
and a strict JSON output schema {reply, confidence, should_escalate}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
SYSTEM_ROLE = (
    "You are a professional, empathetic customer-support assistant for an online retailer. "
    "Use ONLY the company knowledge provided to ground your answer. If the knowledge does "
    "not cover the question, say so and recommend escalation rather than inventing policy. "
    "Be concise, polite, and reference the customer's order/details when available."
)

FEW_SHOT = [
    {
        "message": "My order 12001 was supposed to arrive Tuesday and it's now Friday.",
        "context": "Standard delivery is 3-5 business days. Orders delayed beyond 7 business "
                   "days qualify for a shipping refund on request.",
        "reply": "I'm sorry your order #12001 is running late. Standard delivery is 3-5 business "
                 "days; since yours is past that window I've flagged it for our delivery team and "
                 "you're eligible to request a shipping refund if it passes 7 business days.",
        "confidence": 0.88, "should_escalate": False,
    },
    {
        "message": "I want to talk to someone about a legal dispute over my account.",
        "context": "(no relevant company knowledge found)",
        "reply": "This needs a specialist. I'm connecting you with a human agent who can help "
                 "with account and legal matters.",
        "confidence": 0.35, "should_escalate": True,
    },
]

OUTPUT_SCHEMA = (
    'Respond with ONLY a JSON object, no prose, no code fences:\n'
    '{"reply": "<your message to the customer>", '
    '"confidence": <float 0..1>, '
    '"should_escalate": <true|false>}'
)


def build_prompt(message: str, context: str, entities: list[dict] | None,
                 category: str | None, variant: str = "few_shot_cot") -> str:
    """variant ∈ {zero_shot, few_shot, few_shot_cot}."""
    ent_str = ", ".join(f"{e['label']}={e['text']}" for e in (entities or [])) or "none"
    parts = [SYSTEM_ROLE, ""]

    if variant in ("few_shot", "few_shot_cot"):
        parts.append("Here are examples of good responses:")
        for ex in FEW_SHOT:
            parts.append(f"Customer: {ex['message']}")
            parts.append(f"Company knowledge: {ex['context']}")
            parts.append(json.dumps({"reply": ex["reply"], "confidence": ex["confidence"],
                                     "should_escalate": ex["should_escalate"]}))
            parts.append("")

    parts.append("Now handle this request.")
    parts.append(f"Predicted category: {category or 'unknown'}")
    parts.append(f"Extracted facts: {ent_str}")
    parts.append(f"Company knowledge:\n{context}")
    parts.append(f"Customer message: {message}")
    parts.append("")

    if variant == "few_shot_cot":
        parts.append(
            "Think step by step privately: (1) what does the customer need, (2) does the "
            "company knowledge actually answer it, (3) are you confident enough to resolve or "
            "should this go to a human? Then output ONLY the JSON."
        )
    parts.append(OUTPUT_SCHEMA)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Ollama call
# --------------------------------------------------------------------------- #
def ollama_available() -> bool:
    try:
        requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except Exception:  # noqa: BLE001
        return False


def call_ollama(prompt: str, model: str = config.LLM_MODEL,
                temperature: float = config.LLM_TEMPERATURE) -> str:
    """Single completion call to Ollama's /api/generate."""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature}},
        timeout=config.LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(raw: str) -> dict:
    """Robustly pull the JSON object out of an LLM response."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    candidates = [cleaned]
    m = _JSON_RE.search(cleaned)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            return {
                "reply": str(obj.get("reply", "")).strip(),
                "confidence": float(obj.get("confidence", 0.5)),
                "should_escalate": bool(obj.get("should_escalate", False)),
                "schema_valid": True,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    # Fallback: couldn't parse -> treat raw text as the reply, force escalation.
    return {"reply": raw.strip()[:600], "confidence": 0.3,
            "should_escalate": True, "schema_valid": False}


def generate_response(message: str, context: str, entities: list[dict] | None = None,
                      category: str | None = None, variant: str = "few_shot_cot",
                      model: str = config.LLM_MODEL) -> dict:
    """Full generate step: build prompt -> call LLM -> parse JSON. Degrades gracefully."""
    prompt = build_prompt(message, context, entities, category, variant)
    if not ollama_available():
        return {
            "reply": ("[LLM offline] I couldn't reach a local language model, so I'm routing "
                      "this to a human agent. (Start Ollama and `ollama pull " + model + "`.)"),
            "confidence": 0.0, "should_escalate": True, "schema_valid": False,
            "llm_offline": True, "prompt": prompt,
        }
    try:
        raw = call_ollama(prompt, model=model)
        parsed = parse_json_response(raw)
        parsed["prompt"] = prompt
        parsed["raw"] = raw
        return parsed
    except Exception as exc:  # noqa: BLE001
        return {"reply": f"[generation error: {exc}] Routing to a human agent.",
                "confidence": 0.0, "should_escalate": True, "schema_valid": False,
                "error": str(exc), "prompt": prompt}


if __name__ == "__main__":
    print("Ollama available:", ollama_available())
    out = generate_response(
        "My order 48213 still hasn't arrived after two weeks!",
        context="Orders delayed beyond 7 business days qualify for a shipping refund.",
        entities=[{"label": "ORDER_NUMBER", "text": "48213"}],
        category="DELIVERY",
    )
    print(json.dumps({k: v for k, v in out.items() if k not in ("prompt", "raw")}, indent=2))
