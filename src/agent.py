"""
Agentic layer (Use Case 1, advanced techniques): a ReAct-style agent with tool
use, dialogue state tracking, and automated escalation.

The agent reasons step by step (Thought -> Action -> Observation) and chooses
among tools:
  * search_policy(query)      -> dense RAG retrieval over company knowledge
  * get_order_status(order_id)-> a mocked order database lookup
  * escalate_to_human(reason) -> terminal hand-off

This demonstrates agentic design + ReAct prompting + tool use + automated
escalation as an LLM decision, on top of the same RAG/LLM components used
elsewhere. It degrades gracefully when Ollama is offline (it escalates).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import generation  # noqa: E402

# --------------------------------------------------------------------------- #
# Mock order database (stands in for a real backend system)
# --------------------------------------------------------------------------- #
_MOCK_ORDERS = {
    "48213": {"status": "in transit", "dispatched_days_ago": 14, "carrier": "ParcelCo"},
    "90871": {"status": "processing", "dispatched_days_ago": 0, "carrier": "-"},
    "12001": {"status": "delivered", "dispatched_days_ago": 9, "carrier": "ParcelCo"},
}


def get_order_status(order_id: str) -> str:
    rec = _MOCK_ORDERS.get(re.sub(r"\D", "", str(order_id)))
    if not rec:
        return f"No order found with id {order_id}."
    return (f"Order {order_id}: status={rec['status']}, dispatched "
            f"{rec['dispatched_days_ago']} business days ago via {rec['carrier']}.")


# --------------------------------------------------------------------------- #
# Dialogue state tracking
# --------------------------------------------------------------------------- #
@dataclass
class DialogueState:
    turns: list[dict] = field(default_factory=list)     # [{role, text}]
    entities: dict = field(default_factory=dict)          # accumulated slots
    category: str | None = None
    resolved: bool = False
    escalated: bool = False

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})

    def update_entities(self, ents: list[dict]) -> None:
        for e in ents:
            self.entities.setdefault(e["label"], e["text"])

    def summary(self) -> str:
        slots = ", ".join(f"{k}={v}" for k, v in self.entities.items()) or "none"
        return (f"category={self.category}; slots: {slots}; "
                f"resolved={self.resolved}; escalated={self.escalated}; "
                f"turns={len(self.turns)}")


# --------------------------------------------------------------------------- #
# ReAct agent
# --------------------------------------------------------------------------- #
_TOOLS_DOC = """Available tools:
- search_policy(query): search company policy/FAQ knowledge for relevant text.
- get_order_status(order_id): look up the current status of an order.
- escalate_to_human(reason): hand the conversation to a human agent.
"""

_REACT_SYSTEM = (
    "You are a customer-support agent that reasons step by step and may use tools.\n"
    + _TOOLS_DOC +
    '\nAt each step respond with ONLY a JSON object, either:\n'
    '  {"thought": "...", "action": "search_policy|get_order_status|escalate_to_human", "action_input": "..."}\n'
    'or, when you are ready to answer the customer:\n'
    '  {"thought": "...", "final_answer": "...", "confidence": <0..1>, "should_escalate": <true|false>}\n'
    "Use get_order_status when an order number is known. Use search_policy to ground any "
    "policy claim. Escalate if the knowledge does not cover the request or you are unsure."
)


def _parse_step(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    for cand in ([m.group(0)] if m else []) + [cleaned]:
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
    return {}


class ReActAgent:
    def __init__(self, rag_store=None, max_steps: int = 4, model: str = config.LLM_MODEL):
        self.rag = rag_store
        self.max_steps = max_steps
        self.model = model

    def _run_tool(self, action: str, action_input: str) -> str:
        if action == "get_order_status":
            return get_order_status(action_input)
        if action == "search_policy":
            if self.rag is None:
                return "Policy search unavailable."
            chunks = self.rag.retrieve(action_input, k=3)
            return "\n".join(f"[{c['source']}] {c['text']}" for c in chunks) or "No policy found."
        if action == "escalate_to_human":
            return f"ESCALATED: {action_input}"
        return f"Unknown tool: {action}"

    def run(self, message: str, state: DialogueState | None = None) -> dict:
        state = state or DialogueState()
        state.add_turn("customer", message)

        if not generation.ollama_available():
            state.escalated = True
            return {"final_answer": "[LLM offline] Routing to a human agent.",
                    "confidence": 0.0, "should_escalate": True, "trace": [], "state": state}

        scratch = [f"{_REACT_SYSTEM}\n\nDialogue state: {state.summary()}\n",
                   f"Customer: {message}\n"]
        trace = []
        for _ in range(self.max_steps):
            prompt = "\n".join(scratch) + "\nRespond with the next JSON step:"
            try:
                raw = generation.call_ollama(prompt, model=self.model, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                trace.append({"error": str(exc)})
                break
            step = _parse_step(raw)
            trace.append(step or {"unparsed": raw[:200]})

            if "final_answer" in step:
                state.add_turn("agent", step["final_answer"])
                state.resolved = not bool(step.get("should_escalate"))
                state.escalated = bool(step.get("should_escalate"))
                return {"final_answer": step["final_answer"],
                        "confidence": float(step.get("confidence", 0.6)),
                        "should_escalate": bool(step.get("should_escalate", False)),
                        "trace": trace, "state": state}

            action = step.get("action")
            if not action:
                break
            obs = self._run_tool(action, step.get("action_input", ""))
            scratch.append(json.dumps(step))
            scratch.append(f"Observation: {obs}")
            if action == "escalate_to_human":
                state.escalated = True
                return {"final_answer": "I'm connecting you with a human agent who can help.",
                        "confidence": 0.3, "should_escalate": True,
                        "trace": trace, "state": state}

        # Fell through max steps without a final answer -> escalate.
        state.escalated = True
        return {"final_answer": "I'm escalating this to a human agent to make sure it's handled.",
                "confidence": 0.3, "should_escalate": True, "trace": trace, "state": state}


if __name__ == "__main__":
    from src.rag import RAGStore

    try:
        store = RAGStore()
    except Exception:  # noqa: BLE001
        store = None
    agent = ReActAgent(rag_store=store)
    out = agent.run("Where is my order 48213? It's been ages.")
    print(json.dumps({k: v for k, v in out.items() if k != "state"}, indent=2, default=str))
    print("Final state:", out["state"].summary())
