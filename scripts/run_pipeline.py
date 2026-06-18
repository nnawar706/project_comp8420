#!/usr/bin/env python
"""
Run the full pipeline on a single message from the command line.

Usage:
  python scripts/run_pipeline.py "My order #48213 hasn't arrived and I'm furious"
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline import CustomerServicePipeline  # noqa: E402

DEMO = ("My order #48213 still hasn't arrived after two weeks and nobody is "
        "replying. This is unacceptable.")


def main() -> None:
    message = " ".join(sys.argv[1:]) or DEMO
    pipe = CustomerServicePipeline()
    res = pipe.process(message).to_dict()

    print("\n" + "=" * 70)
    print("MESSAGE     :", res["message"])
    print("CATEGORY    :", res["category"], f"(conf {res['category_confidence']})")
    print("SENTIMENT   :", res["sentiment"]["label"],
          f"(compound {res['sentiment']['compound']})")
    print("ENTITIES    :", ", ".join(f"{e['label']}={e['text']}" for e in res["entities"]) or "none")
    print("RETRIEVED   :", ", ".join(f"{c['source']}" for c in res["retrieved"]) or "none")
    print("-" * 70)
    print("RESPONSE    :", res["response"])
    print("CONFIDENCE  :", res["response_confidence"], "| schema_valid:", res["schema_valid"])
    print("ESCALATE    :", res["should_escalate"])
    if res["escalation_reasons"]:
        print("  reasons   :", "; ".join(res["escalation_reasons"]))
    print("=" * 70)
    print("\nFull JSON:\n", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
