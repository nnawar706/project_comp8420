#!/usr/bin/env python
"""Build (or rebuild) the ChromaDB vector index from data/knowledge_base/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.rag import RAGStore, load_knowledge_documents  # noqa: E402


def main() -> None:
    records = load_knowledge_documents()
    store = RAGStore()
    n = store.build(records)
    print(f"\nIndexed {n} chunks. Quick retrieval check:")
    for c in store.retrieve("how long do refunds take?"):
        print(f"  ({c['score']}) {c['source']}: {c['text'][:70]}...")


if __name__ == "__main__":
    main()
