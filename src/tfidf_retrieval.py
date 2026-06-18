"""
TF-IDF knowledge-base matching (Use Case 1, basic techniques).

A classical sparse-retrieval baseline over the same knowledge base that the
dense RAG store uses. Exposing the SAME .retrieve() interface as rag.RAGStore
lets us run a clean sparse-vs-dense ablation in the evaluation notebook - which
is a clean comparison against an alternative retrieval method.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.rag import load_knowledge_documents  # noqa: E402


class TfidfRetriever:
    def __init__(self):
        self.records: list[dict] = []
        self.vectorizer = None
        self.matrix = None

    def build(self, records: list[dict] | None = None) -> int:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.records = records or load_knowledge_documents()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in self.records])
        print(f"[tfidf-retriever] indexed {len(self.records)} chunks")
        return len(self.records)

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.matrix is None:
            self.build()
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        top = sims.argsort()[-k:][::-1]
        return [{"text": self.records[i]["text"],
                 "source": self.records[i]["source"],
                 "score": round(float(sims[i]), 4)} for i in top]


if __name__ == "__main__":
    r = TfidfRetriever()
    r.build()
    for c in r.retrieve("how long do refunds take?"):
        print(f"  ({c['score']}) {c['source']}: {c['text'][:70]}...")
