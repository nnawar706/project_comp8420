"""
Retrieval-Augmented Generation.

Pipeline: chunk knowledge docs -> embed (sentence-transformers) -> store in a
persistent ChromaDB collection -> at query time embed the message and retrieve
top-k chunks. We attach the SAME embedding model to the collection so indexing
and querying never use a mismatched model (a classic RAG bug).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Simple recursive-ish splitter: break on paragraphs, then pack to `size` chars."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= size:
                buf = p
            else:  # very long paragraph -> sliding window
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i:i + size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def load_knowledge_documents(kb_dir: Path = config.KB_DIR) -> list[dict]:
    """Read every .md/.txt file in the KB dir into chunk records."""
    records = []
    files = sorted(list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No knowledge docs in {kb_dir}. Run scripts/build_knowledge_base.py first."
        )
    for f in files:
        text = f.read_text(encoding="utf-8")
        for i, ch in enumerate(chunk_text(text)):
            records.append({"id": f"{f.stem}-{i}", "text": ch, "source": f.name})
    return records


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
class RAGStore:
    def __init__(self, collection_name: str = config.CHROMA_COLLECTION):
        import chromadb
        from chromadb.utils import embedding_functions

        self.client = chromadb.PersistentClient(path=str(config.VECTOR_DB_DIR))
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, records: list[dict] | None = None) -> int:
        """(Re)build the index from KB documents."""
        records = records or load_knowledge_documents()
        # Reset for a clean rebuild.
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION, embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=[r["id"] for r in records],
            documents=[r["text"] for r in records],
            metadatas=[{"source": r["source"]} for r in records],
        )
        print(f"[rag] indexed {len(records)} chunks into '{self.collection.name}'")
        return len(records)

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[dict]:
        """Return top-k chunks: [{text, source, score}]."""
        n = self.collection.count()
        if n == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, n))
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None] * len(docs)])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            score = None if dist is None else round(1 - dist, 4)  # cosine sim
            out.append({"text": doc, "source": meta.get("source", "?"), "score": score})
        return out

    def count(self) -> int:
        return self.collection.count()


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a prompt-ready context block."""
    if not chunks:
        return "(no relevant company knowledge found)"
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


if __name__ == "__main__":
    store = RAGStore()
    if store.count() == 0:
        store.build()
    for c in store.retrieve("how do I get a refund for a late order?"):
        print(f"({c['score']}) {c['source']}: {c['text'][:80]}...")
