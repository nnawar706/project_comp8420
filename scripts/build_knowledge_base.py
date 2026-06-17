#!/usr/bin/env python
"""
Augment the seed company policies with a FAQ derived from the Bitext gold
`response`s, grouped by category. This gives RAG both hand-written policy docs
and data-grounded answer snippets, to support the answers.

The six seed policy docs already ship in data/knowledge_base/. This script adds
`faq_<category>.md` files. Safe to run even if the dataset isn't downloaded
(it will just skip the FAQ step).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import strip_placeholders  # noqa: E402


def build_faq(max_per_category: int = 12) -> None:
    try:
        from src import data_loader
        train, _, _ = data_loader.load_splits()
    except Exception as exc:  # noqa: BLE001
        print(f"[kb] dataset not available ({exc}); seed policies only. "
              "Run scripts/download_data.py to add the data-derived FAQ.")
        return

    if config.RESPONSE_COL not in train.columns:
        print("[kb] no gold response column; seed policies only.")
        return

    for cat, grp in train.groupby(config.CATEGORY_COL):
        seen, lines = set(), [f"# FAQ - {cat}\n"]
        for _, row in grp.iterrows():
            q = strip_placeholders(str(row[config.TEXT_COL])).strip()
            a = strip_placeholders(str(row[config.RESPONSE_COL])).strip()
            key = a[:60].lower()
            if not a or key in seen:
                continue
            seen.add(key)
            lines.append(f"## Q: {q}\n{a}\n")
            if len(seen) >= max_per_category:
                break
        path = config.KB_DIR / f"faq_{str(cat).lower().replace(' ', '_')}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[kb] wrote {path.name} ({len(seen)} entries)")


if __name__ == "__main__":
    n = len(list(config.KB_DIR.glob("*.md")))
    print(f"[kb] {n} seed policy docs present in {config.KB_DIR}")
    build_faq()
    print("Done. Now run scripts/build_index.py to (re)build the vector store.")
