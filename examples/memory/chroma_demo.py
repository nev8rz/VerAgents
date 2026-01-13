"""Chroma store demo.

Requires chromadb installed.

Env (optional):
- CHROMA_PERSIST_PATH (for local) or CHROMA_HOST/CHROMA_PORT/CHROMA_AUTH_TOKEN

Usage:
  PYTHONPATH=. python examples/memory/chroma_demo.py
"""

from __future__ import annotations

import os
import random

from veragents.memory.storage.chroma_store import ChromaStore
from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    try:
        store = ChromaStore(
            collection=os.getenv("CHROMA_COLLECTION", "memory_demo"),
            persist_path=os.getenv("CHROMA_PERSIST_PATH"),
            host=os.getenv("CHROMA_HOST"),
            port=int(os.getenv("CHROMA_PORT")) if os.getenv("CHROMA_PORT") else None,
            auth_token=os.getenv("CHROMA_AUTH_TOKEN"),
        )
    except ImportError as exc:
        print(f"[demo] chromadb not installed: {exc}")
        return
    except Exception as exc:
        print(f"[demo] failed to init ChromaStore: {exc}")
        return

    vecs = [[random.random() for _ in range(4)] for _ in range(2)]
    metas = [
        {"memory_id": "c1", "memory_type": "working", "user_id": "demo"},
        {"memory_id": "c2", "memory_type": "working", "user_id": "demo"},
    ]

    ok = store.add_vectors(vecs, metas)
    print("add_vectors ok:", ok)

    results = store.search_similar(vecs[0], limit=2)
    print("search results:", results)

    print("collection info:", store.get_collection_info())


if __name__ == "__main__":
    main()
