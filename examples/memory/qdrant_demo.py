"""Qdrant store demo.

Requires qdrant-client and a running Qdrant instance.

Env (optional):
- QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION / QDRANT_VECTOR_SIZE

Usage:
  PYTHONPATH=. python examples/memory/qdrant_demo.py
"""

from __future__ import annotations

import os
import random

from veragents.memory.storage.qdrant_store import QdrantStore
from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    try:
        store = QdrantStore(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection=os.getenv("QDRANT_COLLECTION", "memory_demo"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "4")),
        )
    except ImportError as exc:
        print(f"[demo] qdrant-client not installed: {exc}")
        return
    except Exception as exc:
        print(f"[demo] failed to init QdrantStore: {exc}")
        return

    # create two random vectors
    vecs = [[random.random() for _ in range(store.vector_size)] for _ in range(2)]
    metas = [
        {"memory_id": "m1", "memory_type": "working", "user_id": "demo"},
        {"memory_id": "m2", "memory_type": "working", "user_id": "demo"},
    ]

    ok = store.add_vectors(vecs, metas)
    print("add_vectors ok:", ok)

    # search using first vector
    results = store.search_similar(vecs[0], limit=2)
    print("search results:", results)

    print("collection info:", store.get_collection_info())


if __name__ == "__main__":
    main()
