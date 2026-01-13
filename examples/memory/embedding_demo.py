"""Minimal embedding demo using the unified API embedder.

Usage:
  export EMBEDDING_BASE_URL=...
  export EMBEDDING_MODEL=...
  export EMBEDDING_API_KEY=...
  python -m examples.memory.embedding_demo
"""

from __future__ import annotations

import os
from typing import List

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger as log

from veragents.memory.embedding import EmbeddingService, get_dimension, refresh_embedder

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

def _env(key: str, *aliases: str) -> str | None:
    for k in (key, *aliases):
        val = os.getenv(k)
        if val:
            return val
    return None


def _fmt_vec(vec: List[float], head: int = 6) -> str:
    if not vec:
        return "[]"
    prefix = [round(v, 4) for v in vec[:head]]
    if len(vec) > head:
        prefix.append("...")
    return str(prefix)


def main() -> None:
    base_url = _env("EMBED_BASE_URL", "EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
    model = _env("EMBED_MODEL_NAME", "EMBEDDING_MODEL", "EMBED_MODEL") or "text-embedding-3-small"
    api_key_present = bool(_env("EMBED_API_KEY", "EMBEDDING_API_KEY"))

    log.info("Embedding demo config | base_url={} model={} api_key_set={}", base_url, model, api_key_present)

    refresh_embedder()  # pick up latest env
    embedder = EmbeddingService()

    single_text = "Embedding demo: make this string into a vector."
    batch_texts = [
        "First test sentence.",
        "Second test sentence.",
    ]

    try:
        vec = embedder.embed_text(single_text)
        log.info("Embedding demo single | dim={} preview={}", len(vec), _fmt_vec(vec))
    except Exception as exc:  # pragma: no cover - demo logging
        log.exception("Embedding demo single failed: {}", exc)
        return

    try:
        batch_vecs = embedder.embed_batch(batch_texts)
        dims = {len(v) for v in batch_vecs}
        preview = ", ".join(_fmt_vec(v) for v in batch_vecs)
        log.info("Embedding demo batch | dims={} preview={}", dims, preview)
    except Exception as exc:  # pragma: no cover - demo logging
        log.exception("Embedding demo batch failed: {}", exc)
        return

    log.info("Embedding demo dimension helper | get_dimension() -> {}", get_dimension())


if __name__ == "__main__":
    main()
