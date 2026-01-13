"""Minimal SDK embedding probe (OpenAI-compatible endpoint).

Env keys:
- EMBEDDING_BASE_URL / EMBED_BASE_URL
- EMBEDDING_MODEL / EMBED_MODEL / EMBED_MODEL_NAME (default: text-embedding-3-large)
- EMBEDDING_API_KEY / EMBED_API_KEY
- EMBEDDING_ENCODING_FORMAT / EMBED_ENCODING_FORMAT (default: float)
- EMBEDDING_DIMENSIONS / EMBED_DIMENSIONS (optional)
- EMBEDDING_USER / EMBED_USER (optional)

Usage:
    PYTHONPATH=. python tests/sdk_embedding_test.py
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

def sdk_call(input_payload: Any) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed; pip install openai")

    base_url = (os.getenv("EMBEDDING_BASE_URL") or os.getenv("EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or os.getenv("EMBED_MODEL_NAME") or "text-embedding-3-large"
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("EMBED_API_KEY") or ""
    encoding_format = os.getenv("EMBEDDING_ENCODING_FORMAT") or os.getenv("EMBED_ENCODING_FORMAT") or "float"
    dimensions_raw = os.getenv("EMBEDDING_DIMENSIONS") or os.getenv("EMBED_DIMENSIONS")
    user = os.getenv("EMBEDDING_USER") or os.getenv("EMBED_USER")
    dimensions = int(dimensions_raw) if dimensions_raw and dimensions_raw.isdigit() else None

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[sdk] POST {base_url}/embeddings | model={model} encoding_format={encoding_format} dims={dimensions} user={user}")
    kwargs = {"model": model, "input": input_payload}
    if encoding_format is not None:
        kwargs["encoding_format"] = encoding_format
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    if user is not None:
        kwargs["user"] = user

    resp = client.embeddings.create(**kwargs)
    # openai SDK returns objects; coerce to dict for easier viewing
    data = getattr(resp, "data", None) or []
    usage = getattr(resp, "usage", None)
    first = data[0].embedding[:6] if data else []
    print(f"[sdk] ok | count={len(data)} dim={len(data[0].embedding) if data else 0} preview={first}")
    if usage:
        print(f"[sdk] usage={usage}")
    return {"data": data, "usage": usage}


def rest_call(input_payload: Any) -> Dict[str, Any]:
    raise RuntimeError("REST disabled for this SDK-only diagnostic")


def main() -> None:
    if load_dotenv:
        load_dotenv()

    input_text = "Transformer 的注意力机制是什么？"
    try:
        print("\n=== SDK try: single string ===")
        sdk_call(input_text)
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[sdk] failed: {exc}")


if __name__ == "__main__":
    main()
