"""统一嵌入模块（API 优先，OpenAI 兼容，包括 SDK 与 REST）。

支持环境变量（按优先级读取同义键）：
- 模式：EMBED_MODEL_TYPE / EMBEDDING_MODEL_TYPE（默认 api）
- 模型：EMBED_MODEL_NAME / EMBEDDING_MODEL / EMBED_MODEL
- Key：EMBED_API_KEY / EMBEDDING_API_KEY
- Base URL：EMBED_BASE_URL / EMBEDDING_BASE_URL
- 其他：EMBED_ENCODING_FORMAT / EMBEDDING_ENCODING_FORMAT（默认 float）
        EMBED_DIMENSIONS / EMBEDDING_DIMENSIONS
        EMBED_USER / EMBEDDING_USER
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Union

from loguru import logger as log


class EmbeddingModel:
    """嵌入模型最小接口。"""

    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class ApiEmbedding(EmbeddingModel):
    """通用 OpenAI 兼容接口，优先使用 openai SDK，缺失时回落 REST。"""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        encoding_format: Optional[str] = "float",
        dimensions: Optional[int] = None,
        user: Optional[str] = None,
        use_sdk: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.user = user
        self._dimension: Optional[int] = None
        self._client = None

        if use_sdk:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._use_sdk = True
        else:
            self._use_sdk = False

        log.info(
            "Embedding client initialized | model={} base_url={} use_sdk={} encoding_format={} dimensions={}",
            self.model_name,
            self.base_url,
            self._use_sdk,
            self.encoding_format,
            self.dimensions,
        )

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        log.info(
            "Embedding request | model={} base_url={} use_sdk={} count={} single={}",
            self.model_name,
            self.base_url,
            self._use_sdk,
            len(inputs),
            single,
        )

        try:
            if self._use_sdk and self._client:
                try:
                    vectors = self._encode_sdk(inputs)
                except Exception:
                    # SDK 模式失败时自动回落 REST，避免兼容性问题
                    log.warning("Embedding SDK call failed, fallback to REST | model={} base_url={}", self.model_name, self.base_url)
                    vectors = self._encode_rest(inputs)
            else:
                vectors = self._encode_rest(inputs)
        except Exception:
            log.exception("Embedding request failed | model={} base_url={}", self.model_name, self.base_url)
            raise

        if single:
            return vectors[0]
        return vectors

    def _encode_sdk(self, inputs: List[str]) -> List[List[float]]:
        payload_input: Union[str, List[str]] = inputs[0] if len(inputs) == 1 else inputs
        kwargs: dict[str, Union[str, List[str], int]] = {"model": self.model_name, "input": payload_input}
        if self.encoding_format is not None:
            kwargs["encoding_format"] = self.encoding_format
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.user is not None:
            kwargs["user"] = self.user

        resp = self._client.embeddings.create(**kwargs)
        items = getattr(resp, "data", None) or []
        vectors: List[List[float]] = []
        for item in items:
            emb = getattr(item, "embedding", None) or getattr(item, "vector", None)
            if emb is None:
                continue
            vectors.append([float(v) for v in emb])
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])

        log.info(
            "Embedding response (sdk) | model={} base_url={} count={} dim={}",
            self.model_name,
            self.base_url,
            len(vectors),
            self._dimension,
        )
        return vectors

    def _encode_rest(self, inputs: List[str]) -> List[List[float]]:
        import requests

        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload_input: Union[str, List[str]] = inputs[0] if len(inputs) == 1 else inputs
        payload = {
            "model": self.model_name,
            "input": payload_input,
            "encoding_format": self.encoding_format,
        }
        if self.dimensions is not None:
            payload["dimensions"] = int(self.dimensions)
        if self.user is not None:
            payload["user"] = self.user

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
            log.error(
                "Embedding REST 调用失败 | model={} base_url={} status={} body={}",
                self.model_name,
                self.base_url,
                resp.status_code,
                resp.text,
            )
            raise RuntimeError(f"Embedding REST 调用失败: {resp.status_code} {resp.text}")
        data = resp.json()
        items = data.get("data") or []
        vectors: List[List[float]] = []
        for item in items:
            emb = item.get("embedding") or item.get("vector")
            if emb is None:
                continue
            vectors.append([float(v) for v in emb])
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])

        log.info(
            "Embedding response (rest) | model={} base_url={} count={} dim={}",
            self.model_name,
            self.base_url,
            len(vectors),
            self._dimension,
        )
        return vectors

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return int(self._dimension)
        try:
            probe = self.encode("health_check")
            self._dimension = len(probe)
            return int(self._dimension)
        except Exception:
            return 0


class EmbeddingService:
    """对外暴露的统一嵌入接口（API 优先）。"""

    def __init__(self, model: Optional[EmbeddingModel] = None):
        self._model = model or get_text_embedder()

    def embed_text(self, text: str) -> List[float]:
        vec = self._model.encode(text)
        return list(vec)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts)
        return [list(v) for v in vectors]


_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None


def _env_first(*keys: str, default: str = "") -> str:
    for k in keys:
        val = os.getenv(k)
        if val:
            return val.strip()
    return default


def _build_embedder() -> EmbeddingModel:
    _ = _env_first("EMBED_MODEL_TYPE", "EMBEDDING_MODEL_TYPE", default="api").lower()
    model_name = _env_first("EMBED_MODEL_NAME", "EMBEDDING_MODEL", "EMBED_MODEL")
    api_key = _env_first("EMBED_API_KEY", "EMBEDDING_API_KEY")
    base_url = _env_first("EMBED_BASE_URL", "EMBEDDING_BASE_URL")
    encoding_format = _env_first("EMBED_ENCODING_FORMAT", "EMBEDDING_ENCODING_FORMAT", default="float") or None
    dimensions_raw = _env_first("EMBED_DIMENSIONS", "EMBEDDING_DIMENSIONS")
    user = _env_first("EMBED_USER", "EMBEDDING_USER") or None
    use_sdk_raw = _env_first("EMBED_USE_SDK", "EMBEDDING_USE_SDK")
    dimensions: Optional[int] = None
    if dimensions_raw:
        try:
            dimensions = int(dimensions_raw)
        except ValueError:
            dimensions = None
    use_sdk: Optional[bool] = None
    if use_sdk_raw:
        use_sdk = use_sdk_raw.lower() in {"1", "true", "yes", "y"}
    return ApiEmbedding(
        model_name=model_name or "text-embedding-3-large",
        api_key=api_key,
        base_url=base_url,
        encoding_format=encoding_format,
        dimensions=dimensions,
        user=user,
        use_sdk=use_sdk,
    )


def get_text_embedder() -> EmbeddingModel:
    """获取全局共享的文本嵌入实例（线程安全单例）。"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """获取统一向量维度（失败回退默认值）。"""
    try:
        return int(get_text_embedder().dimension)
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """强制重建嵌入实例（可用于动态切换环境变量）。"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder


__all__ = [
    "EmbeddingModel",
    "ApiEmbedding",
    "EmbeddingService",
    "get_text_embedder",
    "get_dimension",
    "refresh_embedder",
]
