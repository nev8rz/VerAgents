"""感知记忆：多模态数据占位实现。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService
from veragents.memory.storage.document_store import DocumentStore
from veragents.memory.storage.qdrant_store import QdrantStore


class PerceptualMemory(BaseMemory):
    """感知记忆占位实现。"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[QdrantStore] = None,
        embedding: Optional[EmbeddingService] = None,
    ):
        super().__init__(config or MemoryConfig(), storage_backend=document_store)
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding = embedding
        log.info(
            "PerceptualMemory initialized | document_store={} vector_store={} embedding={}",
            type(document_store).__name__ if document_store else None,
            type(vector_store).__name__ if vector_store else None,
            type(embedding).__name__ if embedding else None,
        )

    def add(self, memory_item: MemoryItem) -> str:
        log.warning("PerceptualMemory.add not implemented | id={}", memory_item.id)
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def retrieve(self, query: str, limit: int = 5, **kwargs):
        log.warning("PerceptualMemory.retrieve not implemented | query_len={}", len(query))
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        log.warning("PerceptualMemory.update not implemented | id={}", memory_id)
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def remove(self, memory_id: str) -> bool:
        log.warning("PerceptualMemory.remove not implemented | id={}", memory_id)
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def has_memory(self, memory_id: str) -> bool:
        log.warning("PerceptualMemory.has_memory not implemented | id={}", memory_id)
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def clear(self) -> None:
        log.warning("PerceptualMemory.clear not implemented")
        raise NotImplementedError("PerceptualMemory not implemented yet")

    def get_stats(self) -> Dict[str, Any]:
        log.warning("PerceptualMemory.get_stats not implemented")
        raise NotImplementedError("PerceptualMemory not implemented yet")


__all__ = ["PerceptualMemory"]
