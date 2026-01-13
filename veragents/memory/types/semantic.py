"""语义记忆：知识图谱 + 向量检索。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService
from veragents.memory.storage.neo4j_store import Neo4jStore
from veragents.memory.storage.qdrant_store import QdrantStore


class SemanticMemory(BaseMemory):
    """语义记忆占位实现。"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        graph_store: Optional[Neo4jStore] = None,
        vector_store: Optional[QdrantStore] = None,
        embedding: Optional[EmbeddingService] = None,
    ):
        super().__init__(config or MemoryConfig(), storage_backend=graph_store)
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding = embedding
        log.info(
            "SemanticMemory initialized | graph_store={} vector_store={} embedding={}",
            type(graph_store).__name__ if graph_store else None,
            type(vector_store).__name__ if vector_store else None,
            type(embedding).__name__ if embedding else None,
        )

    def add(self, memory_item: MemoryItem) -> str:
        log.warning("SemanticMemory.add not implemented | id={}", memory_item.id)
        raise NotImplementedError("SemanticMemory not implemented yet")

    def retrieve(self, query: str, limit: int = 5, **kwargs):
        log.warning("SemanticMemory.retrieve not implemented | query_len={}", len(query))
        raise NotImplementedError("SemanticMemory not implemented yet")

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        log.warning("SemanticMemory.update not implemented | id={}", memory_id)
        raise NotImplementedError("SemanticMemory not implemented yet")

    def remove(self, memory_id: str) -> bool:
        log.warning("SemanticMemory.remove not implemented | id={}", memory_id)
        raise NotImplementedError("SemanticMemory not implemented yet")

    def has_memory(self, memory_id: str) -> bool:
        log.warning("SemanticMemory.has_memory not implemented | id={}", memory_id)
        raise NotImplementedError("SemanticMemory not implemented yet")

    def clear(self) -> None:
        log.warning("SemanticMemory.clear not implemented")
        raise NotImplementedError("SemanticMemory not implemented yet")

    def get_stats(self) -> Dict[str, Any]:
        log.warning("SemanticMemory.get_stats not implemented")
        raise NotImplementedError("SemanticMemory not implemented yet")


__all__ = ["SemanticMemory"]
