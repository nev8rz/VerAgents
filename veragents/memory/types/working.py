"""工作记忆：短期 TTL 管理，纯内存实现。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService


class WorkingMemory(BaseMemory):
    """轻量工作记忆，适合快速读写。"""

    def __init__(self, config: Optional[MemoryConfig] = None, embedding: Optional[EmbeddingService] = None):
        super().__init__(config or MemoryConfig(), storage_backend=None)
        self.embedding = embedding
        self._items: List[MemoryItem] = []

    def add(self, memory_item: MemoryItem) -> str:
        item = memory_item

        if not item.id:
            item.id = self._generate_id()

        # Apply TTL based on timestamp
        self._prune_expired()
        if self.embedding:
            item.metadata["embedding"] = self.embedding.embed_text(item.content)

        self._items.append(item)
        log.info(
            "WorkingMemory add | id={} user={} type={} content_len={} ttl_minutes={}",
            item.id,
            getattr(item, "user_id", None),
            item.memory_type,
            len(item.content),
            self.config.working_memory_ttl_minutes,
        )
        return item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        self._prune_expired()
        needle = query.lower()

        results: List[MemoryItem] = []
        for mem in reversed(self._items):
            if needle in mem.content.lower():
                results.append(mem)
            if len(results) >= limit:
                break

        results = list(reversed(results))
        log.info("WorkingMemory retrieve | query_len={} returned={}", len(query), len(results))
        return results

    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        for item in self._items:
            if item.id != memory_id:
                continue
            if content is not None:
                item.content = content
            if importance is not None:
                item.importance = importance
            if metadata:
                item.metadata.update(metadata)
            log.info("WorkingMemory update | id={} content_updated={} importance_updated={}", memory_id, content is not None, importance is not None)
            return True
        return False

    def remove(self, memory_id: str) -> bool:
        before = len(self._items)
        self._items = [item for item in self._items if item.id != memory_id]
        removed = len(self._items) < before
        log.info("WorkingMemory remove | id={} success={}", memory_id, removed)
        return removed

    def has_memory(self, memory_id: str) -> bool:
        return any(item.id == memory_id for item in self._items)

    def clear(self) -> None:
        self._items.clear()
        log.info("WorkingMemory cleared")

    def get_stats(self) -> dict:
        self._prune_expired()
        return {
            "count": len(self._items),
            "capacity": self.config.working_memory_capacity,
            "ttl_minutes": self.config.working_memory_ttl_minutes,
        }

    def _prune_expired(self) -> None:
        ttl_minutes = self.config.working_memory_ttl_minutes
        if not ttl_minutes:
            return
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        before = len(self._items)
        self._items = [item for item in self._items if item.timestamp >= cutoff]
        pruned = before - len(self._items)
        if pruned > 0:
            log.info("WorkingMemory pruned expired items | removed={}", pruned)


__all__ = ["WorkingMemory"]
