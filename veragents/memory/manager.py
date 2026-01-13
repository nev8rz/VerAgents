"""记忆管理器 - 记忆核心层的统一管理接口。"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.types.episodic import EpisodicMemory
from veragents.memory.types.perceptual import PerceptualMemory
from veragents.memory.types.semantic import SemanticMemory
from veragents.memory.types.working import WorkingMemory


class MemoryManager:
    """统一的记忆操作接口，负责多类型记忆协调。"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
    ):
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types: Dict[str, BaseMemory] = {}

        if enable_working:
            self.memory_types["working"] = WorkingMemory(self.config)
        if enable_episodic:
            self.memory_types["episodic"] = EpisodicMemory(self.config)
        if enable_semantic:
            self.memory_types["semantic"] = SemanticMemory(self.config)
        if enable_perceptual:
            self.memory_types["perceptual"] = PerceptualMemory(self.config)

        log.info("MemoryManager initialized | enabled={}", list(self.memory_types.keys()))

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True,
    ) -> str:
        """添加记忆。"""
        target_type = self._classify_memory_type(content, metadata) if auto_classify else memory_type
        if importance is None:
            importance = self._calculate_importance(content, metadata)

        memory_item = MemoryItem(
            id="",
            content=content,
            memory_type=target_type,
            user_id=self.user_id,
            timestamp=datetime.utcnow(),
            importance=importance,
            metadata=metadata or {},
        )

        memory = self.memory_types.get(target_type)
        if not memory:
            raise ValueError(f"不支持的记忆类型: {target_type}")

        memory_id = memory.add(memory_item)
        log.info("Memory added | type={} id={} importance={}", target_type, memory_id, importance)
        return memory_id

    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple[datetime, datetime]] = None,
    ) -> List[MemoryItem]:
        """检索记忆。"""
        types = memory_types or list(self.memory_types.keys())
        per_type_limit = max(1, limit // max(1, len(types)))
        results: List[MemoryItem] = []

        for mtype in types:
            memory = self.memory_types.get(mtype)
            if not memory:
                continue
            try:
                type_results = memory.retrieve(query=query, limit=per_type_limit)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Memory retrieve failed | type={} err={}", mtype, exc)
                continue
            filtered = self._filter_results(type_results, min_importance, time_range)
            results.extend(filtered)

        results.sort(key=lambda x: getattr(x, "importance", 0.0), reverse=True)
        return results[:limit]

    def update_memory(self, memory_id: str, content: Optional[str] = None, importance: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新记忆。"""
        for mtype, memory in self.memory_types.items():
            if hasattr(memory, "has_memory") and memory.has_memory(memory_id):
                updated = memory.update(memory_id, content, importance, metadata)
                log.info("Memory updated | type={} id={} success={}", mtype, memory_id, updated)
                return updated
        log.warning("Memory update failed, id not found | id={}", memory_id)
        return False

    def remove_memory(self, memory_id: str) -> bool:
        """删除记忆。"""
        for mtype, memory in self.memory_types.items():
            if hasattr(memory, "has_memory") and memory.has_memory(memory_id):
                removed = memory.remove(memory_id)
                log.info("Memory removed | type={} id={} success={}", mtype, memory_id, removed)
                return removed
        log.warning("Memory remove failed, id not found | id={}", memory_id)
        return False

    def clear_all_memories(self) -> None:
        """清空所有记忆。"""
        for memory in self.memory_types.values():
            try:
                memory.clear()
            except Exception as exc:  # pragma: no cover
                log.warning("Memory clear failed | type={} err={}", getattr(memory, "memory_type", None), exc)
        log.info("All memories cleared")

    # ------------------------------------------------------------------ #
    # Stats & utils
    # ------------------------------------------------------------------ #
    def get_memory_stats(self) -> Dict[str, Any]:
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor,
            },
        }
        for mtype, memory in self.memory_types.items():
            try:
                mstats = memory.get_stats()
            except Exception:
                mstats = {}
            stats["memories_by_type"][mtype] = mstats
            stats["total_memories"] += mstats.get("count", 0)
        return stats

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        if metadata and metadata.get("type"):
            return metadata["type"]
        if self._is_episodic_content(content):
            return "episodic"
        if self._is_semantic_content(content):
            return "semantic"
        return "working"

    @staticmethod
    def _is_episodic_content(content: str) -> bool:
        episodic_keywords = ["昨天", "今天", "明天", "上次", "记得", "发生", "经历"]
        return any(k in content for k in episodic_keywords)

    @staticmethod
    def _is_semantic_content(content: str) -> bool:
        semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "方法"]
        return any(k in content for k in semantic_keywords)

    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        importance = 0.5
        if len(content) > 100:
            importance += 0.1
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(k in content for k in important_keywords):
            importance += 0.2
        if metadata:
            priority = metadata.get("priority")
            if priority == "high":
                importance += 0.3
            elif priority == "low":
                importance -= 0.2
        return max(0.0, min(1.0, importance))

    def _filter_results(self, items: List[MemoryItem], min_importance: float, time_range: Optional[tuple[datetime, datetime]]) -> List[MemoryItem]:
        filtered: List[MemoryItem] = []
        for item in items:
            if getattr(item, "importance", 0.0) < min_importance:
                continue
            if time_range:
                start, end = time_range
                ts = getattr(item, "timestamp", None)
                if ts and (ts < start or ts > end):
                    continue
            filtered.append(item)
        return filtered

    def __str__(self) -> str:
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"
