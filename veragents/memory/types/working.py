"""工作记忆：短期上下文管理，纯内存实现。

特点：
- 容量有限（通常 10-20 条记忆）
- 时效性强（会话级别，TTL 控制）
- 优先级管理
- 自动清理过期记忆
- 无持久化需求
"""

from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """工作记忆实现。

    特点：
    - 容量有限（通常 10-20 条记忆）
    - 时效性强（会话级别）
    - 优先级管理
    - 自动清理过期记忆
    - 纯内存存储（无持久化）
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
    ):
        super().__init__(config or MemoryConfig(), storage_backend=None)

        # 工作记忆特定配置
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        # 内存存储（工作记忆不需要持久化）
        self.memories: List[MemoryItem] = []

        # 使用优先级队列管理记忆
        self.memory_heap: List[Tuple[float, datetime, MemoryItem]] = []

        log.info(
            "WorkingMemory initialized | capacity={} tokens={} ttl_minutes={}",
            self.max_capacity,
            self.max_tokens,
            self.max_age_minutes,
        )

    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆。"""
        # 过期清理
        self._expire_old_memories()

        # 计算优先级（重要性 + 时间衰减）
        priority = self._calculate_priority(memory_item)

        # 添加到堆中
        heapq.heappush(self.memory_heap, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)

        # 更新 token 计数
        self.current_tokens += self._count_tokens(memory_item.content)

        # 检查容量限制
        self._enforce_capacity_limits()

        log.info(
            "WorkingMemory add | id={} user={} content_len={} tokens={}",
            memory_item.id,
            memory_item.user_id,
            len(memory_item.content),
            self.current_tokens,
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索工作记忆（关键词匹配 + 重要性加权）。"""
        # 过期清理
        self._expire_old_memories()

        if not self.memories:
            return []

        user_id = kwargs.get("user_id")

        # 按用户 ID 过滤（如果提供）
        filtered_memories = self.memories
        if user_id:
            filtered_memories = [m for m in self.memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        # 计算分数
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored_memories: List[Tuple[float, MemoryItem]] = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()

            # 关键词匹配分数
            keyword_score = 0.0
            if query_lower in content_lower:
                # 完整匹配
                keyword_score = 0.8
            else:
                # 分词匹配
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.6

            if keyword_score <= 0:
                continue

            # 时间衰减
            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance = keyword_score * time_decay

            # 重要性加权因子 [0.8, 1.2]
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            scored_memories.append((final_score, memory))

        # 按分数排序并返回
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        results = [memory for _, memory in scored_memories[:limit]]

        log.info("WorkingMemory retrieve | query_len={} returned={}", len(query), len(results))
        return results

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新工作记忆。"""
        for memory in self.memories:
            if memory.id == memory_id:
                old_tokens = self._count_tokens(memory.content)

                if content is not None:
                    memory.content = content
                    # 更新 token 计数
                    new_tokens = self._count_tokens(content)
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens

                if importance is not None:
                    memory.importance = importance

                if metadata is not None:
                    memory.metadata.update(metadata)

                # 重新计算优先级并更新堆
                self._rebuild_heap()

                log.info("WorkingMemory update | id={} success=True", memory_id)
                return True

        return False

    def remove(self, memory_id: str) -> bool:
        """删除工作记忆。"""
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                # 从列表中删除
                removed_memory = self.memories.pop(i)

                # 更新 token 计数
                self.current_tokens -= self._count_tokens(removed_memory.content)
                self.current_tokens = max(0, self.current_tokens)

                # 重建堆
                self._rebuild_heap()

                log.info("WorkingMemory remove | id={} success=True", memory_id)
                return True

        return False

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        return any(memory.id == memory_id for memory in self.memories)

    def clear(self) -> None:
        """清空所有工作记忆。"""
        count = len(self.memories)
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0
        log.info("WorkingMemory cleared | count={}", count)

    def get_stats(self) -> Dict[str, Any]:
        """获取工作记忆统计信息。"""
        # 过期清理（惰性）
        self._expire_old_memories()

        avg_importance = 0.0
        if self.memories:
            avg_importance = sum(m.importance for m in self.memories) / len(self.memories)

        return {
            "count": len(self.memories),
            "total_count": len(self.memories),
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": avg_importance,
            "capacity_usage": len(self.memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working",
        }

    # ------------------------------------------------------------------ #
    # Forgetting Mechanism
    # ------------------------------------------------------------------ #

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 1,
    ) -> int:
        """工作记忆遗忘机制（硬删除）。"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove: List[str] = []

        # 始终先执行 TTL 过期
        self._expire_old_memories()

        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)

        elif strategy == "time_based":
            # 工作记忆以小时计算
            cutoff_time = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_time:
                    to_remove.append(memory.id)

        elif strategy == "capacity_based":
            if len(self.memories) > self.max_capacity:
                # 按优先级排序，删除最低的
                sorted_memories = sorted(self.memories, key=lambda m: self._calculate_priority(m))
                excess_count = len(self.memories) - self.max_capacity
                for memory in sorted_memories[:excess_count]:
                    to_remove.append(memory.id)

        # 执行删除
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
                log.info("WorkingMemory forget | id={}... strategy={}", memory_id[:8], strategy)

        log.info("WorkingMemory forget completed | strategy={} removed={}", strategy, forgotten_count)
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆。"""
        return self.memories.copy()

    # ------------------------------------------------------------------ #
    # Convenience Methods
    # ------------------------------------------------------------------ #

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """获取最近的记忆。"""
        sorted_memories = sorted(self.memories, key=lambda x: x.timestamp, reverse=True)
        return sorted_memories[:limit]

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """获取重要记忆。"""
        sorted_memories = sorted(self.memories, key=lambda x: x.importance, reverse=True)
        return sorted_memories[:limit]

    def get_context_summary(self, max_length: int = 500) -> str:
        """获取上下文摘要。"""
        if not self.memories:
            return "No working memories available."

        # 按重要性和时间排序
        sorted_memories = sorted(
            self.memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True,
        )

        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content
            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                # 截断最后一个记忆
                remaining = max_length - current_length
                if remaining > 50:  # 至少保留 50 个字符
                    summary_parts.append(content[:remaining] + "...")
                break

        return "Working Memory Context:\n" + "\n".join(summary_parts)

    # ------------------------------------------------------------------ #
    # Private Helpers
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str) -> int:
        """简单的 token 计数（按空格分词）。"""
        return len(text.split())

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """计算记忆优先级。"""
        # 基础优先级 = 重要性
        priority = memory.importance

        # 时间衰减
        time_decay = self._calculate_time_decay(memory.timestamp)
        priority *= time_decay

        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """计算时间衰减因子。"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600

        # 指数衰减（工作记忆衰减更快）
        decay_factor = self.config.decay_factor ** (hours_passed / 6)  # 每 6 小时衰减
        return max(0.1, decay_factor)  # 最小保持 10% 的权重

    def _enforce_capacity_limits(self) -> None:
        """强制执行容量限制。"""
        # 检查记忆数量限制
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()

        # 检查 token 限制
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self) -> None:
        """按 TTL 清理过期记忆。"""
        if not self.memories:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        # 过滤保留的记忆
        kept: List[MemoryItem] = []
        removed_token_sum = 0

        for m in self.memories:
            if m.timestamp >= cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += self._count_tokens(m.content)

        if len(kept) == len(self.memories):
            return

        removed_count = len(self.memories) - len(kept)

        # 覆盖列表与 token
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)

        # 重建堆
        self._rebuild_heap()

        log.debug("WorkingMemory expired | removed={}", removed_count)

    def _remove_lowest_priority_memory(self) -> None:
        """删除优先级最低的记忆。"""
        if not self.memories:
            return

        # 找到优先级最低的记忆
        lowest_priority = float("inf")
        lowest_memory = None

        for memory in self.memories:
            priority = self._calculate_priority(memory)
            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory

        if lowest_memory:
            self.remove(lowest_memory.id)

    def _rebuild_heap(self) -> None:
        """重建优先级堆。"""
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))


__all__ = ["WorkingMemory"]
