"""情景记忆：记录具体交互事件，支持时间序列组织、模式识别和遗忘机制。

特点：
- 存储具体的交互事件
- 包含丰富的上下文信息（session、participants、tags 等）
- 按时间序列组织
- 支持模式识别和回溯
- SQLite（权威存储）+ Qdrant（向量检索）双存储架构
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService, get_text_embedder, get_dimension
from veragents.memory.storage.document_store import DocumentStore, SQLiteDocumentStore
from veragents.memory.storage.qdrant_store import QdrantStore


@dataclass
class Episode:
    """情景记忆中的单个情景。"""

    episode_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    importance: float = 0.5


class EpisodicMemory(BaseMemory):
    """情景记忆实现。

    特点：
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    - 遗忘机制（硬删除）
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[QdrantStore] = None,
        embedding: Optional[EmbeddingService] = None,
    ):
        super().__init__(config or MemoryConfig(), storage_backend=document_store)

        # 本地缓存（内存）
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids

        # 模式识别缓存
        self.patterns_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.last_pattern_analysis: Optional[datetime] = None

        # 权威文档存储（SQLite）
        if document_store is not None:
            self.document_store = document_store
        else:
            db_dir = self.config.storage_path if hasattr(self.config, "storage_path") else "./memory_data"
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "memory.db")
            self.document_store = SQLiteDocumentStore(db_path=db_path)

        # 统一嵌入模型
        if embedding is not None:
            self.embedding = embedding
            self.embedder = embedding._model if hasattr(embedding, "_model") else get_text_embedder()
        else:
            self.embedder = get_text_embedder()
            self.embedding = EmbeddingService(self.embedder)

        # 向量存储（Qdrant）
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            collection_name = os.getenv("QDRANT_COLLECTION", "veragents_vectors")
            vector_size = get_dimension(getattr(self.embedder, "dimension", 384))
            self.vector_store = QdrantStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection=collection_name,
                vector_size=vector_size,
            )

        log.info(
            "EpisodicMemory initialized | storage={} vector_store={} embedding={}",
            type(self.document_store).__name__,
            type(self.vector_store).__name__,
            type(self.embedding).__name__,
        )

    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆。"""
        # 从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])

        # 创建情景（内存缓存）
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance,
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 1) 权威存储（SQLite）
        ts_int = int(memory_item.timestamp.timestamp())
        self.document_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags,
            },
        )

        # 2) 向量索引（Qdrant）
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[
                    {
                        "memory_id": memory_item.id,
                        "user_id": memory_item.user_id,
                        "memory_type": "episodic",
                        "importance": memory_item.importance,
                        "session_id": session_id,
                        "content": memory_item.content[:500],  # 截断避免过长
                        "timestamp": ts_int,
                    }
                ],
                ids=[memory_item.id],
            )
        except Exception as e:
            # 向量入库失败不影响权威存储
            log.warning("Qdrant upsert failed for episodic memory | id={} error={}", memory_item.id, e)

        log.info(
            "EpisodicMemory add | id={} user={} session={} content_len={}",
            memory_item.id,
            memory_item.user_id,
            session_id,
            len(memory_item.content),
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索情景记忆（结构化过滤 + 语义向量检索）。"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold")

        # 结构化过滤候选（来自权威库）
        candidate_ids: Optional[set] = None
        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            docs = self.document_store.search_memories(
                user_id=user_id,
                memory_type="episodic",
                start_time=start_ts,
                end_time=end_ts,
                importance_threshold=importance_threshold,
                limit=1000,
            )
            candidate_ids = {d["memory_id"] for d in docs}

        # 向量检索（Qdrant）
        hits: List[Dict[str, Any]] = []
        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec, "tolist"):
                query_vec = query_vec.tolist()
            where: Dict[str, Any] = {"memory_type": "episodic"}
            if user_id:
                where["user_id"] = user_id
            hits = self.vector_store.search_similar(
                query_vector=query_vec,
                limit=max(limit * 5, 20),
                where=where,
            )
        except Exception as e:
            log.warning("Qdrant search failed | error={}", e)
            hits = []

        # 过滤与重排
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen: set = set()

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue

            # 从权威库读取完整记录
            doc = self.document_store.get_memory(mem_id)
            if not doc:
                continue

            # 计算综合分数：基础相似度 0.8 + 近因性 0.2，重要性作为加权因子
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))

            # 基础相似度得分
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            # 重要性作为乘法加权因子，范围 [0.8, 1.2]
            importance_weight = 0.8 + (imp * 0.4)
            combined = base_relevance * importance_weight

            properties = doc.get("properties", {}) or {}
            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **properties,
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score,
                },
            )
            results.append((combined, item))
            seen.add(mem_id)

        # 若向量检索无结果，回退到简单关键词匹配（内存缓存）
        if not results:
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id, session_id, time_range):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0))
                    keyword_score = 0.5  # 简单关键词匹配的基础分数
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined,
                        },
                    )
                    results.append((combined, item))

        results.sort(key=lambda x: x[0], reverse=True)
        log.info("EpisodicMemory retrieve | query_len={} returned={}", len(query), min(len(results), limit))
        return [it for _, it in results[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新情景记忆（SQLite 为权威，Qdrant 按需重嵌入）。"""
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        # 更新 SQLite
        doc_updated = self.document_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata,
        )

        # 如内容变更，重嵌入并 upsert 到 Qdrant
        if content is not None:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                # 获取更新后的记录以同步 payload
                doc = self.document_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "episodic",
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
                    "session_id": (doc.get("properties", {}) or {}).get("session_id"),
                    "content": content[:500],
                    "timestamp": doc["timestamp"] if doc else int(datetime.now().timestamp()),
                }
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[payload],
                    ids=[memory_id],
                )
            except Exception as e:
                log.warning("Qdrant update failed | id={} error={}", memory_id, e)

        log.info("EpisodicMemory update | id={} success={}", memory_id, updated or doc_updated)
        return updated or doc_updated

    def remove(self, memory_id: str) -> bool:
        """删除情景记忆（SQLite + Qdrant）。"""
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    if memory_id in self.sessions[session_id]:
                        self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break

        # 权威库删除
        doc_deleted = self.document_store.delete_memory(memory_id)

        # 向量库删除
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception as e:
            log.warning("Qdrant delete failed | id={} error={}", memory_id, e)

        log.info("EpisodicMemory remove | id={} success={}", memory_id, removed or doc_deleted)
        return removed or doc_deleted

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        # 先检查内存缓存
        if any(episode.episode_id == memory_id for episode in self.episodes):
            return True
        # 再检查权威库
        doc = self.document_store.get_memory(memory_id)
        return doc is not None

    def clear(self) -> None:
        """清空所有情景记忆（仅清理 episodic，不影响其他类型）。"""
        # 内存缓存
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

        # SQLite 内的 episodic 全部删除
        docs = self.document_store.search_memories(memory_type="episodic", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.document_store.delete_memory(mid)

        # Qdrant 按 ID 删除对应向量
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception as e:
            log.warning("Qdrant clear failed | error={}", e)

        log.info("EpisodicMemory cleared | count={}", len(ids))

    def get_stats(self) -> Dict[str, Any]:
        """获取情景记忆统计信息（合并 SQLite 与 Qdrant）。"""
        db_stats = self.document_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {"store_type": "qdrant"}

        avg_importance = 0.0
        if self.episodes:
            avg_importance = sum(e.importance for e in self.episodes) / len(self.episodes)

        return {
            "count": len(self.episodes),
            "total_count": len(self.episodes),
            "sessions_count": len(self.sessions),
            "avg_importance": avg_importance,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
            "document_store": {
                k: v
                for k, v in db_stats.items()
                if k.endswith("_count") or k in ["store_type", "db_path"]
            },
        }

    # ------------------------------------------------------------------ #
    # Session & Timeline
    # ------------------------------------------------------------------ #

    def get_session_episodes(self, session_id: str) -> List[Episode]:
        """获取指定会话的所有情景。"""
        if session_id not in self.sessions:
            return []

        episode_ids = self.sessions[session_id]
        return [e for e in self.episodes if e.episode_id in episode_ids]

    def get_timeline(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取时间线视图。"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)

        timeline = []
        for episode in episodes[:limit]:
            timeline.append(
                {
                    "episode_id": episode.episode_id,
                    "timestamp": episode.timestamp.isoformat(),
                    "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                    "session_id": episode.session_id,
                    "importance": episode.importance,
                    "outcome": episode.outcome,
                }
            )

        return timeline

    # ------------------------------------------------------------------ #
    # Pattern Recognition
    # ------------------------------------------------------------------ #

    def find_patterns(self, user_id: str = None, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """发现用户行为模式。"""
        # 检查缓存
        cache_key = f"{user_id}_{min_frequency}"
        if (
            cache_key in self.patterns_cache
            and self.last_pattern_analysis
            and (datetime.now() - self.last_pattern_analysis).total_seconds() < 3600
        ):
            return self.patterns_cache[cache_key]

        # 过滤情景
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]

        if not episodes:
            return []

        # 简单的模式识别：基于内容关键词
        keyword_patterns: Dict[str, int] = {}
        context_patterns: Dict[str, int] = {}

        for episode in episodes:
            # 提取关键词
            words = episode.content.lower().split()
            for word in words:
                if len(word) > 3:  # 忽略短词
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + 1

            # 提取上下文模式
            for key, value in episode.context.items():
                pattern_key = f"{key}:{value}"
                context_patterns[pattern_key] = context_patterns.get(pattern_key, 0) + 1

        # 筛选频繁模式
        patterns = []

        for keyword, frequency in keyword_patterns.items():
            if frequency >= min_frequency:
                patterns.append(
                    {
                        "type": "keyword",
                        "pattern": keyword,
                        "frequency": frequency,
                        "confidence": frequency / len(episodes),
                    }
                )

        for context_pattern, frequency in context_patterns.items():
            if frequency >= min_frequency:
                patterns.append(
                    {
                        "type": "context",
                        "pattern": context_pattern,
                        "frequency": frequency,
                        "confidence": frequency / len(episodes),
                    }
                )

        # 按频率排序
        patterns.sort(key=lambda x: x["frequency"], reverse=True)

        # 缓存结果
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()

        log.info("EpisodicMemory find_patterns | user={} patterns_found={}", user_id, len(patterns))
        return patterns

    # ------------------------------------------------------------------ #
    # Forgetting Mechanism
    # ------------------------------------------------------------------ #

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """情景记忆遗忘机制（硬删除）。

        Args:
            strategy: 遗忘策略，可选 'importance_based', 'time_based', 'capacity_based'
            threshold: 重要性阈值（用于 importance_based）
            max_age_days: 最大保留天数（用于 time_based）

        Returns:
            遗忘的记忆数量
        """
        forgotten_count = 0
        current_time = datetime.now()
        to_remove: List[str] = []

        for episode in self.episodes:
            should_forget = False

            if strategy == "importance_based":
                # 基于重要性遗忘
                if episode.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                # 基于时间遗忘
                cutoff_time = current_time - timedelta(days=max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                # 基于容量遗忘（保留最重要的）
                if len(self.episodes) > self.config.max_capacity:
                    sorted_episodes = sorted(self.episodes, key=lambda e: e.importance)
                    excess_count = len(self.episodes) - self.config.max_capacity
                    if episode in sorted_episodes[:excess_count]:
                        should_forget = True

            if should_forget:
                to_remove.append(episode.episode_id)

        # 执行硬删除
        for episode_id in to_remove:
            if self.remove(episode_id):
                forgotten_count += 1
                log.info("EpisodicMemory forget | id={}... strategy={}", episode_id[:8], strategy)

        log.info("EpisodicMemory forget completed | strategy={} removed={}", strategy, forgotten_count)
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """获取所有情景记忆（转换为 MemoryItem 格式）。"""
        memory_items = []
        for episode in self.episodes:
            memory_item = MemoryItem(
                id=episode.episode_id,
                content=episode.content,
                memory_type="episodic",
                user_id=episode.user_id,
                timestamp=episode.timestamp,
                importance=episode.importance,
                metadata={
                    "session_id": episode.session_id,
                    "context": episode.context,
                    "outcome": episode.outcome,
                },
            )
            memory_items.append(memory_item)
        return memory_items

    # ------------------------------------------------------------------ #
    # Private Helpers
    # ------------------------------------------------------------------ #

    def _filter_episodes(
        self,
        user_id: str = None,
        session_id: str = None,
        time_range: Tuple[datetime, datetime] = None,
    ) -> List[Episode]:
        """过滤情景。"""
        filtered = self.episodes

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]

        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]

        return filtered

    def _calculate_time_span(self) -> float:
        """计算记忆时间跨度（天）。"""
        if not self.episodes:
            return 0.0

        timestamps = [e.timestamp for e in self.episodes]
        min_time = min(timestamps)
        max_time = max(timestamps)

        return (max_time - min_time).days


__all__ = ["Episode", "EpisodicMemory"]
