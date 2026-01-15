"""感知记忆：长期多模态数据存储与检索。

特点：
- 支持多模态数据（文本、图像、音频、视频等）
- 结构化元数据 + 向量索引（SQLite + Qdrant）
- 同模态语义检索
- 懒加载编码：文本用嵌入模型；图像/音频用轻量确定性哈希向量（无 CLIP/CLAP 依赖时）
- 遗忘机制（硬删除）
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService, get_dimension, get_text_embedder
from veragents.memory.storage.document_store import DocumentStore, SQLiteDocumentStore
from veragents.memory.storage.qdrant_store import QdrantStore


@dataclass
class Perception:
    """感知数据实体。"""

    perception_id: str
    data: Any
    modality: str  # text, image, audio, video, structured
    encoding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    data_hash: str = ""

    def __post_init__(self):
        if not self.data_hash:
            self.data_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """计算数据哈希。"""
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()


class PerceptualMemory(BaseMemory):
    """感知记忆实现。

    特点：
    - 支持多模态数据（文本、图像、音频等）
    - 同模态相似性搜索
    - 感知数据的语义理解
    - SQLite（权威存储）+ Qdrant（向量检索）双存储架构
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

        # 感知数据存储（内存缓存）
        self.perceptions: Dict[str, Perception] = {}
        self.perceptual_memories: List[MemoryItem] = []

        # 模态索引
        self.modality_index: Dict[str, List[str]] = {}  # modality -> perception_ids

        # 支持的模态
        self.supported_modalities = set(self.config.perceptual_memory_modalities)

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
            self.text_embedder = embedding._model if hasattr(embedding, "_model") else get_text_embedder()
        else:
            self.text_embedder = get_text_embedder()
            self.embedding = EmbeddingService(self.text_embedder)

        # 嵌入维度
        self.vector_dim = get_dimension(getattr(self.text_embedder, "dimension", 384))

        # 向量存储（Qdrant）- 使用统一集合，模态作为 payload 过滤
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            collection_name = os.getenv("QDRANT_COLLECTION", "veragents_vectors")
            self.vector_store = QdrantStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection=f"{collection_name}_perceptual",
                vector_size=self.vector_dim,
            )

        # 初始化编码器
        self.encoders = self._init_encoders()

        log.info(
            "PerceptualMemory initialized | storage={} vector_store={} embedding={} modalities={}",
            type(self.document_store).__name__,
            type(self.vector_store).__name__,
            type(self.embedding).__name__,
            list(self.supported_modalities),
        )

    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #

    def add(self, memory_item: MemoryItem) -> str:
        """添加感知记忆。"""
        modality = memory_item.metadata.get("modality", "text")
        raw_data = memory_item.metadata.get("raw_data", memory_item.content)

        if modality not in self.supported_modalities:
            log.warning("Unsupported modality | modality={} supported={}", modality, self.supported_modalities)
            modality = "text"  # 降级为文本模态

        # 编码感知数据
        perception = self._encode_perception(raw_data, modality, memory_item.id)

        # 缓存与索引
        self.perceptions[perception.perception_id] = perception
        if modality not in self.modality_index:
            self.modality_index[modality] = []
        self.modality_index[modality].append(perception.perception_id)

        # 存储记忆项（缓存）
        memory_item.metadata["perception_id"] = perception.perception_id
        memory_item.metadata["modality"] = modality
        self.perceptual_memories.append(memory_item)

        # 1) SQLite 权威入库
        ts_int = int(memory_item.timestamp.timestamp())
        self.document_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="perceptual",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "perception_id": perception.perception_id,
                "modality": modality,
                "data_hash": perception.data_hash,
                "context": memory_item.metadata.get("context", {}),
                "tags": memory_item.metadata.get("tags", []),
            },
        )

        # 2) Qdrant 向量入库
        try:
            self.vector_store.add_vectors(
                vectors=[perception.encoding],
                metadata=[
                    {
                        "memory_id": memory_item.id,
                        "user_id": memory_item.user_id,
                        "memory_type": "perceptual",
                        "modality": modality,
                        "importance": memory_item.importance,
                        "content": memory_item.content[:500],
                        "timestamp": ts_int,
                    }
                ],
                ids=[memory_item.id],
            )
        except Exception as e:
            log.warning("Qdrant upsert failed for perceptual memory | id={} error={}", memory_item.id, e)

        log.info(
            "PerceptualMemory add | id={} user={} modality={} content_len={}",
            memory_item.id,
            memory_item.user_id,
            modality,
            len(memory_item.content),
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索感知记忆（同模态向量检索）。"""
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality = kwargs.get("query_modality", target_modality or "text")

        # 向量检索
        hits: List[Dict[str, Any]] = []
        try:
            qvec = self._encode_data(query, query_modality)
            where: Dict[str, Any] = {"memory_type": "perceptual"}
            if user_id:
                where["user_id"] = user_id
            if target_modality:
                where["modality"] = target_modality
            hits = self.vector_store.search_similar(
                query_vector=qvec,
                limit=max(limit * 5, 20),
                where=where,
            )
        except Exception as e:
            log.warning("Qdrant search failed | error={}", e)
            hits = []

        # 融合排序
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen: set = set()

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            if target_modality and meta.get("modality") != target_modality:
                continue

            doc = self.document_store.get_memory(mem_id)
            if not doc:
                continue

            # 计算综合分数
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))

            base_relevance = vec_score * 0.8 + recency_score * 0.2
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

        # 关键词回退
        if not results:
            query_lower = query.lower()
            for m in self.perceptual_memories:
                if target_modality and m.metadata.get("modality") != target_modality:
                    continue
                if query_lower in (m.content or "").lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(m.timestamp.timestamp())) / 86400.0))
                    keyword_score = 0.5
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (m.importance * 0.4)
                    combined = base_relevance * importance_weight
                    results.append((combined, m))

        results.sort(key=lambda x: x[0], reverse=True)
        log.info("PerceptualMemory retrieve | query_len={} modality={} returned={}", len(query), target_modality, min(len(results), limit))
        return [it for _, it in results[:limit]]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新感知记忆。"""
        updated = False
        modality_cache = None

        for memory in self.perceptual_memories:
            if memory.id == memory_id:
                if content is not None:
                    memory.content = content
                if importance is not None:
                    memory.importance = importance
                if metadata is not None:
                    memory.metadata.update(metadata)
                modality_cache = memory.metadata.get("modality", "text")
                updated = True
                break

        # 更新 SQLite
        doc_updated = self.document_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata,
        )

        # 如内容或原始数据改变，重嵌入
        if content is not None or (metadata and "raw_data" in metadata):
            modality = metadata.get("modality", modality_cache or "text") if metadata else (modality_cache or "text")
            raw = metadata.get("raw_data", content) if metadata else content
            try:
                perception = self._encode_perception(raw or "", modality, memory_id)
                doc = self.document_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "perceptual",
                    "modality": modality,
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
                    "content": content[:500] if content else (doc.get("content", "")[:500] if doc else ""),
                    "timestamp": doc["timestamp"] if doc else int(datetime.now().timestamp()),
                }
                self.vector_store.add_vectors(
                    vectors=[perception.encoding],
                    metadata=[payload],
                    ids=[memory_id],
                )
            except Exception as e:
                log.warning("Qdrant update failed | id={} error={}", memory_id, e)

        log.info("PerceptualMemory update | id={} success={}", memory_id, updated or doc_updated)
        return updated or doc_updated

    def remove(self, memory_id: str) -> bool:
        """删除感知记忆。"""
        removed = False
        for i, memory in enumerate(self.perceptual_memories):
            if memory.id == memory_id:
                removed_memory = self.perceptual_memories.pop(i)
                perception_id = removed_memory.metadata.get("perception_id")
                if perception_id and perception_id in self.perceptions:
                    perception = self.perceptions.pop(perception_id)
                    modality = perception.modality
                    if modality in self.modality_index:
                        if perception_id in self.modality_index[modality]:
                            self.modality_index[modality].remove(perception_id)
                        if not self.modality_index[modality]:
                            del self.modality_index[modality]
                removed = True
                break

        # 权威库删除
        doc_deleted = self.document_store.delete_memory(memory_id)

        # 向量库删除
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception as e:
            log.warning("Qdrant delete failed | id={} error={}", memory_id, e)

        log.info("PerceptualMemory remove | id={} success={}", memory_id, removed or doc_deleted)
        return removed or doc_deleted

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        if any(memory.id == memory_id for memory in self.perceptual_memories):
            return True
        doc = self.document_store.get_memory(memory_id)
        return doc is not None

    def clear(self) -> None:
        """清空所有感知记忆。"""
        self.perceptual_memories.clear()
        self.perceptions.clear()
        self.modality_index.clear()

        # 删除 SQLite 中的 perceptual 记录
        docs = self.document_store.search_memories(memory_type="perceptual", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.document_store.delete_memory(mid)

        # 删除 Qdrant 向量
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception as e:
            log.warning("Qdrant clear failed | error={}", e)

        log.info("PerceptualMemory cleared | count={}", len(ids))

    def get_stats(self) -> Dict[str, Any]:
        """获取感知记忆统计信息。"""
        db_stats = self.document_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {"store_type": "qdrant"}

        modality_counts = {modality: len(ids) for modality, ids in self.modality_index.items()}
        avg_importance = 0.0
        if self.perceptual_memories:
            avg_importance = sum(m.importance for m in self.perceptual_memories) / len(self.perceptual_memories)

        return {
            "count": len(self.perceptual_memories),
            "total_count": len(self.perceptual_memories),
            "perceptions_count": len(self.perceptions),
            "modality_counts": modality_counts,
            "supported_modalities": list(self.supported_modalities),
            "avg_importance": avg_importance,
            "memory_type": "perceptual",
            "vector_store": vs_stats,
            "document_store": {
                k: v
                for k, v in db_stats.items()
                if k.endswith("_count") or k in ["store_type", "db_path"]
            },
        }

    # ------------------------------------------------------------------ #
    # Forgetting Mechanism
    # ------------------------------------------------------------------ #

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """感知记忆遗忘机制（硬删除）。"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove: List[str] = []

        for memory in self.perceptual_memories:
            should_forget = False

            if strategy == "importance_based":
                if memory.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if memory.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.perceptual_memories) > self.config.max_capacity:
                    sorted_memories = sorted(self.perceptual_memories, key=lambda m: m.importance)
                    excess_count = len(self.perceptual_memories) - self.config.max_capacity
                    if memory in sorted_memories[:excess_count]:
                        should_forget = True

            if should_forget:
                to_remove.append(memory.id)

        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
                log.info("PerceptualMemory forget | id={}... strategy={}", memory_id[:8], strategy)

        log.info("PerceptualMemory forget completed | strategy={} removed={}", strategy, forgotten_count)
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """获取所有感知记忆。"""
        return self.perceptual_memories.copy()

    # ------------------------------------------------------------------ #
    # Modality-specific Methods
    # ------------------------------------------------------------------ #

    def get_by_modality(self, modality: str, limit: int = 10) -> List[MemoryItem]:
        """按模态获取记忆。"""
        if modality not in self.modality_index:
            return []

        perception_ids = self.modality_index[modality]
        results = []

        for memory in self.perceptual_memories:
            if memory.metadata.get("perception_id") in perception_ids:
                results.append(memory)
                if len(results) >= limit:
                    break

        return results

    def cross_modal_search(
        self,
        query: Any,
        query_modality: str,
        target_modality: str = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """跨模态搜索（同模态检索，跨模态需要 CLIP/CLAP）。"""
        return self.retrieve(
            query=str(query),
            limit=limit,
            query_modality=query_modality,
            target_modality=target_modality,
        )

    def generate_content(self, prompt: str, target_modality: str) -> Optional[str]:
        """基于感知记忆生成内容（简化实现）。"""
        if target_modality not in self.supported_modalities:
            return None

        relevant_memories = self.retrieve(prompt, limit=3, target_modality=target_modality)

        if not relevant_memories:
            return None

        if target_modality == "text":
            contents = [memory.content for memory in relevant_memories]
            return f"基于感知记忆生成的内容：\n" + "\n".join(contents)

        return f"生成的 {target_modality} 内容（基于 {len(relevant_memories)} 个相关记忆）"

    # ------------------------------------------------------------------ #
    # Private Helpers
    # ------------------------------------------------------------------ #

    def _init_encoders(self) -> Dict[str, Any]:
        """初始化编码器。"""
        encoders = {}
        for modality in self.supported_modalities:
            if modality == "text":
                encoders[modality] = self._text_encoder
            elif modality == "image":
                encoders[modality] = self._image_encoder
            elif modality == "audio":
                encoders[modality] = self._audio_encoder
            else:
                encoders[modality] = self._default_encoder
        return encoders

    def _encode_perception(self, data: Any, modality: str, memory_id: str) -> Perception:
        """编码感知数据。"""
        encoding = self._encode_data(data, modality)

        perception = Perception(
            perception_id=f"perception_{memory_id}",
            data=data,
            modality=modality,
            encoding=encoding,
            metadata={"source": "memory_system"},
        )

        return perception

    def _encode_data(self, data: Any, modality: str) -> List[float]:
        """编码数据为固定维度向量。"""
        encoder = self.encoders.get(modality, self._default_encoder)
        vec = encoder(data)
        if not isinstance(vec, list):
            vec = list(vec)

        # 确保维度一致
        if len(vec) < self.vector_dim:
            vec = vec + [0.0] * (self.vector_dim - len(vec))
        elif len(vec) > self.vector_dim:
            vec = vec[: self.vector_dim]

        return vec

    def _text_encoder(self, text: str) -> List[float]:
        """文本编码器（使用嵌入模型）。"""
        try:
            emb = self.text_embedder.encode(text or "")
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            return emb
        except Exception:
            return self._hash_to_vector(text or "", self.vector_dim)

    def _image_encoder(self, image_data: Any) -> List[float]:
        """图像编码器（确定性哈希向量）。"""
        try:
            if isinstance(image_data, (bytes, bytearray)):
                data_bytes = bytes(image_data)
            elif isinstance(image_data, str) and os.path.exists(image_data):
                with open(image_data, "rb") as f:
                    data_bytes = f.read()
            else:
                data_bytes = str(image_data).encode("utf-8", errors="ignore")
            hex_str = hashlib.sha256(data_bytes).hexdigest()
            return self._hash_to_vector(hex_str, self.vector_dim)
        except Exception:
            return self._hash_to_vector(str(image_data), self.vector_dim)

    def _audio_encoder(self, audio_data: Any) -> List[float]:
        """音频编码器（确定性哈希向量）。"""
        try:
            if isinstance(audio_data, (bytes, bytearray)):
                data_bytes = bytes(audio_data)
            elif isinstance(audio_data, str) and os.path.exists(audio_data):
                with open(audio_data, "rb") as f:
                    data_bytes = f.read()
            else:
                data_bytes = str(audio_data).encode("utf-8", errors="ignore")
            hex_str = hashlib.sha256(data_bytes).hexdigest()
            return self._hash_to_vector(hex_str, self.vector_dim)
        except Exception:
            return self._hash_to_vector(str(audio_data), self.vector_dim)

    def _default_encoder(self, data: Any) -> List[float]:
        """默认编码器。"""
        try:
            return self._text_encoder(str(data))
        except Exception:
            return self._hash_to_vector(str(data), self.vector_dim)

    def _hash_to_vector(self, data_str: str, dim: int) -> List[float]:
        """将字符串哈希为固定维度的 [0,1] 向量（确定性）。"""
        seed = int(hashlib.sha256(data_str.encode("utf-8", errors="ignore")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.random() for _ in range(dim)]


__all__ = ["Perception", "PerceptualMemory"]
