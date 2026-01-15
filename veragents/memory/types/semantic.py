"""语义记忆：结合向量检索和知识图谱的混合语义记忆。

特点：
- 使用 API 嵌入模型进行文本嵌入
- 向量相似度检索进行快速初筛
- 知识图谱（Neo4j）存储实体和关系
- SQLite 作为权威文档存储
- 混合检索策略优化结果质量
- 遗忘机制（硬删除）
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger as log

from veragents.memory.base import BaseMemory, MemoryConfig, MemoryItem
from veragents.memory.embedding import EmbeddingService, get_dimension, get_text_embedder
from veragents.memory.storage.document_store import DocumentStore, SQLiteDocumentStore
from veragents.memory.storage.neo4j_store import Neo4jStore
from veragents.memory.storage.qdrant_store import QdrantStore


@dataclass
class Entity:
    """实体类。"""

    entity_id: str
    name: str
    entity_type: str = "MISC"  # PERSON, ORG, PRODUCT, SKILL, CONCEPT 等
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency,
        }


@dataclass
class Relation:
    """关系类。"""

    from_entity: str
    to_entity: str
    relation_type: str
    strength: float = 1.0
    evidence: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency,
        }


class SemanticMemory(BaseMemory):
    """语义记忆实现。

    特点：
    - 使用 API 嵌入模型进行文本嵌入
    - 向量检索进行快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量 + 图 + 语义推理
    - SQLite（权威存储）+ Qdrant（向量检索）+ Neo4j（知识图谱）
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        document_store: Optional[DocumentStore] = None,
        graph_store: Optional[Neo4jStore] = None,
        vector_store: Optional[QdrantStore] = None,
        embedding: Optional[EmbeddingService] = None,
    ):
        super().__init__(config or MemoryConfig(), storage_backend=document_store)

        # 记忆存储（内存缓存）
        self.semantic_memories: List[MemoryItem] = []

        # 实体和关系缓存
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

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
                collection=f"{collection_name}_semantic",
                vector_size=vector_size,
            )

        # 知识图谱存储（Neo4j）- 可选
        if graph_store is not None:
            self.graph_store = graph_store
        else:
            try:
                neo4j_uri = os.getenv("NEO4J_URI")
                neo4j_user = os.getenv("NEO4J_USER")
                neo4j_password = os.getenv("NEO4J_PASSWORD")
                if neo4j_uri and neo4j_user and neo4j_password:
                    self.graph_store = Neo4jStore(
                        uri=neo4j_uri,
                        user=neo4j_user,
                        password=neo4j_password,
                    )
                else:
                    self.graph_store = None
                    log.warning("Neo4j config not found, graph features disabled")
            except Exception as e:
                self.graph_store = None
                log.warning("Neo4j init failed, graph features disabled | error={}", e)

        log.info(
            "SemanticMemory initialized | storage={} vector_store={} graph_store={} embedding={}",
            type(self.document_store).__name__,
            type(self.vector_store).__name__,
            type(self.graph_store).__name__ if self.graph_store else None,
            type(self.embedding).__name__,
        )

    # ------------------------------------------------------------------ #
    # CRUD Operations
    # ------------------------------------------------------------------ #

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆。"""
        # 1. 生成文本嵌入
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
        except Exception as e:
            log.warning("Embedding failed | error={}", e)
            embedding = None

        # 2. 提取实体和关系（简化版本）
        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)

        # 3. 存储到 Neo4j 图数据库
        if self.graph_store:
            for entity in entities:
                self._add_entity_to_graph(entity, memory_item)
            for relation in relations:
                self._add_relation_to_graph(relation, memory_item)

        # 4. 存储到 SQLite 权威库
        ts_int = int(memory_item.timestamp.timestamp())
        self.document_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="semantic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "entities": [e.entity_id for e in entities],
                "relations": [f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations],
                "context": memory_item.metadata.get("context", {}),
                "tags": memory_item.metadata.get("tags", []),
            },
        )

        # 5. 存储到 Qdrant 向量数据库
        if embedding:
            try:
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    metadata=[
                        {
                            "memory_id": memory_item.id,
                            "user_id": memory_item.user_id,
                            "memory_type": "semantic",
                            "importance": memory_item.importance,
                            "content": memory_item.content[:500],
                            "timestamp": ts_int,
                            "entity_count": len(entities),
                            "relation_count": len(relations),
                        }
                    ],
                    ids=[memory_item.id],
                )
            except Exception as e:
                log.warning("Qdrant upsert failed for semantic memory | id={} error={}", memory_item.id, e)

        # 6. 更新元数据并存储到内存缓存
        memory_item.metadata["entities"] = [e.entity_id for e in entities]
        memory_item.metadata["relations"] = [f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations]
        self.semantic_memories.append(memory_item)

        log.info(
            "SemanticMemory add | id={} user={} entities={} relations={}",
            memory_item.id,
            memory_item.user_id,
            len(entities),
            len(relations),
        )
        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索语义记忆（向量检索 + 图检索混合）。"""
        user_id = kwargs.get("user_id")

        # 1. 向量检索
        vector_results = self._vector_search(query, limit * 2, user_id)

        # 2. 图检索
        graph_results = self._graph_search(query, limit * 2, user_id)

        # 3. 混合排序
        combined_results = self._combine_and_rank_results(vector_results, graph_results, query, limit)

        # 4. 计算概率（softmax 归一化）
        scores = [r.get("combined_score", 0.0) for r in combined_results]
        if scores:
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            denom = sum(exps) or 1.0
            probs = [e / denom for e in exps]
        else:
            probs = []

        # 5. 转换为 MemoryItem
        result_memories = []
        for idx, result in enumerate(combined_results):
            memory_id = result.get("memory_id")

            # 处理时间戳
            timestamp = result.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            memory_item = MemoryItem(
                id=memory_id,
                content=result.get("content", ""),
                memory_type="semantic",
                user_id=result.get("user_id", "default"),
                timestamp=timestamp,
                importance=result.get("importance", 0.5),
                metadata={
                    **result.get("properties", {}),
                    "combined_score": result.get("combined_score", 0.0),
                    "vector_score": result.get("vector_score", 0.0),
                    "graph_score": result.get("graph_score", 0.0),
                    "probability": probs[idx] if idx < len(probs) else 0.0,
                },
            )
            result_memories.append(memory_item)

        log.info("SemanticMemory retrieve | query_len={} returned={}", len(query), len(result_memories))
        return result_memories[:limit]

    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """更新语义记忆。"""
        updated = False
        for memory in self.semantic_memories:
            if memory.id == memory_id:
                if content is not None:
                    # 重新提取实体和关系
                    entities = self._extract_entities(content)
                    relations = self._extract_relations(content, entities)
                    memory.content = content
                    memory.metadata["entities"] = [e.entity_id for e in entities]
                    memory.metadata["relations"] = [f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations]
                if importance is not None:
                    memory.importance = importance
                if metadata is not None:
                    memory.metadata.update(metadata)
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
                doc = self.document_store.get_memory(memory_id)
                payload = {
                    "memory_id": memory_id,
                    "user_id": doc["user_id"] if doc else "",
                    "memory_type": "semantic",
                    "importance": (doc.get("importance") if doc else importance) or 0.5,
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

        log.info("SemanticMemory update | id={} success={}", memory_id, updated or doc_updated)
        return updated or doc_updated

    def remove(self, memory_id: str) -> bool:
        """删除语义记忆。"""
        removed = False
        for i, memory in enumerate(self.semantic_memories):
            if memory.id == memory_id:
                removed_memory = self.semantic_memories.pop(i)
                # 清理关联的实体缓存
                entity_ids = removed_memory.metadata.get("entities", [])
                for eid in entity_ids:
                    if eid in self.entities:
                        self.entities[eid].frequency -= 1
                        if self.entities[eid].frequency <= 0:
                            del self.entities[eid]
                removed = True
                break

        # 权威库删除
        doc_deleted = self.document_store.delete_memory(memory_id)

        # 向量库删除
        try:
            self.vector_store.delete_memories([memory_id])
        except Exception as e:
            log.warning("Qdrant delete failed | id={} error={}", memory_id, e)

        log.info("SemanticMemory remove | id={} success={}", memory_id, removed or doc_deleted)
        return removed or doc_deleted

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        if any(memory.id == memory_id for memory in self.semantic_memories):
            return True
        doc = self.document_store.get_memory(memory_id)
        return doc is not None

    def clear(self) -> None:
        """清空所有语义记忆。"""
        # 清空内存缓存
        self.semantic_memories.clear()
        self.entities.clear()
        self.relations.clear()

        # 清空 SQLite 中的 semantic 记录
        docs = self.document_store.search_memories(memory_type="semantic", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.document_store.delete_memory(mid)

        # 清空 Qdrant 向量
        try:
            if ids:
                self.vector_store.delete_memories(ids)
        except Exception as e:
            log.warning("Qdrant clear failed | error={}", e)

        # 清空 Neo4j 图
        if self.graph_store:
            try:
                self.graph_store.clear_all()
            except Exception as e:
                log.warning("Neo4j clear failed | error={}", e)

        log.info("SemanticMemory cleared | count={}", len(ids))

    def get_stats(self) -> Dict[str, Any]:
        """获取语义记忆统计信息。"""
        db_stats = self.document_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_collection_stats()
        except Exception:
            vs_stats = {"store_type": "qdrant"}

        graph_stats = {}
        if self.graph_store:
            try:
                graph_stats = self.graph_store.get_stats()
            except Exception:
                graph_stats = {}

        avg_importance = 0.0
        if self.semantic_memories:
            avg_importance = sum(m.importance for m in self.semantic_memories) / len(self.semantic_memories)

        return {
            "count": len(self.semantic_memories),
            "total_count": len(self.semantic_memories),
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_edges": graph_stats.get("total_relationships", 0),
            "avg_importance": avg_importance,
            "memory_type": "semantic",
            "vector_store": vs_stats,
            "document_store": {
                k: v
                for k, v in db_stats.items()
                if k.endswith("_count") or k in ["store_type", "db_path"]
            },
            "graph_store": graph_stats,
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
        """语义记忆遗忘机制（硬删除）。"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove: List[str] = []

        for memory in self.semantic_memories:
            should_forget = False

            if strategy == "importance_based":
                if memory.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if memory.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.semantic_memories) > self.config.max_capacity:
                    sorted_memories = sorted(self.semantic_memories, key=lambda m: m.importance)
                    excess_count = len(self.semantic_memories) - self.config.max_capacity
                    if memory in sorted_memories[:excess_count]:
                        should_forget = True

            if should_forget:
                to_remove.append(memory.id)

        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
                log.info("SemanticMemory forget | id={}... strategy={}", memory_id[:8], strategy)

        log.info("SemanticMemory forget completed | strategy={} removed={}", strategy, forgotten_count)
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """获取所有语义记忆。"""
        return self.semantic_memories.copy()

    # ------------------------------------------------------------------ #
    # Entity & Relation Methods
    # ------------------------------------------------------------------ #

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体。"""
        return self.entities.get(entity_id)

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """搜索实体。"""
        query_lower = query.lower()
        scored_entities = []

        for entity in self.entities.values():
            score = 0.0
            if query_lower in entity.name.lower():
                score += 2.0
            if query_lower in entity.entity_type.lower():
                score += 1.0
            if query_lower in entity.description.lower():
                score += 0.5
            score *= math.log(1 + entity.frequency)

            if score > 0:
                scored_entities.append((score, entity))

        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_entities[:limit]]

    def get_related_entities(
        self,
        entity_id: str,
        relation_types: List[str] = None,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """获取相关实体。"""
        if not self.graph_store:
            return []

        try:
            related_entities = self.graph_store.find_related_entities(
                entity_id=entity_id,
                relationship_types=relation_types,
                max_depth=max_hops,
                limit=50,
            )

            related = []
            for entity_data in related_entities:
                entity_obj = self.entities.get(entity_data.get("id"))
                if not entity_obj:
                    entity_obj = Entity(
                        entity_id=entity_data.get("id", entity_id),
                        name=entity_data.get("name", ""),
                        entity_type=entity_data.get("type", "MISC"),
                    )

                related.append(
                    {
                        "entity": entity_obj,
                        "relation_type": entity_data.get("relationship_path", ["RELATED"])[-1]
                        if entity_data.get("relationship_path")
                        else "RELATED",
                        "strength": 1.0 / max(entity_data.get("distance", 1), 1),
                        "distance": entity_data.get("distance", max_hops),
                    }
                )

            related.sort(key=lambda x: (x["distance"], -x["strength"]))
            return related
        except Exception as e:
            log.error("Get related entities failed | error={}", e)
            return []

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """导出知识图谱。"""
        graph_stats = {}
        if self.graph_store:
            try:
                graph_stats = self.graph_store.get_stats()
            except Exception:
                pass

        return {
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "relations": [relation.to_dict() for relation in self.relations],
            "graph_stats": {
                "total_nodes": graph_stats.get("total_nodes", 0),
                "entity_nodes": graph_stats.get("entity_nodes", 0),
                "memory_nodes": graph_stats.get("memory_nodes", 0),
                "total_relationships": graph_stats.get("total_relationships", 0),
                "cached_entities": len(self.entities),
                "cached_relations": len(self.relations),
            },
        }

    # ------------------------------------------------------------------ #
    # Private Helpers
    # ------------------------------------------------------------------ #

    def _vector_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """向量搜索。"""
        try:
            query_embedding = self.embedder.encode(query)
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            where: Dict[str, Any] = {"memory_type": "semantic"}
            if user_id:
                where["user_id"] = user_id

            results = self.vector_store.search_similar(
                query_vector=query_embedding,
                limit=limit,
                where=where,
            )

            formatted_results = []
            for result in results:
                formatted_result = {
                    "id": result["id"],
                    "memory_id": result["metadata"].get("memory_id", result["id"]),
                    "score": result["score"],
                    **result["metadata"],
                }
                formatted_results.append(formatted_result)

            return formatted_results
        except Exception as e:
            log.warning("Vector search failed | error={}", e)
            return []

    def _graph_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """图搜索。"""
        if not self.graph_store:
            return []

        try:
            # 从查询中提取实体
            query_entities = self._extract_entities(query)

            if not query_entities:
                # 按名称搜索
                entities_by_name = self.graph_store.search_entities_by_name(name_pattern=query, limit=10)
                if entities_by_name:
                    query_entities = [
                        Entity(
                            entity_id=e["id"],
                            name=e.get("name", ""),
                            entity_type=e.get("type", "MISC"),
                        )
                        for e in entities_by_name[:3]
                    ]
                else:
                    return []

            # 查找相关记忆
            related_memory_ids: set = set()
            for entity in query_entities:
                try:
                    related_entities = self.graph_store.find_related_entities(
                        entity_id=entity.entity_id,
                        max_depth=2,
                        limit=20,
                    )
                    for rel_entity in related_entities:
                        if "memory_id" in rel_entity:
                            related_memory_ids.add(rel_entity["memory_id"])

                    entity_rels = self.graph_store.get_entity_relationships(entity.entity_id)
                    for rel in entity_rels:
                        rel_data = rel.get("relationship", {})
                        if "memory_id" in rel_data:
                            related_memory_ids.add(rel_data["memory_id"])
                except Exception:
                    continue

            # 构建结果
            results = []
            for memory_id in list(related_memory_ids)[:limit * 2]:
                mem = self._find_memory_by_id(memory_id)
                if not mem:
                    continue
                if user_id and mem.user_id != user_id:
                    continue

                graph_score = self._calculate_graph_relevance(mem, query_entities)
                results.append(
                    {
                        "id": memory_id,
                        "memory_id": memory_id,
                        "content": mem.content,
                        "similarity": graph_score,
                        "user_id": mem.user_id,
                        "memory_type": mem.memory_type,
                        "importance": mem.importance,
                        "timestamp": int(mem.timestamp.timestamp()),
                        "entities": mem.metadata.get("entities", []),
                    }
                )

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
        except Exception as e:
            log.warning("Graph search failed | error={}", e)
            return []

    def _combine_and_rank_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """混合排序结果。"""
        combined: Dict[str, Dict[str, Any]] = {}
        content_seen: set = set()

        # 添加向量结果
        for result in vector_results:
            memory_id = result.get("memory_id", result.get("id"))
            content = result.get("content", "")
            content_hash = hash(content.strip())

            if content_hash in content_seen:
                continue
            content_seen.add(content_hash)

            combined[memory_id] = {
                **result,
                "memory_id": memory_id,
                "vector_score": result.get("score", 0.0),
                "graph_score": 0.0,
            }

        # 添加图结果
        for result in graph_results:
            memory_id = result.get("memory_id", result.get("id"))
            content = result.get("content", "")
            content_hash = hash(content.strip())

            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            elif content_hash not in content_seen:
                content_seen.add(content_hash)
                combined[memory_id] = {
                    **result,
                    "memory_id": memory_id,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0),
                }

        # 计算混合分数
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)

            # 基础相似度：向量 0.7 + 图 0.3
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            # 重要性加权因子 [0.8, 1.2]
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight

            result["combined_score"] = combined_score

        # 过滤并排序
        min_threshold = 0.1
        filtered_results = [r for r in combined.values() if r["combined_score"] >= min_threshold]
        sorted_results = sorted(filtered_results, key=lambda x: x["combined_score"], reverse=True)

        return sorted_results[:limit]

    def _extract_entities(self, text: str) -> List[Entity]:
        """提取实体（简化版本，使用关键词提取）。"""
        entities = []

        # 简单分词提取（按空格和标点）
        words = []
        current_word = []
        for char in text:
            if char.isalnum() or char in "一二三四五六七八九十百千万亿":
                current_word.append(char)
            else:
                if current_word:
                    word = "".join(current_word)
                    if len(word) >= 2:  # 只保留长度 >= 2 的词
                        words.append(word)
                    current_word = []
        if current_word:
            word = "".join(current_word)
            if len(word) >= 2:
                words.append(word)

        # 创建实体（去重）
        seen = set()
        for word in words[:20]:  # 限制数量
            if word.lower() in seen:
                continue
            seen.add(word.lower())

            entity = Entity(
                entity_id=f"entity_{hash(word)}",
                name=word,
                entity_type="CONCEPT",
                description=f"从文本中提取的概念: {word}",
            )
            entities.append(entity)

            # 更新缓存
            if entity.entity_id in self.entities:
                self.entities[entity.entity_id].frequency += 1
            else:
                self.entities[entity.entity_id] = entity

        return entities

    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """提取关系（简化版本，使用共现关系）。"""
        relations = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                relation = Relation(
                    from_entity=entity1.entity_id,
                    to_entity=entity2.entity_id,
                    relation_type="CO_OCCURS",
                    strength=0.5,
                    evidence=text[:100],
                )
                relations.append(relation)
                self.relations.append(relation)

        return relations

    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem) -> bool:
        """添加实体到 Neo4j。"""
        if not self.graph_store:
            return False
        try:
            return self.graph_store.add_entity(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                properties={
                    "description": entity.description,
                    "memory_id": memory_item.id,
                    "user_id": memory_item.user_id,
                    **entity.properties,
                },
            )
        except Exception as e:
            log.warning("Add entity to graph failed | error={}", e)
            return False

    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem) -> bool:
        """添加关系到 Neo4j。"""
        if not self.graph_store:
            return False
        try:
            return self.graph_store.add_relationship(
                from_entity_id=relation.from_entity,
                to_entity_id=relation.to_entity,
                relationship_type=relation.relation_type,
                properties={
                    "strength": relation.strength,
                    "evidence": relation.evidence,
                    "memory_id": memory_item.id,
                },
            )
        except Exception as e:
            log.warning("Add relation to graph failed | error={}", e)
            return False

    def _calculate_graph_relevance(self, memory: MemoryItem, query_entities: List[Entity]) -> float:
        """计算图相关性分数。"""
        memory_entities = memory.metadata.get("entities", [])
        if not memory_entities or not query_entities:
            return 0.0

        query_entity_ids = {e.entity_id for e in query_entities}
        matching_entities = len(set(memory_entities).intersection(query_entity_ids))
        entity_score = matching_entities / len(query_entity_ids) if query_entity_ids else 0

        return min(entity_score, 1.0)

    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        """根据 ID 查找记忆。"""
        for memory in self.semantic_memories:
            if memory.id == memory_id:
                return memory
        return None


__all__ = ["Entity", "Relation", "SemanticMemory"]
