"""Qdrant 向量数据库存储实现。"""

from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger as log

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore
    models = None  # type: ignore
    Distance = VectorParams = Filter = FieldCondition = MatchValue = PointStruct = object  # type: ignore
    QDRANT_AVAILABLE = False

from veragents.memory.base import MemoryItem


class QdrantStore:
    """负责与 Qdrant 交互的向量存储层。"""

    _instances: Dict[tuple[str, str], "QdrantStore"] = {}
    _lock = threading.Lock()

    def __new__(cls, url: Optional[str] = None, collection: str = "memory", api_key: Optional[str] = None, **kwargs):
        key = (url or "local", collection)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        url: Optional[str] = None,
        collection: str = "memory",
        api_key: Optional[str] = None,
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
    ):
        if hasattr(self, "_initialized"):
            return
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client 未安装，请运行: pip install qdrant-client>=1.6.0")

        self.url = url or os.getenv("QDRANT_URL")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "memory")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", str(vector_size)))
        self.timeout = timeout

        distance_map = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclidean": Distance.EUCLID}
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)

        self.hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
        self.hnsw_ef_construct = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
        self.search_ef = int(os.getenv("QDRANT_SEARCH_EF", "128"))
        self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"

        self.client: Optional[QdrantClient] = None
        self._initialize_client()
        self._initialized = True

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _initialize_client(self) -> None:
        if self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout)
            log.info("Qdrant connected | url={} collection={}", self.url, self.collection)
        else:
            self.client = QdrantClient(host="localhost", port=6333, timeout=self.timeout)
            log.info("Qdrant connected | local collection={}", self.collection)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections  # type: ignore
            names = [c.name for c in collections]
            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
                    hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
                )
                log.info("Qdrant collection created | name={}", self.collection)
            else:
                log.info("Qdrant collection ready | name={}", self.collection)
            self._ensure_payload_indexes()
        except Exception as exc:
            log.error("Qdrant collection init failed: {}", exc)
            raise

    def _ensure_payload_indexes(self) -> None:
        index_fields = [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.INTEGER),
        ]
        for field_name, schema_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                log.debug("Qdrant payload index exists or failed | field={}", field_name)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def upsert(self, items: List[MemoryItem]) -> bool:
        """兼容旧接口：接受 MemoryItem，要求 metadata 中提供 embedding。"""
        vectors: List[List[float]] = []
        meta_list: List[Dict[str, Any]] = []
        ids: List[str] = []
        for itm in items:
            emb = None
            if isinstance(itm.metadata, dict):
                emb = itm.metadata.get("embedding")
            if emb is None:
                log.warning("Skip item without embedding | id={}", itm.id)
                continue
            vectors.append(list(emb))
            meta = dict(itm.metadata or {})
            meta.update(
                {
                    "memory_id": itm.id,
                    "memory_type": itm.memory_type,
                    "user_id": getattr(itm, "user_id", None),
                    "timestamp": int(itm.timestamp.timestamp()) if getattr(itm, "timestamp", None) else int(datetime.utcnow().timestamp()),
                }
            )
            meta_list.append(meta)
            ids.append(itm.id)
        return self.add_vectors(vectors, meta_list, ids=ids)

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> bool:
        if not vectors:
            log.warning("Qdrant add_vectors called with empty vectors")
            return False
        if len(vectors) != len(metadata):
            log.error("Qdrant add_vectors length mismatch | vectors={} metadata={}", len(vectors), len(metadata))
            return False

        point_ids = ids or [str(uuid.uuid4()) for _ in vectors]
        points = []
        for vec, meta, pid in zip(vectors, metadata, point_ids):
            if len(vec) != self.vector_size:
                log.warning("Qdrant vector dim mismatch | expected={} got={}", self.vector_size, len(vec))
                continue
            payload = dict(meta)
            payload.setdefault("timestamp", int(datetime.utcnow().timestamp()))
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        if not points:
            log.warning("Qdrant add_vectors produced no valid points")
            return False

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        log.info("Qdrant upsert done | points={}", len(points))
        return True

    def search(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.search_similar(query_embedding, limit=top_k, score_threshold=score_threshold, where=where)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if len(query_vector) != self.vector_size:
            log.error("Qdrant search dim mismatch | expected={} got={}", self.vector_size, len(query_vector))
            return []

        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        try:
            try:
                resp = self.client.query_points(  # type: ignore[attr-defined]
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=models.SearchParams(hnsw_ef=self.search_ef, exact=self.search_exact),
                )
                hits = resp.points
            except AttributeError:
                hits = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=models.SearchParams(hnsw_ef=self.search_ef, exact=self.search_exact),
                )
            results = [{"id": hit.id, "score": hit.score, "metadata": hit.payload or {}} for hit in hits]
            log.debug("Qdrant search returned {} results", len(results))
            return results
        except Exception as exc:
            log.error("Qdrant search failed: {}", exc)
            return []

    def delete_vectors(self, ids: List[str]) -> bool:
        if not ids:
            return True
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            log.info("Qdrant deleted vectors | count={}", len(ids))
            return True
        except Exception as exc:
            log.error("Qdrant delete_vectors failed: {}", exc)
            return False

    def delete_memories(self, memory_ids: List[str]) -> bool:
        if not memory_ids:
            return True
        try:
            conditions = [FieldCondition(key="memory_id", match=MatchValue(value=mid)) for mid in memory_ids]
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=Filter(should=conditions)),
                wait=True,
            )
            log.info("Qdrant delete_memories by payload | count={}", len(memory_ids))
            return True
        except Exception as exc:
            log.error("Qdrant delete_memories failed: {}", exc)
            return False

    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection)
            self._ensure_collection()
            log.info("Qdrant collection cleared | name={}", self.collection)
            return True
        except Exception as exc:
            log.error("Qdrant clear_collection failed: {}", exc)
            return False

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection)
            # qdrant-client cloud may expose different attribute names
            vectors_count = getattr(info, "vectors_count", None) or getattr(info, "points_count", None)
            indexed_vectors_count = getattr(info, "indexed_vectors_count", None) or getattr(info, "points_count", None)
            points_count = getattr(info, "points_count", None) or vectors_count
            segments_count = getattr(info, "segments_count", None)
            return {
                "name": self.collection,
                "vectors_count": vectors_count,
                "indexed_vectors_count": indexed_vectors_count,
                "points_count": points_count,
                "segments_count": segments_count,
                "config": {"vector_size": self.vector_size, "distance": self.distance.value},
            }
        except Exception as exc:
            log.error("Qdrant get_collection_info failed: {}", exc)
            return {"name": self.collection, "store_type": "qdrant"}

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.get_collection_info()
        info["store_type"] = "qdrant"
        return info

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            log.error("Qdrant health_check failed: {}", exc)
            return False

    def __del__(self) -> None:
        client = getattr(self, "client", None)
        if client:
            try:
                client.close()
            except Exception:
                pass


__all__ = ["QdrantStore"]
