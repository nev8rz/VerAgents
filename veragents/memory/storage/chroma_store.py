"""Chroma 向量存储实现，接口与 QdrantStore 对齐。"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger as log

try:
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    embedding_functions = None  # type: ignore
    CHROMA_AVAILABLE = False


class ChromaStore:
    """Chroma 向量存储层，与 QdrantStore 保持相似方法签名。"""

    def __init__(
        self,
        collection: str = "memory",
        persist_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        auth_token: Optional[str] = None,
        distance: str = "cosine",
        embedding_function: Any = None,
    ):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb 未安装，请运行: pip install chromadb")

        self.collection_name = collection
        self.distance = distance

        client_kwargs: Dict[str, Any] = {}
        if host and port:
            client_kwargs["host"] = host
            client_kwargs["port"] = port
            client_kwargs["ssl"] = host.startswith("https")
            client_kwargs["headers"] = {"Authorization": auth_token} if auth_token else {}
            self.client = chromadb.HttpClient(**client_kwargs)
            log.info("Chroma connected (HTTP) | host={} port={} collection={}", host, port, collection)
        else:
            persist_dir = persist_path or os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
            self.client = chromadb.PersistentClient(path=persist_dir)
            log.info("Chroma connected (persistent) | path={} collection={}", persist_dir, collection)

        self.embedding_function = embedding_function or getattr(embedding_functions, "DefaultEmbeddingFunction", lambda: None)()
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
                embedding_function=self.embedding_function,
            )
        except TypeError:
            # older chromadb without embedding_function arg on get_or_create_collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )
        log.info("Chroma collection ready | name={}", self.collection_name)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> bool:
        if not vectors:
            log.warning("Chroma add_vectors called with empty vectors")
            return False
        if len(vectors) != len(metadata):
            log.error("Chroma add_vectors length mismatch | vectors={} metadata={}", len(vectors), len(metadata))
            return False
        doc_ids = ids or [str(uuid.uuid4()) for _ in vectors]
        try:
            self.collection.add(
                ids=doc_ids,
                embeddings=vectors,
                metadatas=metadata,
            )
            log.info("Chroma upsert done | points={}", len(doc_ids))
            return True
        except Exception as exc:
            log.error("Chroma add_vectors failed: {}", exc)
            return False

    def upsert(self, items: List[Any]) -> bool:
        """兼容接口：items 需包含 embedding 和 metadata。"""
        vectors: List[List[float]] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for itm in items:
            emb = getattr(itm, "embedding", None) or (itm.metadata or {}).get("embedding") if hasattr(itm, "metadata") else None
            if emb is None:
                log.warning("Chroma skip item without embedding | item={}", getattr(itm, "id", None))
                continue
            vectors.append(list(emb))
            meta = dict(getattr(itm, "metadata", {}) or {})
            metas.append(meta)
            ids.append(getattr(itm, "id", str(uuid.uuid4())))
        return self.add_vectors(vectors, metas, ids=ids)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            res = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where,
            )
            ids = res.get("ids", [[]])[0]
            scores = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            results: List[Dict[str, Any]] = []
            for i, (pid, score, meta) in enumerate(zip(ids, scores, metas)):
                if score_threshold is not None and score > score_threshold:
                    continue
                results.append({"id": pid, "score": score, "metadata": meta or {}})
            return results
        except Exception as exc:
            log.error("Chroma search failed: {}", exc)
            return []

    def search(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.search_similar(query_embedding, limit=top_k, score_threshold=score_threshold, where=where)

    def delete_vectors(self, ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            log.info("Chroma deleted vectors | count={}", len(ids))
            return True
        except Exception as exc:
            log.error("Chroma delete_vectors failed: {}", exc)
            return False

    def delete_memories(self, memory_ids: List[str]) -> bool:
        try:
            self.collection.delete(where={"memory_id": {"$in": memory_ids}})
            log.info("Chroma delete_memories by payload | count={}", len(memory_ids))
            return True
        except Exception as exc:
            log.error("Chroma delete_memories failed: {}", exc)
            return False

    def clear_collection(self) -> bool:
        try:
            self.collection.delete(where={})
            log.info("Chroma collection cleared | name={}", self.collection_name)
            return True
        except Exception as exc:
            log.error("Chroma clear_collection failed: {}", exc)
            return False

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {"name": self.collection_name, "points_count": count, "store_type": "chroma"}
        except Exception as exc:
            log.error("Chroma get_collection_info failed: {}", exc)
            return {"name": self.collection_name, "store_type": "chroma"}

    def get_collection_stats(self) -> Dict[str, Any]:
        return self.get_collection_info()

    def health_check(self) -> bool:
        try:
            _ = self.collection.count()
            return True
        except Exception as exc:
            log.error("Chroma health_check failed: {}", exc)
            return False

    def __del__(self) -> None:
        try:
            client = getattr(self, "client", None)
            if client:
                client.close()
        except Exception:
            pass


__all__ = ["ChromaStore"]
