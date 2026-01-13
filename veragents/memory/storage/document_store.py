"""文档存储实现，提供 SQLite 后端并预留扩展接口。"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger as log


class DocumentStore(ABC):
    """文档存储基类。"""

    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加记忆。"""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆。"""

    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索记忆。"""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新记忆。"""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆。"""

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息。"""

    @abstractmethod
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加文档（作为 memory 存储）。"""

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档。"""

    @abstractmethod
    def close(self) -> None:
        """关闭资源。"""


class SQLiteDocumentStore(DocumentStore):
    """SQLite 文档存储实现（线程安全，单路径单实例）。"""

    _instances: Dict[str, "SQLiteDocumentStore"] = {}
    _initialized_dbs = set()

    def __new__(cls, db_path: str = "./memory.db"):
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[abs_path] = instance
        return cls._instances[abs_path]

    def __init__(self, db_path: str = "./memory.db"):
        if hasattr(self, "_initialized"):
            return
        self.db_path = db_path
        self.local = threading.local()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            log.info("SQLite 文档存储初始化完成 | db_path={}", db_path)
        self._initialized = True

    # ------------------------------------------------------------------ #
    # DB Helpers
    # ------------------------------------------------------------------ #
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self.local, "connection"):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.local.connection = conn
        return self.local.connection

    def _init_database(self) -> None:
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
            """
        )

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)",
        ]
        for sql in indexes:
            cur.execute(sql)
        conn.commit()
        log.info("SQLite 数据库表和索引创建完成 | db_path={}", self.db_path)

    # ------------------------------------------------------------------ #
    # Memory CRUD
    # ------------------------------------------------------------------ #
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)", (user_id, user_id))

        cur.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                memory_id,
                user_id,
                content,
                memory_type,
                timestamp,
                importance,
                json.dumps(properties) if properties else None,
            ),
        )
        conn.commit()
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "memory_id": row["id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "memory_type": row["memory_type"],
            "timestamp": row["timestamp"],
            "importance": row["importance"],
            "properties": json.loads(row["properties"]) if row["properties"] else {},
            "created_at": row["created_at"],
        }

    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()

        where = []
        params: List[Any] = []
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if memory_type:
            where.append("memory_type = ?")
            params.append(memory_type)
        if start_time:
            where.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where.append("timestamp <= ?")
            params.append(end_time)
        if importance_threshold is not None:
            where.append("importance >= ?")
            params.append(importance_threshold)
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""

        cur.execute(
            f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
            """,
            params + [limit],
        )

        results: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            results.append(
                {
                    "memory_id": row["id"],
                    "user_id": row["user_id"],
                    "content": row["content"],
                    "memory_type": row["memory_type"],
                    "timestamp": row["timestamp"],
                    "importance": row["importance"],
                    "properties": json.loads(row["properties"]) if row["properties"] else {},
                    "created_at": row["created_at"],
                }
            )
        return results

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        conn = self._get_connection()
        cur = conn.cursor()

        updates = []
        params: List[Any] = []
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if properties is not None:
            updates.append("properties = ?")
            params.append(json.dumps(properties))

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)

        cur.execute(
            f"""
            UPDATE memories
            SET {', '.join(updates)}
            WHERE id = ?
            """,
            params,
        )
        conn.commit()
        return cur.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted = cur.rowcount
        conn.commit()
        return deleted > 0

    # ------------------------------------------------------------------ #
    # Docs helpers
    # ------------------------------------------------------------------ #
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        user_id = (metadata or {}).get("user_id", "system")
        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {},
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.get_memory(document_id)

    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cur = conn.cursor()

        stats: Dict[str, Any] = {}
        for table in ["users", "memories", "concepts", "memory_concepts", "concept_relationships"]:
            cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
            stats[f"{table}_count"] = cur.fetchone()["count"]

        cur.execute("SELECT memory_type, COUNT(*) AS count FROM memories GROUP BY memory_type")
        stats["memory_types"] = {row["memory_type"]: row["count"] for row in cur.fetchall()}

        cur.execute(
            """
            SELECT user_id, COUNT(*) AS count
            FROM memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
            """
        )
        stats["top_users"] = {row["user_id"]: row["count"] for row in cur.fetchall()}
        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path
        return stats

    def close(self) -> None:
        if hasattr(self.local, "connection"):
            try:
                self.local.connection.close()
            finally:
                delattr(self.local, "connection")
        log.info("SQLite 连接已关闭 | db_path={}", self.db_path)


__all__ = ["DocumentStore", "SQLiteDocumentStore"]
