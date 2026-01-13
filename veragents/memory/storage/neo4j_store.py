"""Neo4j 图存储实现。"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger as log

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None  # type: ignore
    AuthError = ServiceUnavailable = Exception  # type: ignore
    NEO4J_AVAILABLE = False


class Neo4jStore:
    """负责知识图谱相关的读写。"""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package not installed. Run: pip install neo4j>=5.0.0")

        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.driver = None
        self._initialize_driver(
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout,
        )
        self._create_indexes()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _initialize_driver(self, **config: Any) -> None:
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password), **config)
            self.driver.verify_connectivity()
            log.info("Neo4j connected | uri={} user={}", self.uri, self.user)
        except AuthError as exc:
            log.error("Neo4j authentication failed: {}", exc)
            raise
        except ServiceUnavailable as exc:
            log.error("Neo4j unavailable: {}", exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Neo4j connection failed: {}", exc)
            raise

    def _create_indexes(self) -> None:
        indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
        ]
        with self.driver.session(database=self.database) as session:
            for q in indexes:
                try:
                    session.run(q)
                except Exception:
                    log.debug("Index creation skipped (maybe exists)")
        log.info("Neo4j indexes ensured")

    # ------------------------------------------------------------------ #
    # Entity / Relations
    # ------------------------------------------------------------------ #
    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        props = properties.copy() if properties else {}
        now = datetime.utcnow().isoformat()
        props.update({"id": entity_id, "name": name, "type": entity_type, "created_at": now, "updated_at": now})
        query = """
        MERGE (e:Entity {id: $entity_id})
        SET e += $properties
        RETURN e
        """
        try:
            with self.driver.session(database=self.database) as session:
                record = session.run(query, entity_id=entity_id, properties=props).single()
                ok = record is not None
                log.info("Neo4j add_entity | id={} type={} ok={}", entity_id, entity_type, ok)
                return ok
        except Exception as exc:
            log.error("Neo4j add_entity failed | id={} err={}", entity_id, exc)
            return False

    def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        props = properties.copy() if properties else {}
        now = datetime.utcnow().isoformat()
        props.update({"type": relationship_type, "created_at": now, "updated_at": now})
        query = f"""
        MATCH (from:Entity {{id: $from_id}})
        MATCH (to:Entity {{id: $to_id}})
        MERGE (from)-[r:{relationship_type}]->(to)
        SET r += $properties
        RETURN r
        """
        try:
            with self.driver.session(database=self.database) as session:
                record = session.run(query, from_id=from_entity_id, to_id=to_entity_id, properties=props).single()
                ok = record is not None
                log.info("Neo4j add_relationship | {} -{}-> {} ok={}", from_entity_id, relationship_type, to_entity_id, ok)
                return ok
        except Exception as exc:
            log.error("Neo4j add_relationship failed | {} -{}-> {} err={}", from_entity_id, relationship_type, to_entity_id, exc)
            return False

    def delete_entity(self, entity_id: str) -> bool:
        query = "MATCH (e:Entity {id: $entity_id}) DETACH DELETE e"
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id).consume()
                deleted = result.counters.nodes_deleted
                log.info("Neo4j delete_entity | id={} deleted={}", entity_id, deleted)
                return deleted > 0
        except Exception as exc:
            log.error("Neo4j delete_entity failed | id={} err={}", entity_id, exc)
            return False

    def clear_all(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                summary = session.run("MATCH (n) DETACH DELETE n").consume()
                log.info(
                    "Neo4j cleared | nodes_deleted={} rels_deleted={}",
                    summary.counters.nodes_deleted,
                    summary.counters.relationships_deleted,
                )
                return True
        except Exception as exc:
            log.error("Neo4j clear_all failed: {}", exc)
            return False

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        rel_filter = ""
        params: Dict[str, Any] = {"entity_id": entity_id, "limit": limit}
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)
        query = f"""
        MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
        WHERE start.id <> related.id
        RETURN DISTINCT related,
               length(path) as distance,
               [rel in relationships(path) | type(rel)] as relationship_path
        ORDER BY distance, related.name
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                entities: List[Dict[str, Any]] = []
                for record in result:
                    entity_data = dict(record["related"])
                    entity_data["distance"] = record["distance"]
                    entity_data["relationship_path"] = record["relationship_path"]
                    entities.append(entity_data)
                return entities
        except Exception as exc:
            log.error("Neo4j find_related_entities failed | id={} err={}", entity_id, exc)
            return []

    def search_entities_by_name(self, name_pattern: str, entity_types: Optional[List[str]] = None, limit: int = 20) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"pattern": f".*{name_pattern}.*", "limit": limit}
        type_filter = ""
        if entity_types:
            type_filter = "AND e.type IN $types"
            params["types"] = entity_types
        query = f"""
        MATCH (e:Entity)
        WHERE e.name =~ $pattern {type_filter}
        RETURN e
        ORDER BY e.name
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.database) as session:
                rows = session.run(query, **params)
                return [dict(record["e"]) for record in rows]
        except Exception as exc:
            log.error("Neo4j search_entities_by_name failed | pattern={} err={}", name_pattern, exc)
            return []

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
        RETURN r, other,
               CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
        """
        try:
            with self.driver.session(database=self.database) as session:
                rows = session.run(query, entity_id=entity_id)
                relationships: List[Dict[str, Any]] = []
                for record in rows:
                    rel_data = dict(record["r"])
                    other_data = dict(record["other"])
                    relationships.append({"relationship": rel_data, "other_entity": other_data, "direction": record["direction"]})
                return relationships
        except Exception as exc:
            log.error("Neo4j get_entity_relationships failed | id={} err={}", entity_id, exc)
            return []

    def get_stats(self) -> Dict[str, Any]:
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
            "memory_nodes": "MATCH (n:Memory) RETURN count(n) as count",
        }
        stats: Dict[str, Any] = {}
        try:
            with self.driver.session(database=self.database) as session:
                for key, cypher in queries.items():
                    record = session.run(cypher).single()
                    stats[key] = record["count"] if record else 0
            return stats
        except Exception as exc:
            log.error("Neo4j get_stats failed: {}", exc)
            return stats

    def health_check(self) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                record = session.run("RETURN 1 as health").single()
                return bool(record and record["health"] == 1)
        except Exception as exc:
            log.error("Neo4j health_check failed: {}", exc)
            return False

    def close(self) -> None:
        driver = getattr(self, "driver", None)
        if driver:
            try:
                driver.close()
                log.info("Neo4j connection closed")
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["Neo4jStore"]
