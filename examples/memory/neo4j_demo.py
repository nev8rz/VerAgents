"""Neo4j store demo (requires running Neo4j and neo4j Python driver).

Usage:
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_USER=neo4j
  export NEO4J_PASSWORD=your_password
  PYTHONPATH=. python examples/memory/neo4j_demo.py
"""

from __future__ import annotations

import os

from veragents.memory.storage.neo4j_store import Neo4jStore
from dotenv import load_dotenv

load_dotenv()
def main() -> None:
    try:
        store = Neo4jStore(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )
    except ImportError as exc:
        print(f"[demo] neo4j driver not installed: {exc}")
        return
    except Exception as exc:
        print(f"[demo] failed to connect: {exc}")
        return

    store.add_entity("e1", "Transformer", "concept", {"domain": "NLP"})
    store.add_entity("e2", "Attention", "concept", {"domain": "ML"})
    store.add_relationship("e1", "e2", "RELATED_TO", {"weight": 0.9})

    print("实体 e1 关系:", store.get_entity_relationships("e1"))
    print("按名称搜索 'Att':", store.search_entities_by_name("Att"))
    print("相关实体:", store.find_related_entities("e1", max_depth=2))
    print("统计:", store.get_stats())

    store.close()


if __name__ == "__main__":
    main()
