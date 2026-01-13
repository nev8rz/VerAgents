"""Document store demo (SQLite).

Usage:
  PYTHONPATH=. python examples/memory/document_store_demo.py
"""

from __future__ import annotations

from veragents.memory.storage.document_store import SQLiteDocumentStore


def main() -> None:
    store = SQLiteDocumentStore("./memory.db")

    # Add a memory
    mem_id = store.add_memory(
        memory_id="demo-memory-1",
        user_id="demo-user",
        content="记忆示例：今天学习了 Transformer 的注意力机制。",
        memory_type="working",
        timestamp=1700000000,
        importance=0.7,
        properties={"topic": "NLP"},
    )
    print(f"添加记忆: {mem_id}")

    # Fetch memory
    fetched = store.get_memory(mem_id)
    print("读取记忆:", fetched)

    # Add a document
    doc_id = store.add_document("这是一份示例文档", metadata={"user_id": "demo-user", "category": "demo"})
    print(f"添加文档: {doc_id}")

    # Search
    results = store.search_memories(user_id="demo-user", memory_type="working", limit=5)
    print(f"搜索结果数量: {len(results)}")

    # Stats
    print("数据库统计:", store.get_database_stats())

    store.close()


if __name__ == "__main__":
    main()
