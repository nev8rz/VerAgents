#!/usr/bin/env python
"""语义记忆（SemanticMemory）完整使用示例。

演示功能：
1. 初始化语义记忆
2. 添加多个语义记忆（含实体和关系）
3. 语义检索（向量 + 图混合）
4. 实体搜索
5. 相关实体查询
6. 更新与删除
7. 遗忘机制
8. 统计信息与知识图谱导出

运行前请确保设置好环境变量：
- QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION
- EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL_NAME
- NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD（可选，用于图功能）
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from loguru import logger

from veragents.memory.base import MemoryConfig, MemoryItem
from veragents.memory.types.semantic import SemanticMemory

load_dotenv()


def generate_id() -> str:
    """生成唯一 ID。"""
    return str(uuid.uuid4())


def main():
    logger.info("=== 语义记忆（SemanticMemory）使用示例 ===\n")

    # ------------------------------------------------------------------ #
    # 1. 初始化语义记忆
    # ------------------------------------------------------------------ #
    logger.info("1. 初始化语义记忆...")
    config = MemoryConfig(
        storage_path="./memory_data",
        max_capacity=100,
        importance_threshold=0.1,
    )
    semantic_memory = SemanticMemory(config=config)
    logger.info("初始化完成\n")

    # ------------------------------------------------------------------ #
    # 2. 添加语义记忆
    # ------------------------------------------------------------------ #
    logger.info("2. 添加语义记忆...")

    mem1 = MemoryItem(
        id=generate_id(),
        content="Python 是一种广泛使用的高级编程语言，以其简洁的语法和强大的库生态系统著称。Python 支持多种编程范式，包括面向对象、函数式和过程式编程。",
        memory_type="semantic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=3),
        importance=0.9,
        metadata={
            "context": {"domain": "programming", "topic": "Python"},
            "tags": ["Python", "编程语言", "技术"],
        },
    )
    semantic_memory.add(mem1)
    logger.info(f"添加知识 1: {mem1.id[:8]}... (Python 介绍)")

    mem2 = MemoryItem(
        id=generate_id(),
        content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测。常见的机器学习算法包括线性回归、决策树、随机森林和神经网络。",
        memory_type="semantic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=2),
        importance=0.85,
        metadata={
            "context": {"domain": "AI", "topic": "机器学习"},
            "tags": ["机器学习", "AI", "算法"],
        },
    )
    semantic_memory.add(mem2)
    logger.info(f"添加知识 2: {mem2.id[:8]}... (机器学习)")

    mem3 = MemoryItem(
        id=generate_id(),
        content="TensorFlow 是 Google 开发的开源深度学习框架，广泛用于构建和训练神经网络模型。它提供了灵活的张量操作和自动微分功能。",
        memory_type="semantic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=1),
        importance=0.8,
        metadata={
            "context": {"domain": "AI", "topic": "深度学习框架"},
            "tags": ["TensorFlow", "深度学习", "Google"],
        },
    )
    semantic_memory.add(mem3)
    logger.info(f"添加知识 3: {mem3.id[:8]}... (TensorFlow)")

    mem4 = MemoryItem(
        id=generate_id(),
        content="Docker 是一种容器化技术，允许开发者将应用程序及其依赖打包到一个可移植的容器中。Docker 简化了部署流程，确保应用在不同环境中的一致性。",
        memory_type="semantic",
        user_id="user_002",
        timestamp=datetime.now() - timedelta(minutes=30),
        importance=0.75,
        metadata={
            "context": {"domain": "DevOps", "topic": "容器化"},
            "tags": ["Docker", "容器", "部署"],
        },
    )
    semantic_memory.add(mem4)
    logger.info(f"添加知识 4: {mem4.id[:8]}... (Docker, user_002)")

    # 添加一个低重要性的记忆用于测试遗忘
    mem5 = MemoryItem(
        id=generate_id(),
        content="这是一个临时的测试记录，重要性很低。",
        memory_type="semantic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(days=5),
        importance=0.05,
        metadata={"context": {"type": "test"}, "tags": ["测试"]},
    )
    semantic_memory.add(mem5)
    logger.info(f"添加低重要性知识: {mem5.id[:8]}... (用于遗忘测试)")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 3. 语义检索
    # ------------------------------------------------------------------ #
    logger.info("3. 语义检索...")

    query = "Python 编程"
    results = semantic_memory.retrieve(query, limit=3, user_id="user_001")
    logger.info(f"查询: '{query}'")
    logger.info(f"找到 {len(results)} 条相关知识:")
    for i, mem in enumerate(results, 1):
        score = mem.metadata.get("combined_score", 0)
        prob = mem.metadata.get("probability", 0)
        logger.info(f"  {i}. [分数: {score:.3f}, 概率: {prob:.3f}] {mem.content[:50]}...")

    logger.info("")

    query2 = "深度学习框架"
    results2 = semantic_memory.retrieve(query2, limit=3)
    logger.info(f"查询: '{query2}'")
    logger.info(f"找到 {len(results2)} 条相关知识:")
    for i, mem in enumerate(results2, 1):
        score = mem.metadata.get("combined_score", 0)
        logger.info(f"  {i}. [{score:.3f}] {mem.content[:50]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 4. 实体搜索
    # ------------------------------------------------------------------ #
    logger.info("4. 实体搜索...")

    entities = semantic_memory.search_entities("Python", limit=5)
    logger.info(f"搜索 'Python' 相关实体: 找到 {len(entities)} 个")
    for entity in entities:
        logger.info(f"  - {entity.name} (类型: {entity.entity_type}, 频率: {entity.frequency})")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 5. 统计信息
    # ------------------------------------------------------------------ #
    logger.info("5. 统计信息...")

    stats = semantic_memory.get_stats()
    logger.info("语义记忆统计:")
    logger.info(f"  - 记忆数量: {stats['count']}")
    logger.info(f"  - 实体数量: {stats['entities_count']}")
    logger.info(f"  - 关系数量: {stats['relations_count']}")
    logger.info(f"  - 平均重要性: {stats['avg_importance']:.3f}")
    logger.info(f"  - 向量存储: {stats.get('vector_store', {}).get('store_type', 'unknown')}")
    logger.info(f"  - 图节点数: {stats.get('graph_nodes', 0)}")
    logger.info(f"  - 图边数: {stats.get('graph_edges', 0)}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 6. 更新与删除
    # ------------------------------------------------------------------ #
    logger.info("6. 更新与删除...")

    update_success = semantic_memory.update(
        memory_id=mem1.id,
        importance=0.95,
        metadata={"context": {"domain": "programming", "topic": "Python", "level": "advanced"}},
    )
    logger.info(f"更新知识 {mem1.id[:8]}... 成功: {update_success}")

    exists = semantic_memory.has_memory(mem1.id)
    logger.info(f"知识 {mem1.id[:8]}... 存在: {exists}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 7. 遗忘机制
    # ------------------------------------------------------------------ #
    logger.info("7. 遗忘机制...")

    stats_before = semantic_memory.get_stats()
    logger.info(f"遗忘前记忆数量: {stats_before['count']}")

    forgotten = semantic_memory.forget(strategy="importance_based", threshold=0.1)
    logger.info(f"基于重要性遗忘: 删除了 {forgotten} 条知识")

    stats_after = semantic_memory.get_stats()
    logger.info(f"遗忘后记忆数量: {stats_after['count']}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 8. 知识图谱导出
    # ------------------------------------------------------------------ #
    logger.info("8. 知识图谱导出...")

    kg = semantic_memory.export_knowledge_graph()
    logger.info(f"知识图谱:")
    logger.info(f"  - 缓存实体数: {kg['graph_stats']['cached_entities']}")
    logger.info(f"  - 缓存关系数: {kg['graph_stats']['cached_relations']}")
    logger.info(f"  - 图节点数: {kg['graph_stats']['total_nodes']}")
    logger.info(f"  - 图边数: {kg['graph_stats']['total_relationships']}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 9. 获取所有记忆
    # ------------------------------------------------------------------ #
    logger.info("9. 获取所有记忆...")

    all_memories = semantic_memory.get_all()
    logger.info(f"当前共有 {len(all_memories)} 条语义记忆:")
    for mem in all_memories:
        entities_count = len(mem.metadata.get("entities", []))
        logger.info(
            f"  - [{mem.id[:8]}] {mem.content[:40]}... (重要性: {mem.importance:.2f}, 实体: {entities_count})"
        )

    logger.info("")

    # ------------------------------------------------------------------ #
    # 清理（可选）
    # ------------------------------------------------------------------ #
    # 取消注释以下代码来清空所有语义记忆
    # logger.info("清空所有语义记忆...")
    # semantic_memory.clear()
    # logger.info("清空完成")

    logger.info("=== 示例结束 ===")


if __name__ == "__main__":
    main()
