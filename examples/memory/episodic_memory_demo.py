#!/usr/bin/env python
"""情景记忆（EpisodicMemory）完整使用示例。

演示功能：
1. 初始化情景记忆
2. 添加多个情景记忆
3. 语义检索
4. 会话管理
5. 模式识别
6. 时间线查看
7. 更新与删除
8. 遗忘机制
9. 统计信息

运行前请确保设置好环境变量：
- QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION
- EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL_NAME
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from loguru import logger

from veragents.memory.base import MemoryConfig, MemoryItem
from veragents.memory.types.episodic import EpisodicMemory
from dotenv import load_dotenv

load_dotenv()


def generate_id() -> str:
    """生成唯一 ID。"""
    return str(uuid.uuid4())


def main():
    logger.info("=== 情景记忆（EpisodicMemory）使用示例 ===\n")

    # ------------------------------------------------------------------ #
    # 1. 初始化情景记忆
    # ------------------------------------------------------------------ #
    logger.info("1. 初始化情景记忆...")
    config = MemoryConfig(
        storage_path="./memory_data",
        max_capacity=100,
        importance_threshold=0.1,
    )
    episodic_memory = EpisodicMemory(config=config)
    logger.info("初始化完成\n")

    # ------------------------------------------------------------------ #
    # 2. 添加多个情景记忆
    # ------------------------------------------------------------------ #
    logger.info("2. 添加情景记忆...")

    # 第一个会话的记忆
    session_1 = "session_" + generate_id()[:8]

    mem1 = MemoryItem(
        id=generate_id(),
        content="用户询问了关于 Python 装饰器的使用方法，我详细解释了 @property 和 @classmethod 的区别",
        memory_type="episodic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=2),
        importance=0.8,
        metadata={
            "session_id": session_1,
            "context": {"topic": "Python", "subtopic": "decorators"},
            "outcome": "positive",
            "participants": ["user_001", "assistant"],
            "tags": ["python", "教程", "装饰器"],
        },
    )
    episodic_memory.add(mem1)
    logger.info(f"添加记忆 1: {mem1.id[:8]}... (session: {session_1})")

    mem2 = MemoryItem(
        id=generate_id(),
        content="用户继续询问了如何使用 functools.wraps 保留被装饰函数的元数据",
        memory_type="episodic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=1, minutes=50),
        importance=0.7,
        metadata={
            "session_id": session_1,
            "context": {"topic": "Python", "subtopic": "functools"},
            "outcome": "positive",
            "tags": ["python", "functools"],
        },
    )
    episodic_memory.add(mem2)
    logger.info(f"添加记忆 2: {mem2.id[:8]}...")

    # 第二个会话的记忆
    session_2 = "session_" + generate_id()[:8]

    mem3 = MemoryItem(
        id=generate_id(),
        content="用户请求帮助调试一个 async/await 相关的死锁问题",
        memory_type="episodic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=30),
        importance=0.9,
        metadata={
            "session_id": session_2,
            "context": {"topic": "Python", "subtopic": "async"},
            "outcome": "resolved",
            "tags": ["python", "async", "调试"],
        },
    )
    episodic_memory.add(mem3)
    logger.info(f"添加记忆 3: {mem3.id[:8]}... (new session: {session_2})")

    mem4 = MemoryItem(
        id=generate_id(),
        content="用户询问了 JavaScript 中的 Promise 和 async/await 的区别",
        memory_type="episodic",
        user_id="user_002",
        timestamp=datetime.now() - timedelta(minutes=15),
        importance=0.6,
        metadata={
            "session_id": "session_js",
            "context": {"topic": "JavaScript", "subtopic": "async"},
            "outcome": "positive",
            "tags": ["javascript", "async", "promise"],
        },
    )
    episodic_memory.add(mem4)
    logger.info(f"添加记忆 4: {mem4.id[:8]}... (user_002)")

    # 添加一个低重要性的记忆用于测试遗忘
    mem5 = MemoryItem(
        id=generate_id(),
        content="用户发送了一个简单的问候消息",
        memory_type="episodic",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(days=5),
        importance=0.05,  # 很低的重要性
        metadata={
            "session_id": "session_old",
            "context": {"type": "greeting"},
            "tags": ["greeting"],
        },
    )
    episodic_memory.add(mem5)
    logger.info(f"添加记忆 5: {mem5.id[:8]}... (低重要性，用于遗忘测试)")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 3. 语义检索
    # ------------------------------------------------------------------ #
    logger.info("3. 语义检索...")

    query = "Python 装饰器怎么用"
    results = episodic_memory.retrieve(query, limit=3, user_id="user_001")
    logger.info(f"查询: '{query}'")
    logger.info(f"找到 {len(results)} 条相关记忆:")
    for i, mem in enumerate(results, 1):
        score = mem.metadata.get("relevance_score", 0)
        logger.info(f"  {i}. [{score:.3f}] {mem.content[:50]}...")

    logger.info("")

    query_async = "异步编程调试"
    results_async = episodic_memory.retrieve(query_async, limit=3)
    logger.info(f"查询: '{query_async}'")
    logger.info(f"找到 {len(results_async)} 条相关记忆:")
    for i, mem in enumerate(results_async, 1):
        score = mem.metadata.get("relevance_score", 0)
        logger.info(f"  {i}. [{score:.3f}] {mem.content[:50]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 4. 会话管理
    # ------------------------------------------------------------------ #
    logger.info("4. 会话管理...")

    session_eps = episodic_memory.get_session_episodes(session_1)
    logger.info(f"会话 {session_1} 包含 {len(session_eps)} 条记忆:")
    for ep in session_eps:
        logger.info(f"  - {ep.content[:40]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 5. 模式识别
    # ------------------------------------------------------------------ #
    logger.info("5. 模式识别...")

    patterns = episodic_memory.find_patterns(user_id="user_001", min_frequency=2)
    logger.info(f"发现 {len(patterns)} 个模式 (user_001):")
    for pattern in patterns[:5]:  # 显示前5个
        logger.info(
            f"  - [{pattern['type']}] {pattern['pattern']} "
            f"(出现 {pattern['frequency']} 次, 置信度 {pattern['confidence']:.2f})"
        )

    logger.info("")

    # ------------------------------------------------------------------ #
    # 6. 时间线查看
    # ------------------------------------------------------------------ #
    logger.info("6. 时间线查看...")

    timeline = episodic_memory.get_timeline(user_id="user_001", limit=5)
    logger.info(f"用户 user_001 的最近 {len(timeline)} 条记忆:")
    for entry in timeline:
        logger.info(
            f"  [{entry['timestamp']}] {entry['content'][:40]}... "
            f"(重要性: {entry['importance']:.2f})"
        )

    logger.info("")

    # ------------------------------------------------------------------ #
    # 7. 更新与删除
    # ------------------------------------------------------------------ #
    logger.info("7. 更新与删除...")

    # 更新记忆
    update_success = episodic_memory.update(
        memory_id=mem1.id,
        importance=0.95,
        metadata={"outcome": "very_positive"},
    )
    logger.info(f"更新记忆 {mem1.id[:8]}... 成功: {update_success}")

    # 检查记忆是否存在
    exists = episodic_memory.has_memory(mem1.id)
    logger.info(f"记忆 {mem1.id[:8]}... 存在: {exists}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 8. 遗忘机制
    # ------------------------------------------------------------------ #
    logger.info("8. 遗忘机制...")

    stats_before = episodic_memory.get_stats()
    logger.info(f"遗忘前记忆数量: {stats_before['count']}")

    # 基于重要性遗忘（阈值 0.1，会删除 mem5）
    forgotten = episodic_memory.forget(strategy="importance_based", threshold=0.1)
    logger.info(f"基于重要性遗忘: 删除了 {forgotten} 条记忆")

    stats_after = episodic_memory.get_stats()
    logger.info(f"遗忘后记忆数量: {stats_after['count']}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 9. 统计信息
    # ------------------------------------------------------------------ #
    logger.info("9. 统计信息...")

    stats = episodic_memory.get_stats()
    logger.info("情景记忆统计:")
    logger.info(f"  - 记忆数量: {stats['count']}")
    logger.info(f"  - 会话数量: {stats['sessions_count']}")
    logger.info(f"  - 平均重要性: {stats['avg_importance']:.3f}")
    logger.info(f"  - 时间跨度: {stats['time_span_days']} 天")
    logger.info(f"  - 向量存储: {stats.get('vector_store', {}).get('store_type', 'unknown')}")
    logger.info(f"  - 文档存储: {stats.get('document_store', {}).get('store_type', 'unknown')}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 10. 获取所有记忆
    # ------------------------------------------------------------------ #
    logger.info("10. 获取所有记忆...")

    all_memories = episodic_memory.get_all()
    logger.info(f"当前共有 {len(all_memories)} 条情景记忆:")
    for mem in all_memories:
        logger.info(f"  - [{mem.id[:8]}] {mem.content[:40]}... (重要性: {mem.importance:.2f})")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 清理（可选）
    # ------------------------------------------------------------------ #
    # 取消注释以下代码来清空所有情景记忆
    # logger.info("清空所有情景记忆...")
    # episodic_memory.clear()
    # logger.info("清空完成")

    logger.info("=== 示例结束 ===")


if __name__ == "__main__":
    main()
