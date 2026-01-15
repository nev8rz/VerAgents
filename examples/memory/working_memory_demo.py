#!/usr/bin/env python
"""工作记忆（WorkingMemory）完整使用示例。

演示功能：
1. 初始化工作记忆
2. 添加多个工作记忆
3. 关键词检索
4. 获取最近/重要记忆
5. 容量限制测试
6. 更新与删除
7. 遗忘机制
8. 上下文摘要
9. 统计信息


工作记忆特点：
- 纯内存存储（无持久化）
- 容量有限（默认 20 条）
- TTL 过期机制
- 优先级管理
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from loguru import logger

from veragents.memory.base import MemoryConfig, MemoryItem
from veragents.memory.types.working import WorkingMemory

load_dotenv()


def generate_id() -> str:
    """生成唯一 ID。"""
    return str(uuid.uuid4())


def main():
    logger.info("=== 工作记忆（WorkingMemory）使用示例 ===\n")

    # ------------------------------------------------------------------ #
    # 1. 初始化工作记忆
    # ------------------------------------------------------------------ #
    logger.info("1. 初始化工作记忆...")
    config = MemoryConfig(
        working_memory_capacity=10,  # 最多 10 条记忆
        working_memory_tokens=500,   # 最多 500 个 token
        working_memory_ttl_minutes=60,  # 60 分钟过期
        decay_factor=0.95,
    )
    working_memory = WorkingMemory(config=config)
    logger.info("初始化完成\n")

    # ------------------------------------------------------------------ #
    # 2. 添加工作记忆
    # ------------------------------------------------------------------ #
    logger.info("2. 添加工作记忆...")

    mem1 = MemoryItem(
        id=generate_id(),
        content="用户询问了关于 Python 装饰器的使用方法",
        memory_type="working",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=10),
        importance=0.8,
        metadata={"topic": "Python", "type": "question"},
    )
    working_memory.add(mem1)
    logger.info(f"添加记忆 1: {mem1.id[:8]}...")

    mem2 = MemoryItem(
        id=generate_id(),
        content="我向用户解释了 @property 装饰器的用法",
        memory_type="working",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=8),
        importance=0.7,
        metadata={"topic": "Python", "type": "answer"},
    )
    working_memory.add(mem2)
    logger.info(f"添加记忆 2: {mem2.id[:8]}...")

    mem3 = MemoryItem(
        id=generate_id(),
        content="用户请求帮助调试一个机器学习模型",
        memory_type="working",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=5),
        importance=0.9,
        metadata={"topic": "ML", "type": "request"},
    )
    working_memory.add(mem3)
    logger.info(f"添加记忆 3: {mem3.id[:8]}...")

    mem4 = MemoryItem(
        id=generate_id(),
        content="用户提到他们正在使用 TensorFlow 框架",
        memory_type="working",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=3),
        importance=0.6,
        metadata={"topic": "ML", "type": "context"},
    )
    working_memory.add(mem4)
    logger.info(f"添加记忆 4: {mem4.id[:8]}...")

    # 添加一个低重要性的记忆用于测试遗忘
    mem5 = MemoryItem(
        id=generate_id(),
        content="临时测试消息",
        memory_type="working",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=1),
        importance=0.05,
        metadata={"type": "test"},
    )
    working_memory.add(mem5)
    logger.info(f"添加低重要性记忆: {mem5.id[:8]}... (用于遗忘测试)")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 3. 关键词检索
    # ------------------------------------------------------------------ #
    logger.info("3. 关键词检索...")

    query = "Python 装饰器"
    results = working_memory.retrieve(query, limit=3)
    logger.info(f"查询: '{query}'")
    logger.info(f"找到 {len(results)} 条相关记忆:")
    for i, mem in enumerate(results, 1):
        logger.info(f"  {i}. {mem.content[:50]}...")

    logger.info("")

    query2 = "机器学习"
    results2 = working_memory.retrieve(query2, limit=3)
    logger.info(f"查询: '{query2}'")
    logger.info(f"找到 {len(results2)} 条相关记忆:")
    for i, mem in enumerate(results2, 1):
        logger.info(f"  {i}. {mem.content[:50]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 4. 获取最近/重要记忆
    # ------------------------------------------------------------------ #
    logger.info("4. 获取最近/重要记忆...")

    recent = working_memory.get_recent(limit=3)
    logger.info(f"最近 3 条记忆:")
    for mem in recent:
        age = (datetime.now() - mem.timestamp).total_seconds() / 60
        logger.info(f"  - [{age:.1f}分钟前] {mem.content[:40]}...")

    important = working_memory.get_important(limit=3)
    logger.info(f"最重要的 3 条记忆:")
    for mem in important:
        logger.info(f"  - [重要性: {mem.importance:.2f}] {mem.content[:40]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 5. 上下文摘要
    # ------------------------------------------------------------------ #
    logger.info("5. 上下文摘要...")

    summary = working_memory.get_context_summary(max_length=200)
    logger.info(f"上下文摘要:\n{summary}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 6. 更新与删除
    # ------------------------------------------------------------------ #
    logger.info("6. 更新与删除...")

    update_success = working_memory.update(
        memory_id=mem1.id,
        importance=0.95,
        metadata={"topic": "Python", "type": "question", "priority": "high"},
    )
    logger.info(f"更新记忆 {mem1.id[:8]}... 成功: {update_success}")

    exists = working_memory.has_memory(mem1.id)
    logger.info(f"记忆 {mem1.id[:8]}... 存在: {exists}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 7. 遗忘机制
    # ------------------------------------------------------------------ #
    logger.info("7. 遗忘机制...")

    stats_before = working_memory.get_stats()
    logger.info(f"遗忘前记忆数量: {stats_before['count']}")

    forgotten = working_memory.forget(strategy="importance_based", threshold=0.1)
    logger.info(f"基于重要性遗忘: 删除了 {forgotten} 条记忆")

    stats_after = working_memory.get_stats()
    logger.info(f"遗忘后记忆数量: {stats_after['count']}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 8. 统计信息
    # ------------------------------------------------------------------ #
    logger.info("8. 统计信息...")

    stats = working_memory.get_stats()
    logger.info("工作记忆统计:")
    logger.info(f"  - 记忆数量: {stats['count']}")
    logger.info(f"  - 当前 token 数: {stats['current_tokens']}")
    logger.info(f"  - 最大容量: {stats['max_capacity']}")
    logger.info(f"  - 最大 token: {stats['max_tokens']}")
    logger.info(f"  - TTL: {stats['max_age_minutes']} 分钟")
    logger.info(f"  - 会话时长: {stats['session_duration_minutes']:.2f} 分钟")
    logger.info(f"  - 平均重要性: {stats['avg_importance']:.3f}")
    logger.info(f"  - 容量使用率: {stats['capacity_usage']:.1%}")
    logger.info(f"  - Token 使用率: {stats['token_usage']:.1%}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 9. 容量限制测试
    # ------------------------------------------------------------------ #
    logger.info("9. 容量限制测试...")

    logger.info(f"当前记忆数: {len(working_memory.get_all())}, 最大容量: {config.working_memory_capacity}")

    # 添加超过容量的记忆
    for i in range(8):
        mem = MemoryItem(
            id=generate_id(),
            content=f"这是第 {i+1} 条测试记忆，用于测试容量限制功能",
            memory_type="working",
            user_id="user_001",
            timestamp=datetime.now(),
            importance=0.3 + (i * 0.05),
            metadata={"test": True},
        )
        working_memory.add(mem)

    final_count = len(working_memory.get_all())
    logger.info(f"添加 8 条记忆后，实际记忆数: {final_count} (被容量限制)")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 10. 获取所有记忆
    # ------------------------------------------------------------------ #
    logger.info("10. 获取所有记忆...")

    all_memories = working_memory.get_all()
    logger.info(f"当前共有 {len(all_memories)} 条工作记忆:")
    for mem in all_memories[:5]:  # 只显示前 5 条
        logger.info(f"  - [{mem.id[:8]}] {mem.content[:40]}... (重要性: {mem.importance:.2f})")
    if len(all_memories) > 5:
        logger.info(f"  ... 还有 {len(all_memories) - 5} 条记忆")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #
    logger.info("清空工作记忆...")
    working_memory.clear()
    logger.info(f"清空后记忆数: {len(working_memory.get_all())}")

    logger.info("")
    logger.info("=== 示例结束 ===")


if __name__ == "__main__":
    main()
