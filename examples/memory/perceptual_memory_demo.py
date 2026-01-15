#!/usr/bin/env python
"""感知记忆（PerceptualMemory）完整使用示例。

演示功能：
1. 初始化感知记忆
2. 添加多模态记忆（文本、图像、音频）
3. 同模态语义检索
4. 按模态获取记忆
5. 跨模态搜索
6. 更新与删除
7. 遗忘机制
8. 统计信息

运行前请确保设置好环境变量：
- QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION
- EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL_NAME
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from loguru import logger

from veragents.memory.base import MemoryConfig, MemoryItem
from veragents.memory.types.perceptual import PerceptualMemory

load_dotenv()


def generate_id() -> str:
    """生成唯一 ID。"""
    return str(uuid.uuid4())


def main():
    logger.info("=== 感知记忆（PerceptualMemory）使用示例 ===\n")

    # ------------------------------------------------------------------ #
    # 1. 初始化感知记忆
    # ------------------------------------------------------------------ #
    logger.info("1. 初始化感知记忆...")
    config = MemoryConfig(
        storage_path="./memory_data",
        max_capacity=100,
        importance_threshold=0.1,
        perceptual_memory_modalities=["text", "image", "audio", "video"],
    )
    perceptual_memory = PerceptualMemory(config=config)
    logger.info("初始化完成\n")

    # ------------------------------------------------------------------ #
    # 2. 添加多模态记忆
    # ------------------------------------------------------------------ #
    logger.info("2. 添加多模态记忆...")

    # 文本记忆
    mem1 = MemoryItem(
        id=generate_id(),
        content="用户描述了一幅美丽的日落风景画，画面中有橙红色的天空和宁静的湖面",
        memory_type="perceptual",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=2),
        importance=0.8,
        metadata={
            "modality": "text",
            "context": {"topic": "风景描述", "emotion": "peaceful"},
            "tags": ["日落", "风景", "艺术"],
        },
    )
    perceptual_memory.add(mem1)
    logger.info(f"添加文本记忆: {mem1.id[:8]}...")

    mem2 = MemoryItem(
        id=generate_id(),
        content="用户分享了一段轻柔的钢琴曲，旋律优美，让人感到放松",
        memory_type="perceptual",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(hours=1),
        importance=0.7,
        metadata={
            "modality": "audio",
            "raw_data": "piano_melody_sample",  # 模拟音频数据标识
            "context": {"genre": "classical", "instrument": "piano"},
            "tags": ["音乐", "钢琴", "放松"],
        },
    )
    perceptual_memory.add(mem2)
    logger.info(f"添加音频记忆: {mem2.id[:8]}...")

    mem3 = MemoryItem(
        id=generate_id(),
        content="用户上传了一张高山雪景照片，山峰被白雪覆盖，阳光照耀下闪闪发光",
        memory_type="perceptual",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(minutes=30),
        importance=0.9,
        metadata={
            "modality": "image",
            "raw_data": "mountain_snow_image_data",  # 模拟图像数据标识
            "context": {"location": "Alps", "season": "winter"},
            "tags": ["雪山", "风景", "照片"],
        },
    )
    perceptual_memory.add(mem3)
    logger.info(f"添加图像记忆: {mem3.id[:8]}...")

    mem4 = MemoryItem(
        id=generate_id(),
        content="用户询问了关于色彩理论的问题，想了解互补色的搭配原则",
        memory_type="perceptual",
        user_id="user_002",
        timestamp=datetime.now() - timedelta(minutes=15),
        importance=0.6,
        metadata={
            "modality": "text",
            "context": {"topic": "艺术理论"},
            "tags": ["色彩", "设计", "教程"],
        },
    )
    perceptual_memory.add(mem4)
    logger.info(f"添加文本记忆 (user_002): {mem4.id[:8]}...")

    # 添加一个低重要性的记忆用于测试遗忘
    mem5 = MemoryItem(
        id=generate_id(),
        content="系统记录了一个简单的测试信号",
        memory_type="perceptual",
        user_id="user_001",
        timestamp=datetime.now() - timedelta(days=5),
        importance=0.05,
        metadata={
            "modality": "audio",
            "context": {"type": "test"},
            "tags": ["测试"],
        },
    )
    perceptual_memory.add(mem5)
    logger.info(f"添加低重要性记忆: {mem5.id[:8]}... (用于遗忘测试)")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 3. 同模态语义检索
    # ------------------------------------------------------------------ #
    logger.info("3. 同模态语义检索...")

    query = "美丽的自然风景"
    results = perceptual_memory.retrieve(query, limit=3, user_id="user_001", target_modality="text")
    logger.info(f"查询: '{query}' (模态: text)")
    logger.info(f"找到 {len(results)} 条相关记忆:")
    for i, mem in enumerate(results, 1):
        score = mem.metadata.get("relevance_score", 0)
        modality = mem.metadata.get("modality", "unknown")
        logger.info(f"  {i}. [{score:.3f}] [{modality}] {mem.content[:50]}...")

    logger.info("")

    query_all = "风景 照片"
    results_all = perceptual_memory.retrieve(query_all, limit=5)
    logger.info(f"查询: '{query_all}' (不限模态)")
    logger.info(f"找到 {len(results_all)} 条相关记忆:")
    for i, mem in enumerate(results_all, 1):
        score = mem.metadata.get("relevance_score", 0)
        modality = mem.metadata.get("modality", "unknown")
        logger.info(f"  {i}. [{score:.3f}] [{modality}] {mem.content[:50]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 4. 按模态获取记忆
    # ------------------------------------------------------------------ #
    logger.info("4. 按模态获取记忆...")

    text_memories = perceptual_memory.get_by_modality("text", limit=5)
    logger.info(f"文本模态记忆数量: {len(text_memories)}")
    for mem in text_memories:
        logger.info(f"  - {mem.content[:40]}...")

    image_memories = perceptual_memory.get_by_modality("image", limit=5)
    logger.info(f"图像模态记忆数量: {len(image_memories)}")

    audio_memories = perceptual_memory.get_by_modality("audio", limit=5)
    logger.info(f"音频模态记忆数量: {len(audio_memories)}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 5. 跨模态搜索
    # ------------------------------------------------------------------ #
    logger.info("5. 跨模态搜索...")

    cross_results = perceptual_memory.cross_modal_search(
        query="山峰 雪景",
        query_modality="text",
        target_modality="image",
        limit=3,
    )
    logger.info(f"跨模态搜索 (text -> image): 找到 {len(cross_results)} 条")
    for mem in cross_results:
        logger.info(f"  - [{mem.metadata.get('modality')}] {mem.content[:50]}...")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 6. 更新与删除
    # ------------------------------------------------------------------ #
    logger.info("6. 更新与删除...")

    update_success = perceptual_memory.update(
        memory_id=mem1.id,
        importance=0.95,
        metadata={"context": {"topic": "风景描述", "emotion": "very_peaceful"}},
    )
    logger.info(f"更新记忆 {mem1.id[:8]}... 成功: {update_success}")

    exists = perceptual_memory.has_memory(mem1.id)
    logger.info(f"记忆 {mem1.id[:8]}... 存在: {exists}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 7. 遗忘机制
    # ------------------------------------------------------------------ #
    logger.info("7. 遗忘机制...")

    stats_before = perceptual_memory.get_stats()
    logger.info(f"遗忘前记忆数量: {stats_before['count']}")

    forgotten = perceptual_memory.forget(strategy="importance_based", threshold=0.1)
    logger.info(f"基于重要性遗忘: 删除了 {forgotten} 条记忆")

    stats_after = perceptual_memory.get_stats()
    logger.info(f"遗忘后记忆数量: {stats_after['count']}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 8. 统计信息
    # ------------------------------------------------------------------ #
    logger.info("8. 统计信息...")

    stats = perceptual_memory.get_stats()
    logger.info("感知记忆统计:")
    logger.info(f"  - 记忆数量: {stats['count']}")
    logger.info(f"  - 感知数据数量: {stats['perceptions_count']}")
    logger.info(f"  - 模态分布: {stats['modality_counts']}")
    logger.info(f"  - 支持的模态: {stats['supported_modalities']}")
    logger.info(f"  - 平均重要性: {stats['avg_importance']:.3f}")
    logger.info(f"  - 向量存储: {stats.get('vector_store', {}).get('store_type', 'unknown')}")
    logger.info(f"  - 文档存储: {stats.get('document_store', {}).get('store_type', 'unknown')}")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 9. 获取所有记忆
    # ------------------------------------------------------------------ #
    logger.info("9. 获取所有记忆...")

    all_memories = perceptual_memory.get_all()
    logger.info(f"当前共有 {len(all_memories)} 条感知记忆:")
    for mem in all_memories:
        modality = mem.metadata.get("modality", "unknown")
        logger.info(f"  - [{mem.id[:8]}] [{modality}] {mem.content[:40]}... (重要性: {mem.importance:.2f})")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 10. 内容生成（简化演示）
    # ------------------------------------------------------------------ #
    logger.info("10. 内容生成...")

    generated = perceptual_memory.generate_content("描述一个美丽的自然场景", "text")
    if generated:
        logger.info(f"生成的内容:\n{generated}")
    else:
        logger.info("无法生成内容（没有足够的相关记忆）")

    logger.info("")

    # ------------------------------------------------------------------ #
    # 清理（可选）
    # ------------------------------------------------------------------ #
    # 取消注释以下代码来清空所有感知记忆
    # logger.info("清空所有感知记忆...")
    # perceptual_memory.clear()
    # logger.info("清空完成")

    logger.info("=== 示例结束 ===")


if __name__ == "__main__":
    main()
