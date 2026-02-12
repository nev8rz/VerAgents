"""RAG 工具 — 提供完整的 RAG 能力。

功能：
- 添加多格式文档（PDF、Office、图片、音频等）
- 智能检索与召回
- LLM 增强问答
- 知识库管理
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from loguru import logger as log

from veragents.memory.rag.pipeline import RAGPipeline, create_rag_pipeline
from veragents.tools import register_tool


# ------------------------------------------------------------------ #
# 全局管道管理
# ------------------------------------------------------------------ #

_pipelines: Dict[str, RAGPipeline] = {}


def _get_pipeline(namespace: str = "default") -> RAGPipeline:
    """获取或创建 RAG 管道实例。"""
    if namespace not in _pipelines:
        _pipelines[namespace] = create_rag_pipeline(
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            rag_namespace=namespace,
        )
    return _pipelines[namespace]


# ------------------------------------------------------------------ #
# 注册工具
# ------------------------------------------------------------------ #


@register_tool(name="rag_ingest_file")
def rag_ingest_file(file_path: str, namespace: str = "default") -> dict:
    """载入文件到 RAG 知识库。

    支持格式：PDF、Word、Excel、PPT、TXT、MD、CSV、JSON、HTML、代码文件等。

    Args:
        file_path: 文件路径
        namespace: 命名空间

    Returns:
        dict: 包含 chunks_indexed 和 status
    """
    pipeline = _get_pipeline(namespace)
    count = pipeline.ingest_file(file_path)
    return {"status": "ok", "file": file_path, "chunks_indexed": count, "namespace": namespace}


@register_tool(name="rag_ingest_text")
def rag_ingest_text(text: str, source: str = "direct_input", namespace: str = "default") -> dict:
    """直接载入文本到 RAG 知识库。

    Args:
        text: 文本内容
        source: 来源标识
        namespace: 命名空间

    Returns:
        dict: 包含 chunks_indexed 和 status
    """
    pipeline = _get_pipeline(namespace)
    count = pipeline.ingest_text(text, source=source)
    return {"status": "ok", "source": source, "chunks_indexed": count, "namespace": namespace}


@register_tool(name="rag_ingest_directory")
def rag_ingest_directory(
    dir_path: str,
    extensions: Optional[List[str]] = None,
    namespace: str = "default",
) -> dict:
    """递归载入目录下的文件到 RAG 知识库。

    Args:
        dir_path: 目录路径
        extensions: 文件扩展名过滤列表（如 ['.txt', '.md']）
        namespace: 命名空间

    Returns:
        dict: 包含 chunks_indexed 和 status
    """
    pipeline = _get_pipeline(namespace)
    count = pipeline.ingest_directory(dir_path, extensions=extensions)
    return {"status": "ok", "directory": dir_path, "chunks_indexed": count, "namespace": namespace}


@register_tool(name="rag_search")
def rag_search(
    query: str,
    top_k: int = 5,
    enable_mqe: bool = False,
    enable_hyde: bool = False,
    namespace: str = "default",
) -> dict:
    """检索 RAG 知识库。

    Args:
        query: 查询文本
        top_k: 返回数量
        enable_mqe: 启用多查询扩展
        enable_hyde: 启用假设文档嵌入
        namespace: 命名空间

    Returns:
        dict: 包含 results 列表
    """
    pipeline = _get_pipeline(namespace)
    results = pipeline.search(
        query=query,
        top_k=top_k,
        enable_mqe=enable_mqe,
        enable_hyde=enable_hyde,
    )
    return {"status": "ok", "query": query, "results": results, "count": len(results)}


@register_tool(name="rag_query")
def rag_query(
    question: str,
    top_k: int = 5,
    enable_mqe: bool = False,
    enable_hyde: bool = False,
    namespace: str = "default",
) -> dict:
    """RAG 智能问答：检索知识库并用 LLM 生成答案。

    Args:
        question: 用户问题
        top_k: 检索数量
        enable_mqe: 启用多查询扩展
        enable_hyde: 启用假设文档嵌入
        namespace: 命名空间

    Returns:
        dict: 包含 answer、sources
    """
    pipeline = _get_pipeline(namespace)
    result = pipeline.query(
        question=question,
        top_k=top_k,
        enable_mqe=enable_mqe,
        enable_hyde=enable_hyde,
    )
    return {
        "status": "ok",
        "question": question,
        "answer": result["answer"],
        "sources_count": len(result["sources"]),
        "sources": result["sources"],
    }


@register_tool(name="rag_stats")
def rag_stats(namespace: str = "default") -> dict:
    """获取 RAG 知识库状态信息。

    Args:
        namespace: 命名空间

    Returns:
        dict: 统计信息
    """
    pipeline = _get_pipeline(namespace)
    stats = pipeline.get_stats()
    return {"status": "ok", "stats": stats}


@register_tool(name="rag_clear")
def rag_clear(namespace: str = "default") -> dict:
    """清空 RAG 知识库。

    Args:
        namespace: 命名空间

    Returns:
        dict: 操作结果
    """
    pipeline = _get_pipeline(namespace)
    pipeline.clear()
    return {"status": "ok", "namespace": namespace, "message": "RAG 知识库已清空"}


__all__ = [
    "rag_ingest_file",
    "rag_ingest_text",
    "rag_ingest_directory",
    "rag_search",
    "rag_query",
    "rag_stats",
    "rag_clear",
]
