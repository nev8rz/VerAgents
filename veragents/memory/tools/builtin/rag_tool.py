"""RAG 工具占位实现，后续按需补充。"""

from __future__ import annotations

from loguru import logger as log

from veragents.tools import ToolError, register_tool


@register_tool(name="rag_query")
def rag_query(question: str, top_k: int = 5) -> dict:
    """智能问答占位，等待后续实现。"""
    log.warning("rag_query called but not implemented | question_len={} top_k={}", len(question), top_k)
    raise ToolError("rag_query", "RAG tool not implemented yet", "NotImplemented")


__all__ = ["rag_query"]
