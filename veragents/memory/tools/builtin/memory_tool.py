"""记忆工具（占位实现）。"""

from __future__ import annotations

from loguru import logger as log

from veragents.tools import register_tool


@register_tool(name="memory_status")
def memory_status() -> dict:
    """返回记忆子系统的占位状态，后续可替换为真实查询。"""
    status = {
        "status": "placeholder",
        "available_memories": ["working", "episodic", "semantic", "perceptual"],
        "message": "Memory tool not implemented yet",
    }
    log.info("memory_status called | status={}", status["status"])
    return status


__all__ = ["memory_status"]
