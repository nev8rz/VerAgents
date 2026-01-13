"""RAG 文档处理器占位实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from veragents.memory.base import MemoryItem


@dataclass
class Document:
    """RAG 文档封装。"""

    id: str
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_memory_item(self) -> MemoryItem:
        """转换为通用 MemoryItem。"""
        return MemoryItem(id=self.id, content=self.content, metadata=self.metadata)


__all__ = ["Document"]
