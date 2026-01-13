"""记忆系统基础类和配置。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """记忆项数据结构。"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """记忆系统配置。"""

    storage_path: str = "./memory_data"
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """记忆基类，定义通用接口和行为。"""

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项。"""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索相关记忆。"""
        raise NotImplementedError

    @abstractmethod
    def update(self, memory_id: str, content: str = None, importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """更新记忆。"""
        raise NotImplementedError

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆。"""
        raise NotImplementedError

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆。"""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息。"""
        raise NotImplementedError

    def _generate_id(self) -> str:
        """生成记忆ID。"""
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """计算记忆重要性。"""
        importance = base_importance

        if len(content) > 100:
            importance += 0.1

        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["MemoryItem", "MemoryConfig", "BaseMemory"]
