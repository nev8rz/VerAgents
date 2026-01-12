from datetime import datetime
from typing import Dict, Any, Literal, Optional

from pydantic import BaseModel


MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        content: str,
        role: MessageRole,
        **kwargs
    ):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp") or datetime.now(),
            metadata=kwargs.get("metadata") or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self):
        return f"[{self.role}]: {self.content}"
