import os

from typing import Dict, Any, Optional
from pydantic import BaseModel


class Config(BaseModel):
    default_model: str = "GLM-4.5V"
    default_provider: str = "zhipu"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    debug: bool = False
    log_level: str = "INFO"

    max_history_length: int = 100

    @classmethod
    def from_env(cls):
        return cls(
            debug=os.environ.get("DEBUG", False),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            temperature=float(os.environ.get("TEMPERATURE", 0.7)),
            max_tokens=int(os.environ.get("MAX_TOKENS", 1024)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
