from .exceptions import VerAgentException
from .llm import VerAgentLLM
from .config import Config
from .message import Message

from typing import Optional
from abc import ABC, abstractmethod


# Agent base类
class Agent(ABC):

    def __init__(self,
                 name: str,
                 llm: VerAgentLLM,
                 system_prompt: Optional[str] = None,
                 config: Optional[Config] = None
                 ):

        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt
        self.config = config or Config.from_env()

        self._history: list[Message] = []

    @abstractmethod
    def run(self, input: str, **kwargs) -> str:
        # return self.llm.think([{"role": "user", "content": input}])
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history

    def __str__(self) -> str:
        return f"Agent(name={self.name}, llm={self.llm})"

    def __repr__(self) -> str:
        return self.__str__()
