"""Reflection agent: draft -> reflect -> refine."""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger as log

from veragents.core.agent import Agent
from veragents.core.config import Config
from veragents.core.llm import VerAgentsLLM
from veragents.core.messages import Message
from veragents.core.prompts import REFLECTION_PROMPTS


class _Memory:
    """Lightweight memory for iterations."""

    def __init__(self):
        self.records: List[Dict[str, str]] = []

    def add(self, kind: str, content: str) -> None:
        self.records.append({"type": kind, "content": content})

    def last_execution(self) -> str:
        for rec in reversed(self.records):
            if rec["type"] == "execution":
                return rec["content"]
        return ""

    def trajectory(self) -> str:
        parts = []
        for rec in self.records:
            label = "执行" if rec["type"] == "execution" else "反思"
            parts.append(f"[{label}]\n{rec['content']}")
        return "\n\n".join(parts)


class ReflectionAgent(Agent):
    """Draft -> Reflect -> Refine loop for self-improvement tasks."""

    def __init__(
        self,
        name: str,
        llm: VerAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.prompts = prompts or REFLECTION_PROMPTS
        self.memory = _Memory()

    def run(self, input_text: str, **kwargs) -> str:
        self.memory = _Memory()

        # Initial draft
        initial_prompt = self.prompts["initial"].format(task=input_text)
        log.info(f"[user]\n{initial_prompt}\n")
        draft = self._call_llm(initial_prompt, **kwargs)
        log.info(f"[assistant]\n{draft}\n")
        self.memory.add("execution", draft)

        # Iterate reflect -> refine
        for i in range(self.max_iterations):
            last = self.memory.last_execution()
            reflect_prompt = self.prompts["reflect"].format(task=input_text, content=last)
            log.info(f"[user]\n{reflect_prompt}\n")
            feedback = self._call_llm(reflect_prompt, **kwargs)
            log.info(f"[assistant]\n{feedback}\n")
            self.memory.add("reflection", feedback)

            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                final = last
                self._commit_history(input_text, final)
                return final

            refine_prompt = self.prompts["refine"].format(task=input_text, last_attempt=last, feedback=feedback)
            log.info(f"[user]\n{refine_prompt}\n")
            refined = self._call_llm(refine_prompt, **kwargs)
            log.info(f"[assistant]\n{refined}\n")
            self.memory.add("execution", refined)

        final_result = self.memory.last_execution()
        self._commit_history(input_text, final_result)
        return final_result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _call_llm(self, prompt: str, **kwargs) -> str:
        messages = [Message(role="system", content=self.system_prompt or ""), Message(role="user", content=prompt)]
        for m in messages:
            if m.content:
                log.info(f"[{m.role}]\n{m.content}\n")
        reply = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
        log.info(f"[assistant]\n{reply}\n")
        return reply

    def _commit_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(role="user", content=user_text))
        self.add_message(Message(role="assistant", content=assistant_text))
