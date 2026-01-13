"""Plan-and-solve agent: plan decomposition then sequential execution."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional

from loguru import logger as log

from veragents.core.agent import Agent
from veragents.core.config import Config
from veragents.core.llm import VerAgentsLLM
from veragents.core.messages import Message
from veragents.core.prompts import (
    EXECUTOR_FINAL_PROMPT_TEMPLATE,
    EXECUTOR_PROMPT_TEMPLATE,
    PLANNER_PROMPT_TEMPLATE,
)


class PlanAndSolveAgent(Agent):
    """Decompose a question into steps, then execute step-by-step."""

    def __init__(
        self,
        name: str,
        llm: VerAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        planner_prompt: Optional[str] = None,
        executor_prompt: Optional[str] = None,
        executor_final_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.planner_prompt = planner_prompt or PLANNER_PROMPT_TEMPLATE
        self.executor_prompt = executor_prompt or EXECUTOR_PROMPT_TEMPLATE
        self.executor_final_prompt = executor_final_prompt or EXECUTOR_FINAL_PROMPT_TEMPLATE

    def run(self, input_text: str, **kwargs) -> str:
        # Plan
        plan = self._plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            self._commit_history(input_text, final_answer)
            return final_answer

        # Execute steps
        history: List[Dict[str, str]] = []
        for idx, step in enumerate(plan, 1):
            exec_prompt = self.executor_prompt.format(
                question=input_text,
                plan=plan,
                history=self._format_history(history),
                current_step=step,
            )
            messages = [Message(role="system", content=self.system_prompt or ""), Message(role="user", content=exec_prompt)]
            self._log_messages("executor", messages)
            response = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
            log.info(f"[executor-assistant]\n{response}\n")
            step_answer = response or ""
            history.append({"step": step, "result": step_answer})

        # Final synthesis over all steps/results
        final_prompt = self.executor_final_prompt.format(
            question=input_text,
            plan=plan,
            history=self._format_history(history),
        )
        messages = [Message(role="system", content=self.system_prompt or ""), Message(role="user", content=final_prompt)]
        self._log_messages("executor-final", messages)
        final_answer = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
        log.info(f"[executor-final-assistant]\n{final_answer}\n")

        self._commit_history(input_text, final_answer)
        return final_answer

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.planner_prompt.format(question=question)
        messages = [Message(role="system", content=self.system_prompt or ""), Message(role="user", content=prompt)]
        self._log_messages("planner", messages)
        reply = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
        log.info(f"[planner-assistant]\n{reply}\n")
        plan = self._parse_plan(reply or "")
        return plan

    @staticmethod
    def _parse_plan(text: str) -> List[str]:
        # Try to extract python list in code fence first
        candidates = []
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if "[" in part and "]" in part:
                    candidates.append(part)
        candidates.append(text)

        for cand in candidates:
            try:
                start = cand.find("[")
                end = cand.rfind("]")
                if start != -1 and end != -1 and end > start:
                    plan_str = cand[start : end + 1]
                    parsed = ast.literal_eval(plan_str)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
            except (ValueError, SyntaxError):
                continue
        return []

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        if not history:
            return "无"
        lines = []
        for idx, rec in enumerate(history, 1):
            lines.append(f"步骤 {idx}: {rec['step']}\n结果: {rec['result']}")
        return "\n\n".join(lines)

    def _commit_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(role="user", content=user_text))
        self.add_message(Message(role="assistant", content=assistant_text))

    @staticmethod
    def _log_messages(prefix: str, messages: List[Message]) -> None:
        for m in messages:
            if m.content:
                log.info(f"[{prefix}-{m.role}]\n{m.content}\n")
