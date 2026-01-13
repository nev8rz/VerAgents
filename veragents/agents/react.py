"""ReAct-style agent: reasoning + tool calling."""

from __future__ import annotations

import json
import re
from textwrap import shorten
from typing import List, Optional, Tuple

from loguru import logger as log

from veragents.core.agent import Agent
from veragents.core.config import Config
from veragents.core.llm import VerAgentsLLM
from veragents.core.messages import Message
from veragents.core.prompts import REACT_PROMPT_TEMPLATE, format_tool_descriptions
from veragents.core.tool_utils import parse_tool_args
from veragents.tools.registry import ToolRegistry, ToolError

# Default ReAct prompt template
DEFAULT_REACT_PROMPT = REACT_PROMPT_TEMPLATE


class ReActAgent(Agent):
    """ReAct (Reasoning + Acting) agent with tool calling."""

    def __init__(
        self,
        name: str,
        llm: VerAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        prompt_template: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.prompt_template = prompt_template or DEFAULT_REACT_PROMPT
        self._trace: List[str] = []

    def run(self, input_text: str, **kwargs) -> str:
        self._trace.clear()
        step = 0
        # Log initial messages (simple-style)
        boot = self._bootstrap_messages(input_text)
        for m in boot:
            if m.content:
                log.info(f"[{m.role}]\n{m.content}\n")

        while step < self.max_steps:
            step += 1
            prompt = self._build_prompt(input_text)
            messages = [Message(role="system", content=self.system_prompt or ""), Message(role="user", content=prompt)]
            reply = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
            thought, action = self._parse_output(reply)
            log.info(f"[assistant]\n{reply}\n")
            if thought:
                self._trace.append(f"Thought: {thought}")
            if not action:
                self._trace.append("Observation: 无有效 Action，结束。")
                break

            if action.startswith("Finish"):
                answer = self._parse_finish(action)
                self._commit_history(input_text, answer)
                return answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                self._trace.append("Observation: Action 格式错误")
                continue

            observation = self._call_tool(tool_name, tool_input)
            log.info(f"[tool]\n{tool_name} with {tool_input}\n")
            obs_preview = observation
            if len(obs_preview) > 200:
                obs_preview = obs_preview[:200] + "..."
            log.info(f"[tool result]\n{obs_preview}\n")
            self._trace.append(f"Action: {action}")
            self._trace.append(f"Observation: {observation}")

        final = "抱歉，未能在限定步数内完成。"
        self._commit_history(input_text, final)
        return final

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _bootstrap_messages(self, question: str) -> list[Message]:
        sys_prompt = self.system_prompt or ""
        prompt = self._build_prompt(question)
        messages: list[Message] = []
        if sys_prompt:
            messages.append(Message(role="system", content=sys_prompt))
        messages.append(Message(role="user", content=prompt))
        return messages

    def _build_prompt(self, question: str) -> str:
        tools_list = self.tool_registry.list_tools() if self.tool_registry else []
        tools_desc = format_tool_descriptions(self.tool_registry) if tools_list else "无可用工具"
        history = "\n".join(self._trace) if self._trace else "（无）"
        return self.prompt_template.format(tools=tools_desc, question=question, history=history)

    @staticmethod
    def _parse_output(text: str) -> Tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought:\s*(.*)", text)
        action_match = re.search(r"Action:\s*(.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    @staticmethod
    def _parse_action(action_text: str) -> Tuple[Optional[str], Optional[str]]:
        # Support [TOOL:foo:bar] or foo[bar]
        tool_call = re.match(r"\[TOOL:([^:]+):([^\]]+)\]", action_text, re.IGNORECASE)
        if tool_call:
            return tool_call.group(1).strip(), tool_call.group(2).strip()
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    @staticmethod
    def _parse_finish(action_text: str) -> str:
        match = re.match(r"Finish\[(.*)\]", action_text, re.IGNORECASE)
        return match.group(1).strip() if match else action_text

    def _call_tool(self, tool_name: str, raw_input: str) -> str:
        if not self.tool_registry:
            return "未配置工具注册表"
        args = parse_tool_args(self.tool_registry, tool_name, raw_input)
        try:
            result = self.tool_registry.dispatch(tool_name, args, strict=False)
            return str(result)
        except ToolError as exc:
            return f"工具调用失败: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"工具调用异常: {exc}"

    def _commit_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(role="user", content=user_text))
        self.add_message(Message(role="assistant", content=assistant_text))

    @staticmethod
    def _short(text: str, width: int = 140) -> str:
        return shorten(str(text).replace("\n", " "), width=width, placeholder="...")
