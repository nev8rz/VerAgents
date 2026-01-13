"""Simple agent with optional tool calling via ToolRegistry."""

from __future__ import annotations

import json
import re
from datetime import datetime
from textwrap import shorten
from typing import Iterator, Optional

from loguru import logger as log

from veragents.core.agent import Agent
from veragents.core.config import Config
from veragents.core.llm import VerAgentsLLM
from veragents.core.messages import Message
from veragents.core.prompts import build_tool_section, format_tool_descriptions
from veragents.core.tool_utils import parse_tool_args
from veragents.tools.registry import ToolRegistry, ToolError


class SimpleAgent(Agent):
    """Lightweight agent that can optionally call tools using a simple markup."""

    def __init__(
        self,
        name: str,
        llm: VerAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        input_text: str,
        max_tool_calls: int = 3,
        force_tool_prompt: bool = True,
        **kwargs,
    ) -> str:
        """Run the agent; optionally perform iterative tool calls."""
        messages = self._bootstrap_messages(input_text)
        
        # Log initial messages once
        for m in messages:
            log.info(f"[{m.role}]\n{m.content}\n")

        iterations = 0
        prompted_for_tools = False
        has_executed_tools = False

        while iterations < max_tool_calls:
            reply = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
            tool_calls = self._extract_tool_calls(reply)

            if tool_calls:
                has_executed_tools = True

            if not (self.enable_tool_calling and self.tool_registry and tool_calls):
                # If tools are enabled but LLM未调用工具，提示一次强制使用工具
                # Only force if we haven't forced before AND haven't used tools yet
                if (
                    self.enable_tool_calling
                    and self.tool_registry
                    and force_tool_prompt
                    and not prompted_for_tools
                    and not has_executed_tools
                ):
                    messages.append(Message(role="assistant", content=reply))
                    log.info(f"[assistant]\n{reply}\n")
                    messages.append(
                        Message(
                            role="user",
                            content="请按要求使用工具完成任务，使用标记 [TOOL:tool_name:params]，不要直接回答。",
                        )
                    )
                    log.info(f"[user]\n请按要求使用工具完成任务...\n")
                    prompted_for_tools = True
                    iterations += 1
                    continue

                messages.append(Message(role="assistant", content=reply))
                # log.info(f"[assistant]\n{reply}\n")
                self._commit_history(input_text, reply)
                return reply

            # Log the raw reply (with tool calls) for visibility
            log.info(f"[assistant]\n{reply}\n")
            clean_reply = self._strip_tool_markup(reply, tool_calls)
            messages.append(Message(role="assistant", content=clean_reply))

            # Execute tools and feed results back as user message
            results = []
            for call in tool_calls:
                log.info(f"[tool]\n{call['name']} with {call['args']}\n")
                res = self._execute_tool(call["name"], call["args"])
                results.append(res)
                # Log tool result (truncated if too long)
                result_preview = res[:200] + "..." if len(res) > 200 else res
                log.info(f"[tool result]\n{result_preview}\n")
            messages.append(Message(role="user", content="工具结果:\n" + "\n".join(results)))
            iterations += 1

        # Fallback final call if loop exhausted
        final_reply = self.llm.chat(messages, **kwargs)  # type: ignore[arg-type]
        messages.append(Message(role="assistant", content=final_reply))
        # Don't log here - caller will handle the output
        self._commit_history(input_text, final_reply)
        return final_reply

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """Stream tokens back; no tool handling in streaming mode."""
        messages = self._bootstrap_messages(input_text)
        full_reply = ""
        for chunk in self.llm.chat(messages, stream=True, **kwargs):  # type: ignore[arg-type]
            full_reply += chunk
            yield chunk
        self._commit_history(input_text, full_reply)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _bootstrap_messages(self, user_text: str) -> list[Message]:
        messages: list[Message] = []
        sys_prompt = self._compose_system_prompt()
        if sys_prompt:
            messages.append(Message(role="system", content=sys_prompt))
        messages.extend(self._history)
        messages.append(Message(role="user", content=user_text))
        return messages

    def _compose_system_prompt(self) -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        base = self.system_prompt or "你是一个有帮助的助手。"
        base = f"当前日期: {current_date}\n{base}"

        if not (self.enable_tool_calling and self.tool_registry):
            return base

        tools = self.tool_registry.list_tools()
        if not tools:
            return base

        tool_list_str = format_tool_descriptions(self.tool_registry)
        return base + build_tool_section(tool_list_str)

    def _extract_tool_calls(self, text: str) -> list[dict]:
        pattern = r"\[TOOL:([^:]+):([^\]]+)\]"
        matches = re.findall(pattern, text)
        calls = []
        for name, arg_text in matches:
            calls.append({"name": name.strip(), "args": arg_text.strip(), "raw": f"[TOOL:{name}:{arg_text}]"})
        return calls

    def _strip_tool_markup(self, text: str, calls: list[dict]) -> str:
        cleaned = text
        for call in calls:
            cleaned = cleaned.replace(call["raw"], "")
        return cleaned.strip()

    def _execute_tool(self, tool_name: str, arg_text: str) -> str:
        if not self.tool_registry:
            return f"{tool_name}: 未配置工具注册表"
        try:
            args = parse_tool_args(self.tool_registry, tool_name, arg_text)
            result = self.tool_registry.dispatch(tool_name, args, strict=False)
            return f"{tool_name} -> {result}"
        except ToolError as exc:
            return f"{tool_name} -> 调用失败: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"{tool_name} -> 未知错误: {exc}"

    def _commit_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(role="user", content=user_text))
        self.add_message(Message(role="assistant", content=assistant_text))

    @staticmethod
    def _short(text: str, width: int = 140) -> str:
        return shorten(text.replace("\n", " "), width=width, placeholder="...")
