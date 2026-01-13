"""Function-call Agent using OpenAI tools/function calling style."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Union

from loguru import logger as log

from veragents.core.agent import Agent
from veragents.core.config import Config
from veragents.core.llm import VerAgentsLLM
from veragents.core.messages import Message
from veragents.core.tool_utils import parse_tool_args
from veragents.tools.registry import ToolRegistry


class FunctionCallAgent(Agent):
    """Agent that drives OpenAI function-calling with registered tools."""

    def __init__(
        self,
        name: str,
        llm: VerAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_calling: bool = True,
        default_tool_choice: Union[str, dict] = "auto",
        max_tool_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.default_tool_choice = default_tool_choice
        self.max_tool_iterations = max_tool_iterations

    def run(
        self,
        input_text: str,
        *,
        max_tool_iterations: Optional[int] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> str:
        messages: List[dict[str, Any]] = []
        sys_prompt = self.system_prompt or "你是一个可靠的AI助理，必要时会调用工具完成任务。"
        messages.append({"role": "system", "content": sys_prompt})

        for m in self._history:
            messages.append({"role": m.role, "content": m.content})

        messages.append({"role": "user", "content": input_text})
        self._log_messages(messages)

        tool_schemas = self._build_tool_schemas()
        if not (self.enable_tool_calling and tool_schemas):
            reply = self.llm.chat([Message(role="user", content=input_text)], **kwargs)  # type: ignore[arg-type]
            self._commit_history(input_text, reply)
            return reply

        iterations_limit = max_tool_iterations if max_tool_iterations is not None else self.max_tool_iterations
        effective_tool_choice: Union[str, dict] = tool_choice if tool_choice is not None else self.default_tool_choice

        final_response = ""
        iteration = 0

        while iteration < iterations_limit:
            resp = self._invoke_with_tools(messages, tool_schemas, effective_tool_choice, **kwargs)
            choice = resp.choices[0]
            msg = choice.message
            content = self._extract_content(msg.content)
            tool_calls = list(msg.tool_calls or [])

            if tool_calls:
                # append assistant message with tool_calls
                assistant_payload: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": []}
                for call in tool_calls:
                    assistant_payload["tool_calls"].append(
                        {
                            "id": call.id,
                            "type": call.type,
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                    )
                messages.append(assistant_payload)
                log.info(f"[assistant]\n{content}\n")

                for call in tool_calls:
                    tool_name = call.function.name
                    args_dict = self._parse_arguments(tool_name, call.function.arguments)
                    log.info(f"[tool]\n{tool_name} with {args_dict}\n")
                    result = self._execute_tool(tool_name, args_dict)
                    log.info(f"[tool result]\n{result}\n")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": tool_name,
                            "content": result,
                        }
                    )
                iteration += 1
                continue

            # no tool call, final content
            final_response = content
            messages.append({"role": "assistant", "content": final_response})
            log.info(f"[assistant]\n{final_response}\n")
            break

        if not final_response:
            # force a final answer without tools
            resp = self._invoke_with_tools(messages, tool_schemas, tool_choice="none", **kwargs)
            final_response = self._extract_content(resp.choices[0].message.content)
            messages.append({"role": "assistant", "content": final_response})
            log.info(f"[assistant]\n{final_response}\n")

        self._commit_history(input_text, final_response)
        return final_response

    def _commit_history(self, user_text: str, assistant_text: str) -> None:
        self.add_message(Message(role="user", content=user_text))
        self.add_message(Message(role="assistant", content=assistant_text))

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """Streaming not implemented for function-calling; fallback to single run."""
        yield self.run(input_text, **kwargs)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_tool_schemas(self) -> List[dict[str, Any]]:
        if not (self.enable_tool_calling and self.tool_registry):
            return []
        try:
            return self.tool_registry.export_openai_tools()
        except Exception:
            return []

    def _invoke_with_tools(self, messages: List[dict[str, Any]], tools: List[dict[str, Any]], tool_choice, **kwargs):
        client = getattr(self.llm, "client", None)
        if client is None:
            raise RuntimeError("LLM client未暴露底层 OpenAI client，无法使用 function calling。")
        client_kwargs = dict(kwargs)
        return client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **client_kwargs,
        )

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        if not self.tool_registry:
            return "未配置工具注册表"
        try:
            result = self.tool_registry.dispatch(tool_name, args, strict=False)
            return str(result)
        except Exception as exc:  # ToolError already stringified in dispatch result
            return f"工具调用异常: {exc}"

    def _parse_arguments(self, tool_name: str, raw_args: Optional[str]) -> Dict[str, Any]:
        if raw_args is None:
            return {}
        # raw_args is a JSON string per OpenAI spec
        return parse_tool_args(self.tool_registry, tool_name, raw_args) if self.tool_registry else {}

    @staticmethod
    def _extract_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content)

    @staticmethod
    def _log_messages(messages: List[dict[str, Any]]) -> None:
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if content:
                log.info(f"[{role}]\n{content}\n")
