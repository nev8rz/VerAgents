"""Tool chaining support."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger as log

from .registry import ToolRegistry


class ToolChain:
    """A simple ordered tool chain."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: Optional[str] = None) -> None:
        """
        Add a step to the chain.

        :param tool_name: Registered tool name.
        :param input_template: Template string; rendered with context via str.format(**ctx).
                               If rendering yields a JSON object string, it will be parsed to dict;
                               otherwise it will be sent as {"input": rendered}.
        :param output_key: Key to store the result in context for later steps. Defaults to step_N_result.
        """
        step = {
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result",
        }
        self.steps.append(step)

    def execute(self, registry: ToolRegistry, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute chain synchronously."""
        if not self.steps:
            raise ValueError(f"ToolChain '{self.name}' is empty")

        ctx: Dict[str, Any] = dict(context or {})
        ctx["input"] = input_data
        result: Any = input_data

        for idx, step in enumerate(self.steps, start=1):
            tool_name = step["tool_name"]
            tpl = step["input_template"]
            output_key = step["output_key"]

            try:
                rendered = tpl.format(**ctx)
            except KeyError as exc:
                raise ValueError(f"ToolChain '{self.name}' step {idx} format missing key: {exc}") from exc

            args = self._render_to_args(rendered)
            log.info("ToolChain {} step {}/{} calling {} with args {}", self.name, idx, len(self.steps), tool_name, args)
            result = registry.dispatch(tool_name, args, strict=False)
            ctx[output_key] = result

        return result

    @staticmethod
    def _render_to_args(rendered: str) -> Dict[str, Any]:
        text = rendered.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"input": rendered}


class ToolChainManager:
    """Manage multiple tool chains."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None:
        self.chains[chain.name] = chain

    def execute_chain(self, chain_name: str, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        chain = self.chains.get(chain_name)
        if not chain:
            raise KeyError(f"ToolChain '{chain_name}' not found")
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        return list(self.chains.keys())

    def get_chain(self, chain_name: str) -> Optional[ToolChain]:
        return self.chains.get(chain_name)
