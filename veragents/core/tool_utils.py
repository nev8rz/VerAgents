"""Shared helpers for tool argument parsing and formatting."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from veragents.tools.registry import ToolRegistry


def parse_tool_args(registry: ToolRegistry, tool_name: str, raw: str) -> Dict[str, Any]:
    """
    Parse raw tool arguments into a dict with a common strategy:
    1) JSON object -> dict
    2) key=value pairs (comma separated) -> dict
    3) otherwise: try mapping to the first arg name from args_model; fallback to {"input": raw}
    """
    raw = (raw or "").strip()

    # 1) JSON object
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 2) key=value[,key=value]
    if "=" in raw:
        args: Dict[str, Any] = {}
        for part in [p for p in raw.split(",") if p.strip()]:
            if "=" in part:
                k, v = part.split("=", 1)
                args[k.strip()] = v.strip()
        if args:
            return args

    # 3) positional -> map to first param if available
    tool = registry.get(tool_name) if registry else None
    if tool and getattr(tool, "args_model", None):
        schema = tool.args_model.model_json_schema()
        props = list(schema.get("properties", {}).keys())
        if props:
            return {props[0]: raw}

    return {"input": raw}
