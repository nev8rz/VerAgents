"""Centralized prompt templates and helpers."""

from __future__ import annotations

TOOL_CALL_INSTRUCTIONS = (
    "当问题涉及外部/实时信息（如搜索、天气、计算等）时，必须调用工具，不要臆测。\n"
    "调用格式: [TOOL:tool_name:params]\n"
    "params 可为 JSON 格式。例如:\n"
    "[TOOL:search_web:{\"query\":\"python\",\"num\":3}]\n"
    "[TOOL:get_current_weather:{\"city\":\"北京\"}]\n"
    "可连续调用多个工具，先写出工具标记，待结果返回再给最终回答。\n"
)


def build_tool_section(tool_descriptions: str) -> str:
    """Compose a standard tool usage section given a formatted tool list."""
    return f"\n\n可用工具:\n{tool_descriptions}\n{TOOL_CALL_INSTRUCTIONS}"


def format_tool_descriptions(registry) -> str:
    """Build detailed tool descriptions with parameters (if args_model exists)."""
    tool_descriptions = []
    for name in registry.list_tools():
        tool = registry.get(name)
        if not tool:
            continue
        desc = tool.description or ""
        params_info = ""
        if getattr(tool, "args_model", None):
            schema = tool.args_model.model_json_schema()
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            param_parts = []
            for param_name, param_schema in properties.items():
                param_type = param_schema.get("type", "any")
                optional = "" if param_name in required else "?"
                desc_hint = param_schema.get("description", "")
                default = ""
                if "default" in param_schema:
                    default_val = param_schema["default"]
                    default = f"={default_val}"
                param_parts.append(f"{param_name}{optional}:{param_type}{default} {desc_hint}")
            if param_parts:
                params_info = " | ".join(param_parts)
        tool_descriptions.append(f"- **{name}**\n  {desc}\n  参数: {params_info}")
    return "\n".join(tool_descriptions) if tool_descriptions else "无可用工具"


# ReAct 默认模板（与工具提示统一由此集中管理）
REACT_PROMPT_TEMPLATE = """你是一个具备推理和行动能力的AI助手。

可用工具:
{tools}

工作流程（每次回复必须包含 Thought 与 Action，并且一次只执行一个 Action）:
Thought: 分析问题，说明下一步要做什么。
Action: 使用以下格式之一（与工具调用规范一致）：
  - [TOOL:tool_name:params]  # params 可为 JSON 或 key=value 逗号分隔
  - Finish[最终答案]

重要提醒:
1) 遇到需要外部信息时必须用工具，不要臆测。
2) Action 必须严格使用 tool_name[参数] 格式；结束时用 Finish[答案]。
3) 如果结果不足以回答，继续调用工具。

Question: {question}
History:
{history}

请开始 Thought 和 Action："""
