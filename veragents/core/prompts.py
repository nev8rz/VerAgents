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

# Planner / Executor prompts for PlanAndSolveAgent
PLANNER_PROMPT_TEMPLATE = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成多个按序排列的可执行子任务。

问题: {question}

请仅输出一个 Python 列表，每个元素为一个字符串子任务，格式示例:
[\"步骤1\", \"步骤2\", \"步骤3\"]"""

EXECUTOR_PROMPT_TEMPLATE = """你是一位 AI 执行专家，将严格按计划逐步完成任务。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请只输出“当前步骤”的最终答案，不要额外解释。"""

EXECUTOR_FINAL_PROMPT_TEMPLATE = """你是一位 AI 执行专家。基于已完成的所有步骤结果，给出最终答案。

# 原始问题:
{question}

# 完整计划:
{plan}

# 已完成步骤与结果:
{history}

请整合以上信息输出最终答案：简洁、直接，不要重复逐条列出步骤标题。"""


# Reflection Agent 默认模板
REFLECTION_PROMPTS = {
    "initial": """请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。""",
    "reflect": """请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。""",
    "refine": """请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。"""
}
