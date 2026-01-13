# ReActAgent 使用说明

基于 ReAct（Reasoning + Acting）范式的 Agent，实现推理 + 工具调用的闭环。提示词模板集中在 `veragents/core/prompts.py`，与 SimpleAgent 工具调用说明保持一致。

- 工具调用格式统一为 `[TOOL:tool_name:params]`（params 可为 JSON 或 `key=value` 逗号分隔）；结束用 `Finish[...]`。

## 核心特性
- 使用 `Thought` / `Action` / `Finish` 模式驱动推理与工具调用。
- 支持多轮工具调用（默认最大 5 步）。
- 工具来自现有 `ToolRegistry`（`@tool`/`@register_tool` 注册的工具均可用）。
- Action 格式：`tool_name[参数]`；结束用 `Finish[答案]`。
- 参数支持 JSON（优先）或纯文本（映射为 `{"input": ...}`）。

## 快速上手
```python
from veragents.agents import ReActAgent
from veragents.core.llm import LLMClient
from veragents.tools import registry, tool

# 定义并注册一个简单工具
@tool
def double(input: str) -> str:
    return f"double({input})"

llm = LLMClient()
agent = ReActAgent("react-demo", llm, tool_registry=registry, max_steps=5)

answer = agent.run("请用工具计算并返回 double(21)")
print(answer)
```

## 提示词与流程
- 系统提示包含可用工具列表与 ReAct 流程要求：
  - 每轮必须输出 `Thought:` 和 `Action:`。
  - Action 使用 `tool_name[参数]`；结束用 `Finish[答案]`。
  - 需要外部信息时必须调用工具，不要臆测。
- Agent 将 LLM 回复解析出 Thought/Action，调用对应工具（`registry.dispatch`），将 `Observation` 记录进历史后继续下一轮。

## 注意事项
- 确保所需工具已注册到 `ToolRegistry`（默认全局 `registry`）。
- JSON 参数请确保格式正确；非 JSON 将包装为 `{"input": <text>}`。
- `max_steps` 控制最多迭代次数；超出后返回兜底答复。
