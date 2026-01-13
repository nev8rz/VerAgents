# PlanAndSolveAgent 使用说明

基于“规划→执行”两阶段的智能体，适合多步骤推理、复杂问题拆解。
提示模板集中在 `veragents/core/prompts.py`（`PLANNER_PROMPT_TEMPLATE` / `EXECUTOR_PROMPT_TEMPLATE`），日志风格与 Simple/React/Reflection 一致。

## 主要特性
- 规划阶段：用规划提示生成 Python 列表形式的步骤。
- 执行阶段：按步骤顺序调用 LLM，传入原始问题、完整计划、已完成历史和当前步骤。
- 历史管理：记录到 Agent `_history`，便于后续连续对话。

## 快速上手
```python
from dotenv import load_dotenv
from veragents.agents import PlanAndSolveAgent
from veragents.core.llm import LLMClient

load_dotenv(override=True)
agent = PlanAndSolveAgent("planner-demo", LLMClient())
answer = agent.run("用 3 步讲明白如何冲一杯手冲咖啡")
print(answer)
```

## 提示词摘要
- 规划提示：要求输出 Python 列表，每项为字符串子任务。
- 执行提示：包含原始问题、完整计划、历史（无则“无”）、当前步骤；要求只输出当前步骤结果。

完整内容见 `veragents/core/prompts.py`。

## 注意事项
- 规划解析依赖 LLM 返回的可解析列表（尝试 code fence/原文提取）；格式异常会导致空计划。
- `system_prompt` 可自定义执行风格；`planner_prompt` / `executor_prompt` 可覆写默认模板。
- 若需要工具调用，可在执行阶段的提示里指示模型使用 `[TOOL:...]`，然后在 Agent 中扩展执行逻辑。
