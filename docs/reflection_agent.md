# ReflectionAgent 使用说明

基于“草稿 → 反思 → 优化”三段式的自我改进智能体，适合代码/文档/报告等需要多轮打磨的任务。提示模板集中在 `veragents/core/prompts.py`，日志风格与 Simple/React 一致。

## 主要特性
- 初稿（initial）→ 反思（reflect）→ 优化（refine）循环，默认最多 3 轮。
- 模板可自定义，也可直接使用内置 `REFLECTION_PROMPTS`。
- 日志：输出每轮的用户提示与模型回复，便于观察迭代过程。
- 不依赖工具注册表，纯粹的自我反思流程（如需工具可扩展）。

## 快速上手
```python
from dotenv import load_dotenv
from veragents.agents import ReflectionAgent
from veragents.core.llm import LLMClient

load_dotenv(override=True)
agent = ReflectionAgent("reflect-demo", LLMClient(), max_iterations=3)
res = agent.run("请写一个 100 字的产品介绍，并优化用词。")
print(res)
```

## 提示词模板（摘要）
- initial: 生成初稿。
- reflect: 审查当前回答，指出不足/改进建议；如足够好，回复“无需改进”。
- refine: 基于反馈改进回答。

完整内容见 `veragents/core/prompts.py` 的 `REFLECTION_PROMPTS`。

## 注意事项
- `max_iterations` 控制反思-优化循环次数；遇到“无需改进”会提前停止。
- 历史会记录到 Agent `_history`，便于后续连续对话。
- 若需接入工具链，可扩展 run 流程或提示词，引导 LLM 输出 `[TOOL:...]` 调用。***
