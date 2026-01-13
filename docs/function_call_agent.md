# FunctionCallAgent 使用说明

基于 OpenAI function-calling 的工具调用 Agent，自动读取 `ToolRegistry.export_openai_tools()` 生成的 schema 传给模型。

## 特性
- 统一的工具参数解析策略（复用 `parse_tool_args`）：支持 JSON、key=value、或首参数名映射。
- 多轮函数调用（默认 3 轮），无结果时强制 `tool_choice="none"` 产出最终回答。
- 日志风格与其它 Agent 一致：打印 system/user/assistant、工具调用与结果。

## 快速上手
```python
from dotenv import load_dotenv
from veragents.agents import FunctionCallAgent
from veragents.core.llm import LLMClient
from veragents.tools import registry
import veragents.tools.builtin.search  # 注册工具
import veragents.tools.builtin.weather

load_dotenv(override=True)
agent = FunctionCallAgent("fc-demo", LLMClient(), tool_registry=registry)
print(agent.run("查询北京天气，并搜索最新 LLM 动态"))
```

## 依赖
- LLM 需为 OpenAI SDK 兼容模型（`LLMClient` 已封装 client）。
- 工具需注册到 `registry`，并支持 `export_openai_tools()`。

## 注意事项
- streaming 未实现，`stream_run` 回退为单次调用。
- `default_tool_choice` 默认 `"auto"`，可改为 `"none"` 强制不调用工具。
- 如需自定义工具 schema，可扩展 registry 的导出逻辑。
