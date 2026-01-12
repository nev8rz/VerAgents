# VerAgents

轻量的 Agent/LLM 调用与工具系统样板，内置：
- 统一的 LLM 客户端（OpenAI SDK 兼容，支持流式与非流式）
- Agent 基类与配置管理（环境变量加载）
- 基于装饰器的工具系统，自动注册 / 参数校验（Pydantic v2）/ OpenAI tools schema 导出
- 示例：核心 LLM 调用、工具注册与调用、开放天气工具（open-meteo）

## 快速开始
```bash
# 可选：创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

环境变量示例（.env）：
```env
PROVIDER=aiping
AIPING_API_KEY=...
AIPING_BASE_URL=https://aiping.cn/api/v1
AIPING_MODEL=deepSeek-V3.2
```

## 运行示例
- LLM 流式对话：`python examples/core/llm.py`
- 工具示例（自动参数建模）：`python examples/tools/easy.py`
- 工具系统演示（含 schema 导出、异步调用等）：`python examples/tools.py`

## 工具系统用法
注册函数工具（自动从签名生成参数模型）：
```python
from veragents.tools import tool, registry

@tool
def search(query: str, tags: list[str] | None = None) -> list[str]:
    return [f"{query} - {t}" for t in tags or []]

result = registry.dispatch("search", {"query": "llm", "tags": ["python"]})
```

导出 OpenAI tools schema：
```python
from veragents.tools import registry
schemas = registry.export_openai_tools()
```

## builtin 工具集（示例）
开放天气（基于 open-meteo，无需密钥）：
```python
import veragents.tools.builtin.weather  # 触发注册
from veragents.tools import registry

res = registry.dispatch("get_current_weather", {"city": "Beijing"})
print(res)
```

## 目录
- `veragents/core`：LLM 客户端、Agent、配置
- `veragents/tools`：工具基类、注册表、装饰器
- `veragents/tools/builtin`：示例工具（天气）
- `examples/`：LLM 与工具使用示例
