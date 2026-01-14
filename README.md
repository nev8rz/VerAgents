# VerAgents

<div align="center">

**轻量级、可扩展的 LLM Agent 框架**

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

提供统一的 LLM 客户端、多种 Agent 实现、强大的工具系统和记忆管理

</div>

---

## ✨ 特性

### 🎯 核心功能

- **统一 LLM 客户端**
  - OpenAI SDK 兼容接口，支持流式与非流式调用
  - 多提供商支持（智谱 AI、InternLM、爱平 AI、OpenAI 等）
  - 环境变量配置管理
  - 自动重试与错误处理

- **多种 Agent 实现**
  - `SimpleAgent` - 基础对话与工具调用
  - `ReActAgent` - 推理+行动范式（Reasoning + Acting）
  - `ReflectionAgent` - 自我反思与改进
  - `PlanAndSolveAgent` - 规划与执行分离
  - `FunctionCallAgent` - OpenAI Function Calling 模式

- **强大的工具系统**
  - 基于 `@tool` 装饰器的自动注册
  - Pydantic v2 参数校验，从函数签名自动生成参数模型
  - OpenAI Tools Schema 一键导出
  - 支持函数工具和类工具（Toolkit）
  - 同步/异步调用支持

- **记忆管理系统**
  - 多种记忆类型：工作记忆、情节记忆、语义记忆、感知记忆
  - 向量嵌入服务（支持多种嵌入模型）
  - RAG（检索增强生成）支持
  - 灵活的存储后端（Qdrant 等）

## 📦 安装

### 环境要求

- Python 3.13+

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/VerAgents.git
cd VerAgents

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

## ⚙️ 配置

### 环境变量配置

创建 `.env` 文件并配置以下变量：

```env
# 选择提供商：zhipu | intern | aiping | openai
PROVIDER=aiping

# 爱平 AI 配置
AIPING_API_KEY=your_api_key_here
AIPING_BASE_URL=https://aiping.cn/api/v1
AIPING_MODEL=deepSeek-V3.2

# 智谱 AI 配置
# ZHIPU_API_KEY=your_api_key
# ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# ZHIPU_MODEL=glm-4

# InternLM 配置
# INTERN_API_KEY=your_api_key
# INTERN_BASE_URL=https://api.intern-ai.org.cn/paas/v4
# INTERN_MODEL=internlm2_5-20b-chat

# OpenAI 配置
# OPENAI_API_KEY=your_api_key
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4

# 可选：工具 API 密钥
SERPAPI_API_KEY=your_serpapi_key  # 用于 Web 搜索
TVLY_API_KEY=your_tavily_key      # Tavily 搜索 API
```

## 🚀 快速开始

### 基础 LLM 调用

```python
from veragents.core.llm import LLMClient
from veragents.core.messages import Message

# 初始化客户端
llm = LLMClient(provider="aiping")

# 非流式调用
messages = [Message.user("什么是 AI？")]
response = llm.chat(messages)
print(response)

# 流式调用
for chunk in llm.chat(messages, stream=True):
    print(chunk, end="", flush=True)
```

### 使用 SimpleAgent

```python
from veragents.agents import SimpleAgent
from veragents.core.llm import LLMClient
from veragents.tools import tool, registry

# 定义并注册工具
@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"结果: {result}"
    except Exception as e:
        return f"错误: {e}"

# 创建 Agent
llm = LLMClient()
agent = SimpleAgent("calculator", llm, tool_registry=registry)

# 运行
response = agent.run("帮我计算 123 * 456")
print(response)
```

### 使用 ReActAgent

```python
from veragents.agents import ReActAgent
from veragents.core.llm import LLMClient
from veragents.tools import tool, registry

# 注册工具
@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    # 实现搜索逻辑
    return f"关于 '{query}' 的搜索结果..."

# 创建 ReAct Agent
llm = LLMClient()
agent = ReActAgent(
    "researcher",
    llm,
    tool_registry=registry,
    max_steps=5  # 最大推理步数
)

# 运行
response = agent.run("搜索最新的 AI 发展趋势")
print(response)
```

### 工具系统使用

```python
from veragents.tools import tool, registry

# 定义简单工具
@tool
def greet(name: str, greeting: str = "Hello") -> str:
    """向某人打招呼"""
    return f"{greeting}, {name}!"

# 调用工具
result = registry.dispatch("greet", {"name": "世界", "greeting": "你好"})
print(result)  # 输出: 你好, 世界!

# 导出 OpenAI Schema
import json
schemas = registry.export_openai_tools()
print(json.dumps(schemas, indent=2, ensure_ascii=False))
```

### 定义类工具（Toolkit）

```python
from veragents.tools import tool

@tool
class FileSystem:
    """文件操作工具箱"""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def read(self, path: str) -> str:
        """读取文件内容"""
        with open(f"{self.root_dir}/{path}", "r") as f:
            return f.read()

    def write(self, path: str, content: str) -> str:
        """写入文件"""
        with open(f"{self.root_dir}/{path}", "w") as f:
            f.write(content)
        return f"已写入 {path}"

# 自动注册为：FileSystem_read 和 FileSystem_write
```

## 📚 文档

### Agent 类型对比

| Agent | 特点 | 适用场景 |
|-------|------|---------|
| `SimpleAgent` | 基础对话 + 工具调用 | 简单任务、快速原型 |
| `ReActAgent` | 推理-行动循环 | 需要多步推理的复杂任务 |
| `ReflectionAgent` | 自我反思与改进 | 需要优化输出的任务 |
| `PlanAndSolveAgent` | 规划-执行分离 | 复杂问题求解 |
| `FunctionCallAgent` | OpenAI Function Calling | 与 OpenAI API 深度集成 |

详细文档：
- [ReAct Agent 使用说明](docs/react_agent.md)
- [Reflection Agent 使用说明](docs/reflection_agent.md)
- [Plan and Solve Agent 使用说明](docs/plan_and_solve_agent.md)
- [Function Call Agent 使用说明](docs/function_call_agent.md)
- [工具系统详细文档](docs/tools_system.md)

### 示例代码

```bash
# 运行各种 Agent 示例
python examples/agents/simple_agent_demo.py
python examples/agents/react_agent_demo.py
python examples/agents/reflection_agent_demo.py
python examples/agents/plan_and_solve_agent_demo.py
python examples/agents/function_call_agent_demo.py
```

## 🧰 内置工具

### 天气工具（无需 API Key）

```python
import veragents.tools.builtin.weather
from veragents.tools import registry

# 获取当前天气
result = registry.dispatch("get_current_weather", {"city": "北京"})
print(result)
```

### Web 搜索工具

需要配置 `SERPAPI_API_KEY` 或 `TVLY_API_KEY`：

```python
import veragents.tools.builtin.search
from veragents.tools import registry

# 搜索网络
result = registry.dispatch("search_web", {
    "query": "Python AI 框架",
    "num_results": 5
})
print(result)
```

## 🧠 记忆系统

VerAgents 提供完整的记忆管理系统：

### 记忆类型

- **工作记忆（Working Memory）**：临时存储当前上下文信息
- **情节记忆（Episodic Memory）**：存储过去的事件和经验
- **语义记忆（Semantic Memory）**：存储知识和事实
- **感知记忆（Perceptual Memory）**：存储原始感知数据

### 使用示例

```python
from veragents.memory import MemoryManager, MemoryConfig
from veragents.memory.types import WorkingMemory

# 初始化记忆管理器
config = MemoryConfig()
memory_manager = MemoryManager(config)

# 添加记忆
memory_manager.add_memory(
    memory_type="working",
    content="用户询问了关于 AI 的问题",
    metadata={"timestamp": "2024-01-01"}
)

# 检索记忆
memories = memory_manager.retrieve_memories(
    memory_type="working",
    query="AI",
    limit=5
)
```

## 🏗️ 项目结构

```
VerAgents/
├── veragents/
│   ├── core/               # 核心模块
│   │   ├── agent.py        # Agent 基类
│   │   ├── llm.py          # LLM 客户端
│   │   ├── config.py       # 配置管理
│   │   ├── messages.py     # 消息模型
│   │   └── prompts.py      # 提示词模板
│   ├── agents/             # Agent 实现
│   │   ├── simple.py       # 简单 Agent
│   │   ├── react.py        # ReAct Agent
│   │   ├── reflection.py   # 反思 Agent
│   │   ├── plan_and_solve.py  # 规划求解 Agent
│   │   └── function_call.py   # Function Call Agent
│   ├── tools/              # 工具系统
│   │   ├── registry.py     # 工具注册表
│   │   ├── decorators.py   # 装饰器
│   │   ├── base.py         # 工具基类
│   │   └── builtin/        # 内置工具
│   │       ├── weather.py  # 天气工具
│   │       └── search.py   # 搜索工具
│   └── memory/             # 记忆系统
│       ├── manager.py      # 记忆管理器
│       ├── embedding.py    # 嵌入服务
│       ├── types/          # 记忆类型
│       ├── storage/        # 存储后端
│       └── rag/            # RAG 模块
├── examples/               # 示例代码
│   ├── agents/            # Agent 示例
│   ├── core/              # 核心功能示例
│   └── tools/             # 工具示例
├── tests/                  # 测试文件
├── docs/                   # 文档
├── main.py
├── pyproject.toml
└── README.md
```

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/sdk_embedding_test.py

# 查看测试覆盖率
pytest --cov=veragents tests/
```

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发建议

- 遵循 PEP 8 代码规范
- 添加类型注解
- 编写单元测试
- 更新相关文档

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [OpenAI](https://openai.com/) - OpenAI SDK 和 API
- [Pydantic](https://pydantic-docs.helpmanual.io/) - 数据验证
- [Loguru](https://github.com/Delgan/loguru) - 日志记录
- [Qdrant](https://qdrant.tech/) - 向量数据库

## 📮 联系方式

- 项目主页：[https://github.com/yourusername/VerAgents](https://github.com/yourusername/VerAgents)
- 问题反馈：[Issues](https://github.com/yourusername/VerAgents/issues)
- 讨论区：[Discussions](https://github.com/yourusername/VerAgents/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐️ Star！**

Made with ❤️ by VerAgents Team

</div>
