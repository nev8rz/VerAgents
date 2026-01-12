# VerAgents Tools System 使用文档

VerAgents Tools System 是一个轻量级、工程化的工具管理框架，专为 LLM Agent 设计。它支持自动注册、类型检查（Pydantic）、OpenAI Schema 导出以及灵活的子工具发现机制。

## 核心特性

- **自动注册**：使用 `@tool` 装饰器即可完成注册。
- **类型安全**：基于 Pydantic v2，自动从函数签名生成参数模型，支持运行时参数校验。
- **OpenAI 兼容**：一键导出 `function calling` 需要的 JSON Schema。
- **子工具套件**：支持将普通类（Class）作为工具箱（Toolkit），自动扫描并注册其中的方法。
- **同步/异步**：原生支持同步和异步调用。

## 快速开始

### 1. 定义简单工具 (Function Tool)

最简单的方式是直接在函数上使用 `@tool` 装饰器。

```python
from veragents.tools import tool, registry

@tool
def calculate_sum(a: int, b: int) -> int:
    """
    计算两个数的和。
    
    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b

# 调用
result = registry.dispatch("calculate_sum", {"a": 10, "b": 20})
print(result) # 30
```

### 2. 定义复杂参数工具

对于复杂结构，可以使用 Pydantic 模型作为参数注解。

```python
from typing import List
from pydantic import BaseModel
from veragents.tools import tool

class SearchParams(BaseModel):
    query: str
    tags: List[str] = []

@tool
def search(params: SearchParams):
    """搜索数据库"""
    return f"Searching {params.query} with tags {params.tags}"
```

### 3. 定义类工具 (Class Tool)

如果工具需要状态或继承 `BaseTool`，可以这样定义：

```python
from veragents.tools import tool, BaseTool
from pydantic import BaseModel

class WeatherArgs(BaseModel):
    city: str

@tool
class WeatherTool(BaseTool):
    name = "get_weather"
    description = "获取天气信息"
    args_model = WeatherArgs

    def run(self, city: str):
        return f"{city} 天气晴朗"
```

## 高级特性：工具箱与子工具 (Toolkit)

你可以将一组相关的工具组织在一个类中。系统会自动发现类中的公开方法并将其注册为独立的工具。

### 自动发现 (Implicit Discovery)

只需在类上使用 `@tool`，系统会自动实例化该类，并将所有公开方法（不以 `_` 开头）注册为工具。
工具名称格式默认为 `{类名}_{方法名}`。

```python
@tool
class MathHelper:
    """数学工具箱"""
    
    def add(self, a: int, b: int) -> int:
        """加法"""
        return a + b
        
    def sub(self, a: int, b: int) -> int:
        """减法"""
        return a - b

# 注册结果：
# - MathHelper_add
# - MathHelper_sub
```

### 注册配置好的实例

如果工具箱需要初始化参数（如数据库连接、文件路径），你可以先实例化，然后手动注册。

```python
class FileSystem:
    def __init__(self, root: str):
        self.root = root

    def read(self, path: str):
        """读取文件"""
        ...

# 实例化并注册
fs = FileSystem(root="/tmp")
# name参数指定前缀，注册为: fs_read
registry.register(fs, name="fs") 
```

## API 参考

### `veragents.tools.registry`

全局单例注册表。

- **`registry.register(obj, name=None)`**: 注册工具。支持函数、`BaseTool`、普通类或实例。
- **`registry.dispatch(name, args, strict=False)`**: 同步执行工具。
- **`registry.async_dispatch(name, args, strict=False)`**: 异步执行工具。
- **`registry.export_openai_tools()`**: 返回 OpenAI 格式的工具列表 (`list[dict]`)。
- **`registry.list_tools()`**: 列出所有注册的工具名称。

### `veragents.tools.tool`

装饰器，用于标记和注册工具。

## 导出给 LLM (OpenAI Schema)

直接将导出的 Schema 传给 OpenAI SDK。

```python
import json
from veragents.tools import registry

tools_schema = registry.export_openai_tools()
print(json.dumps(tools_schema, indent=2))

# 配合 OpenAI SDK 使用
# client.chat.completions.create(
#     model="gpt-4",
#     messages=[...],
#     tools=tools_schema
# )
```
