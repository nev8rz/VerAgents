#!/usr/bin/env python3
"""异步工具执行器示例

演示如何使用 AsyncToolExecutor 进行并行和批量工具执行。
"""

import asyncio
from ver_agent.tools import ToolRegistry, tool, run_parallel_tools_sync, run_batch_tool_sync


# 定义一些示例工具
@tool
def calculator(expression: str) -> str:
    """简单计算器，支持基本数学运算

    Args:
        expression: 数学表达式，如 "2 + 2" 或 "3 * 4"

    Returns:
        计算结果字符串
    """
    try:
        # 安全地评估表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def reverse_text(text: str) -> str:
    """反转文本

    Args:
        text: 要反转的文本

    Returns:
        反转后的文本
    """
    return text[::-1]


@tool
def count_words(text: str) -> str:
    """统计文本中的单词数量

    Args:
        text: 要统计的文本

    Returns:
        单词数量
    """
    count = len(text.split())
    return f"Text '{text}' has {count} words"


def demo_parallel_execution():
    """演示并行执行多个不同工具"""
    print("\n" + "="*60)
    print("📋 演示 1: 并行执行多个不同工具")
    print("="*60)

    # 创建注册表并注册工具
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(reverse_text)
    registry.register(count_words)

    # 定义并行任务
    tasks = [
        {"tool_name": "calculator", "tool_args": {"expression": "2 + 2"}},
        {"tool_name": "reverse_text", "tool_args": {"text": "Hello"}},
        {"tool_name": "count_words", "tool_args": {"text": "Python is awesome"}},
        {"tool_name": "calculator", "tool_args": {"expression": "10 * 5"}},
        {"tool_name": "reverse_text", "tool_args": {"text": "Async Tool"}},
    ]

    # 并行执行
    results = run_parallel_tools_sync(registry, tasks, max_workers=3)

    # 显示结果
    print("\n📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        args = result.get("tool_args", {})
        print(f"{status_icon} [{result['tool_name']}] {args} => {result['result']}")


def demo_batch_execution():
    """演示批量执行同一个工具"""
    print("\n" + "="*60)
    print("📋 演示 2: 批量执行同一个工具")
    print("="*60)

    # 创建注册表并注册工具
    registry = ToolRegistry()
    registry.register(calculator)

    # 定义批量参数
    args_list = [
        {"expression": "2 + 2"},
        {"expression": "3 * 4"},
        {"expression": "10 / 2"},
        {"expression": "5 ** 2"},
        {"expression": "100 - 25"},
    ]

    # 批量执行
    results = run_batch_tool_sync(registry, "calculator", args_list, max_workers=3)

    # 显示结果
    print("\n📊 批量执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['result']}")


async def demo_async_usage():
    """演示异步使用方式"""
    print("\n" + "="*60)
    print("📋 演示 3: 异步上下文管理器用法")
    print("="*60)

    from ver_agent.tools import AsyncToolExecutor

    # 创建注册表并注册工具
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(reverse_text)

    # 使用上下文管理器
    async with AsyncToolExecutor(registry, max_workers=2) as executor:
        tasks = [
            {"tool_name": "calculator", "tool_args": {"expression": "7 * 8"}},
            {"tool_name": "reverse_text", "tool_args": {"text": "Async Python"}},
        ]
        results = await executor.execute_tools_parallel(tasks)

    print("\n📊 异步执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} [{result['tool_name']}] => {result['result']}")


async def demo_stream_execution():
    """演示流式执行"""
    print("\n" + "="*60)
    print("📋 演示 4: 流式执行（边执行边返回）")
    print("="*60)

    from ver_agent.tools import AsyncToolExecutor

    # 创建注册表并注册工具
    registry = ToolRegistry()
    registry.register(calculator)

    tasks = [
        {"tool_name": "calculator", "tool_args": {"expression": "1 + 1"}},
        {"tool_name": "calculator", "tool_args": {"expression": "2 + 2"}},
        {"tool_name": "calculator", "tool_args": {"expression": "3 + 3"}},
    ]

    print("\n流式执行结果:")
    async with AsyncToolExecutor(registry, max_workers=2) as executor:
        async for result in executor.execute_tools_stream(tasks):
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {result['result']}")


def main():
    """运行所有演示"""
    print("\n" + "="*60)
    print("🚀 VerAgent 异步工具执行器演示")
    print("="*60)

    # 演示 1: 并行执行
    demo_parallel_execution()

    # 演示 2: 批量执行
    demo_batch_execution()

    # 演示 3: 异步用法
    asyncio.run(demo_async_usage())

    # 演示 4: 流式执行
    asyncio.run(demo_stream_execution())

    print("\n" + "="*60)
    print("🎉 所有演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
