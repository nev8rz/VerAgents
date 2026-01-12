"""异步工具执行器 - VerAgent 异步工具执行支持

提供异步和并行执行工具的能力，支持批量操作。
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器

    支持异步执行工具，实现并行调用和批量操作。

    Attributes:
        registry: 工具注册表
        max_workers: 最大工作线程数
        executor: 线程池执行器
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        """
        初始化异步工具执行器

        Args:
            registry: 工具注册表
            max_workers: 最大工作线程数
        """
        self.registry = registry
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        异步执行单个工具

        Args:
            tool_name: 工具名称
            tool_args: 工具参数字典

        Returns:
            工具执行结果
        """
        loop = asyncio.get_event_loop()

        def _execute():
            return self.registry.execute(tool_name, tool_args)

        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"Error executing '{tool_name}': {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并行执行多个工具

        Args:
            tasks: 任务列表，每个任务包含 tool_name 和 tool_args

        Returns:
            执行结果列表，包含任务信息和结果
        """
        print(f"\n🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            tool_args = task.get("tool_args")

            if not tool_name:
                continue

            print(f"📝 创建任务 {i + 1}: {tool_name}")
            async_task = self.execute_tool_async(tool_name, tool_args)
            async_tasks.append((i, task, async_task))

        # 等待所有任务完成
        results = []
        for i, task, async_task in async_tasks:
            try:
                result = await async_task
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "tool_args": task.get("tool_args", {}),
                    "result": result,
                    "status": "success"
                })
                print(f"✅ 任务 {i + 1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "tool_args": task.get("tool_args", {}),
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ 任务 {i + 1} 失败: {task['tool_name']} - {e}")

        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"🎉 并行执行完成，成功: {success_count}/{len(results)}\n")
        return results

    async def execute_tools_batch(
        self,
        tool_name: str,
        args_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量执行同一个工具

        Args:
            tool_name: 工具名称
            args_list: 参数列表

        Returns:
            执行结果列表
        """
        tasks = [
            {"tool_name": tool_name, "tool_args": args}
            for args in args_list
        ]
        return await self.execute_tools_parallel(tasks)

    async def execute_tools_stream(self, tasks: List[Dict[str, Any]]) -> Any:
        """
        流式执行工具（边执行边返回结果）

        Args:
            tasks: 任务列表

        Yields:
            执行结果
        """
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            tool_args = task.get("tool_args")

            if not tool_name:
                continue

            print(f"📝 执行任务 {i + 1}: {tool_name}")

            try:
                result = await self.execute_tool_async(tool_name, tool_args)
                yield {
                    "task_id": i,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result,
                    "status": "success"
                }
                print(f"✅ 任务 {i + 1} 完成: {tool_name}")
            except Exception as e:
                yield {
                    "task_id": i,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": str(e),
                    "status": "error"
                }
                print(f"❌ 任务 {i + 1} 失败: {tool_name} - {e}")

    def close(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
async def run_parallel_tools(
    registry: ToolRegistry,
    tasks: List[Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    便捷函数：并行执行多个工具

    Args:
        registry: 工具注册表
        tasks: 任务列表，每个任务包含 tool_name 和 tool_args
        max_workers: 最大工作线程数

    Returns:
        执行结果列表

    Example:
        tasks = [
            {"tool_name": "calculator", "tool_args": {"expression": "2 + 2"}},
            {"tool_name": "calculator", "tool_args": {"expression": "3 * 4"}},
        ]
        results = await run_parallel_tools(registry, tasks)
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(
    registry: ToolRegistry,
    tool_name: str,
    args_list: List[Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    便捷函数：批量执行同一个工具

    Args:
        registry: 工具注册表
        tool_name: 工具名称
        args_list: 参数列表
        max_workers: 最大工作线程数

    Returns:
        执行结果列表

    Example:
        args_list = [
            {"expression": "2 + 2"},
            {"expression": "3 * 4"},
            {"expression": "10 / 2"},
        ]
        results = await run_batch_tool(registry, "calculator", args_list)
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, args_list)


async def run_tools_stream(
    registry: ToolRegistry,
    tasks: List[Dict[str, Any]],
    max_workers: int = 4
) -> Any:
    """
    便捷函数：流式执行工具（边执行边返回）

    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数

    Yields:
        执行结果

    Example:
        async for result in run_tools_stream(registry, tasks):
            print(result)
    """
    async with AsyncToolExecutor(registry, max_workers) as executor:
        async for result in executor.execute_tools_stream(tasks):
            yield result


# 同步包装函数（为了兼容性）
def run_parallel_tools_sync(
    registry: ToolRegistry,
    tasks: List[Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    同步版本的并行工具执行

    Args:
        registry: 工具注册表
        tasks: 任务列表
        max_workers: 最大工作线程数

    Returns:
        执行结果列表
    """
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(
    registry: ToolRegistry,
    tool_name: str,
    args_list: List[Dict[str, Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]:
    """
    同步版本的批量工具执行

    Args:
        registry: 工具注册表
        tool_name: 工具名称
        args_list: 参数列表
        max_workers: 最大工作线程数

    Returns:
        执行结果列表
    """
    return asyncio.run(run_batch_tool(registry, tool_name, args_list, max_workers))


# 示例函数
async def demo_parallel_execution():
    """演示并行执行的示例"""
    from . import global_registry

    # 定义并行任务（示例：计算器）
    tasks = [
        {"tool_name": "calculator", "tool_args": {"expression": "2 + 2"}},
        {"tool_name": "calculator", "tool_args": {"expression": "3 * 4"}},
        {"tool_name": "calculator", "tool_args": {"expression": "10 / 2"}},
        {"tool_name": "calculator", "tool_args": {"expression": "sqrt(16)"}},
    ]

    # 并行执行
    results = await run_parallel_tools(global_registry, tasks)

    # 显示结果
    print("📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['tool_name']}({result.get('tool_args', {})}) = {result['result']}")

    return results


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo_parallel_execution())
