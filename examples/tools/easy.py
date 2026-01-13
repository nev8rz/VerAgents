"""Async tool demo: two local tools printing numbers to show interleaving."""

from __future__ import annotations

import asyncio

from veragents.tools import registry, tool


@tool(name="print_fast")
async def print_fast(prefix: str = "F", delay: float = 0.01, upto: int = 50) -> str:
    """快速打印数字，演示异步工具输出交错。"""
    for i in range(1, upto + 1):
        print(f"{prefix}:{i}", flush=True)
        await asyncio.sleep(delay)
    return "print_fast_done"


@tool(name="print_slow")
async def print_slow(prefix: str = "S", delay: float = 0.02, upto: int = 50) -> str:
    """慢速打印数字，演示异步工具输出交错。"""
    for i in range(1, upto + 1):
        print(f"{prefix}:{i}", flush=True)
        await asyncio.sleep(delay)
    return "print_slow_done"


async def main():
    print("== Async tool calls (interleaving print_fast / print_slow) ==")
    tasks = [
        registry.async_dispatch("print_fast", {"prefix": "A", "delay": 0.01, "upto": 30}),
        registry.async_dispatch("print_slow", {"prefix": "B", "delay": 0.02, "upto": 30}),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("\n== Tool results ==")
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(main())
