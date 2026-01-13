from __future__ import annotations

from dotenv import load_dotenv

from veragents.tools import ToolChain, ToolChainManager, registry, tool

# Load env for search providers
load_dotenv(override=True)

# Register builtin search tool
import veragents.tools.builtin.search  # noqa: F401


@tool(name="pretty_print")
def pretty_print(input: str) -> str:
    """简单打印并返回输入，演示链式第二步。"""
    print("=== Pretty Output ===")
    print(input)
    return input


def main():
    chain = ToolChain("search_then_pretty", "示例链：搜索后打印")
    # 注意：JSON模板里需要对外层花括号转义为双花括号，保留 {input} 变量占位
    chain.add_step("search_web", '{{"query":"{input}","num":2}}', output_key="search_result")
    chain.add_step("pretty_print", "{search_result}", output_key="final_result")

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    res = manager.execute_chain("search_then_pretty", "LLM 最新动态")
    print("Final result:", res)


if __name__ == "__main__":
    main()
