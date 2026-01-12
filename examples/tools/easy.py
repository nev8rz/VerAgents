from typing import List, Optional

from veragents.tools import tool, registry


@tool
def calculate_sum(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b


@tool
def search(query: str, tags: Optional[List[str]] = None) -> str:
    """搜索数据库"""
    tags = tags or []
    return f"Searching {query} with tags {tags}"


if __name__ == "__main__":
    print(registry.dispatch("calculate_sum", {"a": 10, "b": 20}))
    print(registry.dispatch("search", {"query": "LLM", "tags": ["python", "ai"]}))
