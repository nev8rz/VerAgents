"""Demo for ReActAgent using simple local tools."""

from __future__ import annotations

from dotenv import load_dotenv

from veragents.agents import ReActAgent
from veragents.core.llm import LLMClient
from veragents.tools import registry, tool

load_dotenv(override=True)


# Define simple tools for the demo
@tool
def double_number(n: int) -> int:
    """Return n*2."""
    return n * 2


@tool
def echo(input: str) -> str:
    """Echo input back."""
    return f"echo: {input}"


def main():
    llm = LLMClient()
    agent = ReActAgent(
        name="react-demo",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    question = "请用工具计算 double_number(21)，并把结果再 echo 一下后给出最终答案。"
    answer = agent.run(question)
    print("Agent answer:", answer)


if __name__ == "__main__":
    main()
