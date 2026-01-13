"""Demo for FunctionCallAgent using OpenAI function-calling style."""

from __future__ import annotations

from dotenv import load_dotenv

from veragents.agents import FunctionCallAgent
from veragents.core.llm import LLMClient
from veragents.tools import registry


def main():
    load_dotenv(override=True)

    # Register builtin tools (search + weather)
    import veragents.tools.builtin.search  # noqa: F401
    import veragents.tools.builtin.weather  # noqa: F401

    agent = FunctionCallAgent(
        name="fc-demo",
        llm=LLMClient(),
        tool_registry=registry,
        max_tool_iterations=3,
    )

    question = "请用工具告诉我：北京当前天气，并搜索下最新的 LLM 动态。"
    answer = agent.run(question)
    print("Final answer:\n", answer)


if __name__ == "__main__":
    main()
