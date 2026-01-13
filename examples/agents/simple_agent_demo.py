"""Demo for SimpleAgent with optional tool calling."""

from __future__ import annotations

from dotenv import load_dotenv

from veragents.agents.simple import SimpleAgent
from veragents.core.llm import LLMClient
from veragents.tools import registry

# Load environment for LLM provider/API keys
load_dotenv(override=True)

# Register builtin tools (optional)
import veragents.tools.builtin.weather  # noqa: F401
# import veragents.tools.builtin.search  # noqa: F401


def main():
    llm = LLMClient()
    agent = SimpleAgent(
        name="demo",
        llm=llm,
        tool_registry=registry,  # enable tool calling
    )

    user_input = "请务必用工具完成：告诉我上海实时天气。"
    response = agent.run(user_input, max_tool_calls=3)
    print("Agent:", response)

if __name__ == "__main__":
    main()
