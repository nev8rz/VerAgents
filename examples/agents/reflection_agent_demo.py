"""Demo for ReflectionAgent."""

from __future__ import annotations

from dotenv import load_dotenv

from veragents.agents import ReflectionAgent
from veragents.core.llm import LLMClient


def main():
    load_dotenv(override=True)
    agent = ReflectionAgent("reflect-demo", LLMClient(), max_iterations=3)
    question = "请写一段 100 字左右的新品发布介绍，并优化用词。"
    answer = agent.run(question)
    print("Final answer:\n", answer)


if __name__ == "__main__":
    main()
