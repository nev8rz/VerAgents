"""Demo for PlanAndSolveAgent."""

from __future__ import annotations

from dotenv import load_dotenv

from veragents.agents import PlanAndSolveAgent
from veragents.core.llm import LLMClient


def main():
    load_dotenv(override=True)
    agent = PlanAndSolveAgent("planner-demo", LLMClient())
    question = "请用 3-4 个步骤讲清楚如何完成一次 10 公里的跑步训练计划。"
    answer = agent.run(question)
    print("Final answer:\n", answer)


if __name__ == "__main__":
    main()
