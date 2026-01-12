
import dotenv
from ver_agent.agents import ReActAgent
from ver_agent.tools import ToolRegistry, tool
from ver_agent.core import VerAgentLLM


# ========================
# 定义工具
# ========================

@tool
def calculator(expression: str) -> str:
    """数学计算器"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_weather(city: str) -> str:
    """获取天气信息"""
    weather_data = {
        "北京": "晴天，温度 25°C",
        "上海": "多云，温度 28°C",
        "深圳": "阴天，温度 30°C",
        "广州": "小雨，温度 27°C",
    }
    return weather_data.get(city, f"{city} 的天气信息暂不可用")


@tool
def search_knowledge(query: str) -> str:
    """知识库搜索"""
    knowledge = {
        "python": "Python 是一种高级编程语言，广泛用于 Web 开发、数据分析、人工智能等领域。",
        "react": "ReAct 是一种结合推理和行动的 AI 框架。",
        "openai": "OpenAI 是一家人工智能研究公司，开发了 GPT 系列等知名模型。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"关于 '{query}' 的知识暂未收录"


@tool
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# ========================
# 创建 Agent
# ========================

def create_agent(llm: VerAgentLLM) -> ReActAgent:
    registry = ToolRegistry()
    registry.register(calculator)
    registry.register(get_weather)
    registry.register(search_knowledge)
    registry.register(get_current_time)

    return ReActAgent(
        name="ReAct助手",
        llm=llm,
        tool_registry=registry,
        max_steps=10,
        verbose=False
    )


# ========================
# 主程序
# ========================

def main():
    dotenv.load_dotenv(override=True)

    print("=" * 60)
    print("🤖 ReAct Agent 示例")
    print("=" * 60)

    llm = VerAgentLLM()
    agent = create_agent(llm)

    # 示例1: 数学计算
    print("\n📌 示例1: 数学计算")
    print("❓ 问题: 计算 123 * 456 + 789")
    result = agent.run("计算 123 * 456 + 789")
    print(f"📝 结果: {result}")

    # 示例2: 天气查询
    print("\n📌 示例2: 天气查询")
    print("❓ 问题: 今天北京的天气怎么样？")
    agent.clear_history()
    result = agent.run("今天北京的天气怎么样？")
    print(f"📝 结果: {result}")

    # 示例3: 知识搜索
    print("\n📌 示例3: 知识搜索")
    print("❓ 问题: 请介绍一下 Python")
    agent.clear_history()
    result = agent.run("请介绍一下 Python")
    print(f"📝 结果: {result}")

    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)


def interactive_mode():
    """交互式模式"""
    dotenv.load_dotenv(override=True)

    print("=" * 60)
    print("🤖 ReAct Agent 交互式模式")
    print("=" * 60)
    print("输入 'quit' 退出 | 输入 'stream' 切换流式模式\n")

    llm = VerAgentLLM()
    agent = create_agent(llm)
    stream_mode = False

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break

            if user_input.lower() == 'stream':
                stream_mode = not stream_mode
                print(f"📡 流式模式: {'开启' if stream_mode else '关闭'}")
                continue

            if not user_input:
                continue

            agent.clear_history()

            if stream_mode:
                print("\n🤖 Agent:")
                for chunk in agent.run_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                result = agent.run(user_input)
                print(f"\n🤖 {result}")

        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        main()
