
import dotenv
from ver_agent.agents import ReflectionAgent, ReActAgent
from ver_agent.tools import global_registry
from ver_agent.core import VerAgentLLM
from ver_agent.tools.builtin import WeatherFetcher


dotenv.load_dotenv(override=True)

# 注册工具
global_registry.register(WeatherFetcher)
print(global_registry.get_tools_description())

# 创建 LLM
llm = VerAgentLLM()

# 创建执行器（ReActAgent）
executor = ReActAgent(
    name="ReAct Executor",
    llm=llm,
    tool_registry=global_registry,
    max_steps=10,
    verbose=True
)

# 创建 Reflection Agent，包装 ReAct 执行器
agent = ReflectionAgent(
    name="Reflection Test",
    executor=executor,
    llm=llm,
    max_iterations=3,
    verbose=True
)

# 运行任务
for i in agent.run_stream("对比一下上海，北京，深圳三个地方的天气，哪一个更适合出游"):
    print(i, end="", flush=True)
