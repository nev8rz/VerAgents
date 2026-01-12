from ver_agent.core import VerAgentLLM
from ver_agent.agents import ReActAgent

from ver_agent.tools.registry import global_registry
from ver_agent.tools.builtin.weather import WeatherFetcher
import dotenv
dotenv.load_dotenv(override=True)


global_registry.register(WeatherFetcher)



llm = VerAgentLLM()
agent = ReActAgent(
    name="ReAct助手",
    llm=llm,
    tool_registry=global_registry,
    max_steps=10,
    # verbose=False
)
for i in agent.run_stream("上海徐汇区的天气是什么"):
    print(i,end="",flush=True)
    # pass

