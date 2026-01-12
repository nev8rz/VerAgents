
import dotenv
from ver_agent.agents import 
from ver_agent.tools import global_registry
from ver_agent.core import VerAgentLLM
from ver_agent.tools.builtin import WeatherFetcher


global_registry.register(WeatherFetcher)

dotenv.load_dotenv(override=True)

llm = VerAgentLLM()
agent = PlannerSolverAgent(
    name="Plan Solver Test",
    llm=llm,
    tool_registry=global_registry,
    max_steps_per_task=10,
    # verbose=False
)
for i in agent.run_stream("对比一下上海，北京，深圳三个地方的天气，哪一个更适合出游"):
    print(i,end="",flush=True)
    # pass