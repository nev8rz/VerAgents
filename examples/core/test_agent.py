from ver_agent.core import Agent,VerAgentLLM
import dotenv

dotenv.load_dotenv(override=True)

class VerAgent(Agent):
    def run(self,input:str,**kwargs) -> str:
        return self.llm.think([{"role": "user", "content": input}])

agent = VerAgent(
    name="test",
    llm=VerAgentLLM(),
    system_prompt="You are a helpful assistant."
)


for message in agent.run("你好"):
    print(message,end="",flush=True)