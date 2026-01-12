

from ver_agent.core.llm import VerAgentLLM
import dotenv
dotenv.load_dotenv()


llm = VerAgentLLM()

# stream response
# for chunk in llm.think([{"role": "user", "content": "你好"}]):
#     print(chunk,end = "",flush=True)


# non-stream response
response = llm.invoke([{"role": "user", "content": "你好"}])
print(response)