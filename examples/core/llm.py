from loguru import logger as log
from veragents.core.llm import LLMClient
from veragents.core.messages import Message
from dotenv import load_dotenv

load_dotenv(override=True)


llm_client = LLMClient()

questions = [
    "请简要介绍一下人工智能的发展历史。",
    "什么是机器学习？"
]


def main():
    """演示如何使用 LLMClient 进行对话。"""
    log.info("开始 LLM 示例对话")

    for i, question in enumerate(questions, 1):
        log.info("问题 {}: {}", i, question)

        # 创建用户消息
        message = Message(content=question, role="user")

        try:
            # 流式调用 LLM
            print(f"\n问题 {i}: {question}")
            print(f"回答 {i}: ", end="", flush=True)

            response_text = ""
            for chunk in llm_client.chat([message], stream=True):
                print(chunk, end="", flush=True)
                response_text += chunk

            print("\n")
            # log.info("回答 {}: {}", i, response_text)

        except Exception as e:
            log.exception("处理问题 {} 时出错: {}", i, e)
            print(f"处理问题 {i} 时出错: {e} ({type(e).__name__})")


if __name__ == "__main__":
    main()
