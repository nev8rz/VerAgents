import os

import openai
from typing import Literal, Iterator, Optional, Any

from .exceptions import LLMAgentException

SUPPORTED_PROVIDERS = Literal[
    "openai",
    "intern",
    "zhipu",
    "local"
]


class VerAgentLLM:
    """
    参数加载策略：
        1. 从参数中获取provider，model，api_key，base_url等
        2. 从环境变量中获取这些信息
        3. 如果参数中没有提供，也没有从环境变量中获取到，抛出异常
    """

    def __init__(
        self,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        self.provider = (provider or os.environ.get("PROVIDER")).lower()
        self.model = model or os.environ.get(f"{self.provider.upper()}_MODEL") or os.environ.get("MODEL")
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY") or os.environ.get("API_KEY")
        self.base_url = base_url or os.environ.get(f"{self.provider.upper()}_BASE_URL") or os.environ.get("BASE_URL")

        if not self.api_key:
            raise LLMAgentException(f"api_key is required for {self.provider}")
        if not self.base_url:
            raise LLMAgentException(f"base_url is required for {self.provider}")

        if not self.model:
            raise LLMAgentException(f"model is required for {self.provider}")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        quiet: bool = False,
    ) -> Iterator[str]:
        """
        stream response

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            quiet: 是否静默模式（不输出 "thinking..." 消息）

        Returns:
            流式响应的迭代器
        """
        if not quiet:
            print(f"\n🤔 {self.model} is thinking...\n")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True
            )
            for chunk in response:
                # print(chunk,end="",flush=True)
                yield chunk.choices[0].delta.content or ""
        except Exception as e:
            raise LLMAgentException(f"❌ ERROR: {e}")

        # print(f"👍{self.model} is done thinking!")

    def invoke(
        self,
        message: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        调用 LLM 并返回完整响应（包含 tool_calls）

        Args:
            message: 消息列表
            temperature: 温度参数
            max_tokens: 最大 token 数
            tools: OpenAI 工具列表格式
            **kwargs: 其他参数

        Returns:
            包含 content 和 tool_calls 的响应字典
        """
        try:
            api_kwargs = {
                "model": self.model,
                "messages": message,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }

            # 添加 tools 参数（如果提供）
            if tools:
                api_kwargs["tools"] = tools

            # 添加其他 kwargs
            for k, v in kwargs.items():
                if k not in ['temperature', 'max_tokens', 'tools']:
                    api_kwargs[k] = v

            response = self._client.chat.completions.create(**api_kwargs)

            result = {
                "content": response.choices[0].message.content or "",
                "tool_calls": response.choices[0].message.tool_calls
            }

            return result
        except Exception as e:
            raise LLMAgentException(f"❌ ERROR: {e}")
