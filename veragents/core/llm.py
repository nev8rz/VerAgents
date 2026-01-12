"""统一的 LLM 客户端，支持流式与非流式调用。"""

import os
from typing import Generator, Iterable, List, Optional

from loguru import logger as log
from openai import OpenAI

from .exceptions import LLMException
from .messages import Message
from .config import Config


class LLMClient:
    """LLM 调用封装，基于环境变量选择模型提供商。"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Config] = None,
        timeout: int = 60,
    ):
        self.config = config or Config.from_env()
        resolved_provider = provider or os.getenv("PROVIDER") or self.config.default_provider
        if not resolved_provider:
            raise LLMException("Provider is required for LLMClient")

        self.provider = resolved_provider.lower()
        self.timeout = timeout
        self.base_url, self.api_key, env_model = self._load_provider_config(self.provider)
        self.model = model or env_model or self.config.default_model
        if not self.model:
            raise LLMException("Model is required for LLMClient")

        # Initialize OpenAI-compatible client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        log.info(
            "LLM client initialized | provider={} model={} base_url={}",
            self.provider,
            self.model,
            self.base_url,
        )

    def chat(
        self,
        messages: Iterable[Message],
        stream: bool = False,
        **extra_params,
    ) -> str | Generator[str, None, None]:
        """发送对话请求。

        :param messages: Message 列表。
        :param stream: 是否流式返回。
        :param extra_params: 透传给大模型的其他参数。
        """
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        # Apply defaults from config when caller has not specified.
        if "temperature" not in extra_params and self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if "max_tokens" not in extra_params and self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        payload.update(extra_params)

        log.info(
            "Sending LLM request | provider={} model={} stream={} messages={}",
            self.provider,
            self.model,
            stream,
            len(payload["messages"]),
        )

        try:
            resp = self.client.chat.completions.create(**payload)
        except Exception as exc:  # SDK surfaces network and API errors via exceptions
            log.exception("LLM request failed | provider={} model={}", self.provider, self.model)
            raise LLMException(f"LLM request failed: {exc}") from exc
        return self._handle_stream(resp) if stream else self._handle_response(resp)

    def _handle_response(self, resp) -> str:
        """处理非流式响应。"""
        try:
            content = resp.choices[0].message.content
            log.info("LLM response received (non-stream) | chars={}", len(content))
            return content
        except (AttributeError, IndexError, TypeError) as exc:
            log.error("Failed to parse LLM response: {}", exc)
            raise LLMException(f"Failed to parse LLM response: {exc}") from exc

    def _handle_stream(self, resp) -> Generator[str, None, None]:
        """处理流式响应，逐块产出内容。"""
        log.info("LLM streaming started")
        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content:
                yield reasoning_content
            content = getattr(delta, "content", None)
            if content:
                yield content
        log.info("LLM streaming finished")

    @staticmethod
    def _load_provider_config(provider: str) -> tuple[str, str, Optional[str]]:
        """根据 provider 读取基础配置。"""
        if provider == "zhipu":
            base_url = os.getenv("ZHIPU_BASE_URL")
            api_key = os.getenv("ZHIPU_API_KEY")
            model = os.getenv("ZHIPU_MODEL")
        elif provider == "intern":
            base_url = os.getenv("INTERN_BASE_URL")
            api_key = os.getenv("INTERN_API_KEY")
            model = os.getenv("INTERN_MODEL")
        elif provider == "aiping":
            base_url = os.getenv("AIPING_BASE_URL", "https://aiping.cn/api/v1")
            api_key = os.getenv("AIPING_API_KEY")
            model = os.getenv("AIPING_MODEL")
        elif provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL")
        else:
            raise LLMException(f"Unsupported provider: {provider!r}")

        missing = [name for name, value in [("base_url", base_url), ("api_key", api_key)] if not value]
        if missing:
            raise LLMException(f"Missing provider config: {', '.join(missing)}")

        return base_url, api_key, model


# 便捷函数
def chat(messages: List[Message], stream: bool = False, **extra_params):
    """使用默认环境配置直接调用。"""
    client = LLMClient()
    return client.chat(messages, stream=stream, **extra_params)

# 兼容命名，保持与 Agent 类型提示一致
VerAgentsLLM = LLMClient
