"""根据供应商名称创建统一的 LLM 客户端。"""

from dataclasses import replace
import math
from typing import Optional

from .base_client import BaseLLMClient

# 使用 OpenAI 兼容 chat completions API 的供应商
_OPENAI_COMPATIBLE = (
    "openai", "xai", "deepseek", "qwen", "glm", "ollama", "openrouter", "minimax",
)


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    runtime=None,
    **kwargs,
) -> BaseLLMClient:
    """为指定供应商创建 LLM 客户端。

    供应商模块采用懒加载方式导入，这样仅在导入此工厂函数时（例如在测试
    收集期间）不会拉入重量级的 LLM SDK，也不会因缺少 API Key 而失败。

    Args:
        provider: LLM 供应商名称
        model: 模型名称/标识符
        base_url: API 端点的可选基础 URL
        runtime: 同一任务共享的 TaskBudget；省略时为此客户端创建独立预算
        **kwargs: 供应商特定的额外参数

    Returns:
        get_llm() 返回受预算保护 Runnable 的客户端。当前验证了 OpenAI
        兼容协议和 Azure；其他供应商在发送请求前明确拒绝。

    Raises:
        LLMConfigurationError: 如果供应商或保护参数未经支持/验证
    """
    from AShareAgents.config import DEFAULT_CONFIG
    from .guardrails import GuardedClient, LLMConfigurationError, LLMPolicy, TaskBudget
    import httpx

    provider_lower = provider.lower()
    if provider_lower not in (*_OPENAI_COMPATIBLE, "azure"):
        # Do not silently run a provider whose SDK controls were not verified.
        raise LLMConfigurationError()
    if kwargs.get("max_retries", 0) != 0:
        raise LLMConfigurationError()
    if runtime is None:
        policy = LLMPolicy.from_config(DEFAULT_CONFIG)
        # Preserve explicit standalone no-retry/probe timeouts conservatively.
        if "max_retries" in kwargs:
            policy = replace(policy, max_retries=0)
        if "timeout" in kwargs:
            value = kwargs["timeout"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise LLMConfigurationError()
            policy = replace(policy, connect_timeout=min(policy.connect_timeout, value),
                             read_timeout=min(policy.read_timeout, value),
                             write_timeout=min(policy.write_timeout, value),
                             pool_timeout=min(policy.pool_timeout, value))
        runtime = TaskBudget(policy)
    elif "timeout" in kwargs:
        raise LLMConfigurationError()
    kwargs["max_retries"] = 0
    kwargs["timeout"] = runtime.policy.http_timeout()
    try:
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.Client(timeout=kwargs["timeout"], follow_redirects=False)
            runtime.own_http_client(kwargs["http_client"])
        elif not isinstance(kwargs["http_client"], httpx.Client) or kwargs["http_client"].follow_redirects:
            raise LLMConfigurationError()

        if provider_lower in _OPENAI_COMPATIBLE:
            from .openai_client import OpenAIClient
            return GuardedClient(OpenAIClient(model, base_url, provider=provider_lower, **kwargs), runtime)

        if provider_lower == "azure":
            from .azure_client import AzureOpenAIClient
            return GuardedClient(AzureOpenAIClient(model, base_url, **kwargs), runtime)
    except BaseException:
        runtime.close()
        raise

    raise ValueError(f"不支持的LLM供应商：{provider}")
