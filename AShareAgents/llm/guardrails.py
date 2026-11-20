"""Task-local conservative model-attempt limits, not a billing/HTTP counter.

Deadlines are cooperative. They gate new attempts and reject late returns;
they cannot terminate a Python thread already blocked in a provider call.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import logging
import random
import threading
import time
from typing import Any, Callable

import httpx
from langchain_core.runnables import Runnable

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


def _close_http_clients(clients):
    for client in clients:
        try:
            client.close()
        except Exception as exc:
            # Cleanup must neither mask task failures nor expose provider data.
            logger.warning("Owned LLM HTTP client cleanup failed (%s)", type(exc).__name__)


class LLMGuardError(RuntimeError):
    code = "llm_guard_closed"
    public_message = "模型运行保护已结束，不能继续发起请求"

    def __init__(self):
        super().__init__(self.public_message)


class LLMConfigurationError(LLMGuardError):
    code = "llm_guard_configuration"
    public_message = "模型运行保护配置无效，或当前供应商尚未验证支持"


class LLMBudgetExceeded(LLMGuardError):
    code = "llm_budget_exhausted"
    public_message = "本次分析的模型调用预算已耗尽，请减少分析范围后重试"


class LLMDeadlineExceeded(LLMGuardError):
    def __init__(self, scope: str):
        self.scope = scope
        self.code = "llm_" + scope + "_deadline"
        self.public_message = ("分析任务达到时间限制" if scope == "task" else "模型调用达到时间限制") + "，或剩余时间不足以重试"
        super().__init__()


_REQUEST_MESSAGES = {
    "connect_timeout": "连接模型服务超时，请稍后重试",
    "read_timeout": "读取模型响应超时，请稍后重试",
    "request_timeout": "模型请求超时，请稍后重试",
    "connection_failed": "暂时无法连接模型服务，请稍后重试",
    "authentication_failed": "模型服务鉴权失败，请检查本地凭据配置",
    "insufficient_balance": "模型服务账户余额不足，请检查账户状态",
    "permission_denied": "模型服务拒绝访问，请检查权限配置",
    "model_not_found": "模型或服务端点不存在，请检查配置",
    "invalid_request": "模型请求参数不受支持，请检查模型配置",
    "rate_limited": "模型服务请求受限，请稍后重试",
    "upstream_unavailable": "模型服务暂时不可用，请稍后重试",
    "request_failed": "模型请求失败，请检查配置或稍后重试",
}


class LLMRequestFailed(LLMGuardError):
    def __init__(self, code: str):
        self.code = code if code in _REQUEST_MESSAGES else "request_failed"
        self.public_message = _REQUEST_MESSAGES[self.code]
        super().__init__()


def _fresh_error(error: LLMGuardError) -> LLMGuardError:
    # Do not store/reuse raised exceptions (or their provider traceback) across threads.
    if isinstance(error, LLMDeadlineExceeded):
        return LLMDeadlineExceeded(error.scope)
    if isinstance(error, LLMRequestFailed):
        return LLMRequestFailed(error.code)
    return type(error)()


@dataclass(frozen=True)
class LLMPolicy:
    max_attempts: int = 64
    max_retries: int = 1
    connect_timeout: float = 5.0
    read_timeout: float = 60.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0
    call_timeout: float = 90.0
    task_timeout: float = 600.0
    retry_backoff: float = 0.5
    retry_backoff_max: float = 4.0

    def __post_init__(self):
        for name in ("max_attempts", "max_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "max_attempts" else 0):
                raise LLMConfigurationError()
        if self.max_retries > 3:
            raise LLMConfigurationError()
        for field in fields(self):
            if field.name in ("max_attempts", "max_retries"):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise LLMConfigurationError()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMPolicy":
        defaults = cls()
        return cls(**{field.name: config.get("llm_" + field.name, getattr(defaults, field.name))
                      for field in fields(cls)})

    def http_timeout(self, remaining: float | None = None) -> httpx.Timeout:
        cap = remaining if remaining is not None else math.inf
        return httpx.Timeout(connect=min(self.connect_timeout, cap), read=min(self.read_timeout, cap),
                             write=min(self.write_timeout, cap), pool=min(self.pool_timeout, cap))


class TaskBudget:
    """One explicit object shared by a task's models and bound Runnables.

    The observer is an internal, short snapshot-copy function; it runs under
    this lock. It must not perform I/O or acquire the budget from another thread.
    """
    def __init__(self, policy: LLMPolicy, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, jitter: Callable[[], float] | None = None,
                 started_at: float | None = None, on_change: Callable[[dict], None] | None = None):
        self.policy, self.clock, self.sleep = policy, clock, sleep
        self.jitter = jitter or (lambda: random.uniform(0.75, 1.25))
        self._deadline = (clock() if started_at is None else started_at) + policy.task_timeout
        self._lock = threading.RLock()
        self._logical_calls = self._attempts = self._retries = 0
        self._closed = False
        self._failure: LLMGuardError | None = None
        self._on_change = on_change
        self._owned_http_clients: dict[int, httpx.Client] = {}
        self._active_http_users = 0

    def _snapshot_locked(self) -> dict[str, Any]:
        return {"logical_calls": self._logical_calls, "attempts": self._attempts,
                "retries": self._retries, "attempt_limit": self.policy.max_attempts,
                "sdk_max_retries": 0, "http_requests_observed": None, "cost_estimate": None,
                "closed": self._closed, "stop_reason": self._failure.code if self._failure else None}

    def _publish_locked(self):
        if self._on_change:
            self._on_change(self._snapshot_locked())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _stop_locked(self, error: LLMGuardError) -> LLMGuardError:
        if self._failure is None:
            self._failure = _fresh_error(error)
        self._publish_locked()
        return _fresh_error(self._failure)

    def stop(self, error: LLMGuardError) -> LLMGuardError:
        with self._lock:
            return self._stop_locked(error)

    def _check_locked(self, call_deadline: float | None = None):
        if self._failure:
            raise _fresh_error(self._failure)
        if self._closed:
            raise LLMGuardError()
        now = self.clock()
        if now >= self._deadline:
            raise self._stop_locked(LLMDeadlineExceeded("task"))
        if call_deadline is not None and now >= call_deadline:
            raise self._stop_locked(LLMDeadlineExceeded("call"))

    def checkpoint(self, call_deadline: float | None = None):
        with self._lock:
            self._check_locked(call_deadline)

    def begin_call(self) -> float:
        with self._lock:
            self._check_locked()
            self._logical_calls += 1
            self._publish_locked()
            return self.clock() + self.policy.call_timeout

    def reserve(self, call_deadline: float, *, retry: bool = False):
        with self._lock:
            self._check_locked(call_deadline)
            if self._attempts >= self.policy.max_attempts:
                raise self._stop_locked(LLMBudgetExceeded())
            self._attempts += 1
            self._retries += int(retry)
            self._publish_locked()

    def remaining(self, call_deadline: float) -> float:
        with self._lock:
            self._check_locked(call_deadline)
            remaining = min(self._deadline, call_deadline) - self.clock()
            if remaining <= 0:
                scope = "task" if self._deadline <= call_deadline else "call"
                raise self._stop_locked(LLMDeadlineExceeded(scope))
            return remaining

    def wait_for_retry(self, call_deadline: float, delay: float):
        with self._lock:
            self._check_locked(call_deadline)
            if delay >= min(self._deadline, call_deadline) - self.clock():
                scope = "task" if self._deadline <= call_deadline else "call"
                raise self._stop_locked(LLMDeadlineExceeded(scope))
        self.sleep(delay)
        self.checkpoint(call_deadline)

    def own_http_client(self, client: httpx.Client):
        """Transfer factory-created client ownership; borrowed clients stay out."""
        with self._lock:
            if not self._closed:
                self._owned_http_clients.setdefault(id(client), client)
                return
            # A repeated registration must not close an already active client.
            pending = [] if id(client) in self._owned_http_clients else [client]
        _close_http_clients(pending)
        raise LLMGuardError()

    def _take_idle_clients_locked(self):
        if not self._closed or self._active_http_users:
            return []
        clients = list(self._owned_http_clients.values())
        self._owned_http_clients.clear()
        return clients

    @contextmanager
    def using_http_clients(self):
        """Protect a whole guarded invocation, including bindings and retries."""
        with self._lock:
            if self._closed:
                self._check_locked()  # Preserve an earlier terminal failure category.
            self._active_http_users += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_http_users -= 1
                pending = self._take_idle_clients_locked()
            _close_http_clients(pending)

    def close(self):
        """Close permits now; release owned pools once active invokes unwind."""
        with self._lock:
            self._closed = True
            self._publish_locked()
            pending = self._take_idle_clients_locked()
        _close_http_clients(pending)


def _classify_failure(exc: Exception) -> tuple[LLMRequestFailed, bool] | None:
    import openai

    # A concrete response outranks an older exception context. Never classify a
    # programming/parser error as retryable just because it has a network cause.
    if isinstance(exc, (openai.APIStatusError, httpx.HTTPStatusError)):
        status = exc.status_code if isinstance(exc, openai.APIStatusError) else exc.response.status_code
        code = {400: "invalid_request", 401: "authentication_failed", 402: "insufficient_balance",
                403: "permission_denied", 404: "model_not_found", 422: "invalid_request",
                408: "request_timeout", 429: "rate_limited"}.get(status, "upstream_unavailable" if status >= 500 else "request_failed")
        return LLMRequestFailed(code), status in {408, 429, 500, 502, 503, 504}
    if not isinstance(exc, (openai.APIConnectionError, httpx.TimeoutException, httpx.ConnectError,
                            httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return None
    cause: BaseException | None = exc
    seen = set()
    timeout_code = None
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if type(cause).__module__.startswith("httpx"):
            if isinstance(cause, httpx.ConnectTimeout) or type(cause).__name__ == "ConnectTimeout":
                timeout_code = "connect_timeout"
            elif isinstance(cause, httpx.ReadTimeout) or type(cause).__name__ == "ReadTimeout":
                timeout_code = "read_timeout"
        cause = cause.__cause__ or cause.__context__
    if timeout_code:
        return LLMRequestFailed(timeout_code), True
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return LLMRequestFailed("request_timeout"), True
    if isinstance(exc, (openai.APIConnectionError, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return LLMRequestFailed("connection_failed"), True
    return None


def _retry_delay(exc: Exception, runtime: TaskBudget, retry_index: int) -> float | None:
    policy = runtime.policy
    delay = min(policy.retry_backoff_max, policy.retry_backoff * 2 ** retry_index * runtime.jitter())
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    if value is not None:
        try:
            try:
                seconds = float(value)
            except ValueError:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            if not math.isfinite(seconds) or seconds < 0 or seconds > policy.retry_backoff_max:
                return None
        except (ValueError, TypeError, OverflowError):
            return None
        delay = max(delay, seconds)
    return delay


class GuardedModel(Runnable[Any, Any]):
    """Current workflow model interface; async/batch retain this sync gate.

    Inherited stream/astream yield the full guarded response, not native token
    streaming. Native provider generate/stream clients are not delegated.
    """
    _PROTECTED = {"timeout", "request_timeout", "max_retries", "http_client", "http_async_client",
                  "client", "root_client", "stream", "streaming"}

    def __init__(self, target: Runnable, runtime: TaskBudget, *, metadata: Any = None):
        self._target, self.runtime = target, runtime
        self._metadata = metadata if metadata is not None else target

    @property
    def model_name(self):
        return getattr(self._metadata, "model_name", getattr(self._metadata, "model", ""))

    @property
    def openai_api_base(self):
        return getattr(self._metadata, "openai_api_base", None)

    def bind_tools(self, tools, **kwargs):
        if self._PROTECTED.intersection(kwargs):
            raise LLMConfigurationError()
        return GuardedModel(self._target.bind_tools(tools, **kwargs), self.runtime, metadata=self._metadata)

    def with_structured_output(self, schema, **kwargs):
        if self._PROTECTED.intersection(kwargs):
            raise LLMConfigurationError()
        return GuardedModel(self._target.with_structured_output(schema, **kwargs), self.runtime, metadata=self._metadata)

    def invoke(self, input, config=None, **kwargs):
        with self.runtime.using_http_clients():
            return self._invoke(input, config=config, **kwargs)

    def _invoke(self, input, config=None, **kwargs):
        if self._PROTECTED.intersection(kwargs):
            raise LLMConfigurationError()
        deadline = self.runtime.begin_call()
        for index in range(self.runtime.policy.max_retries + 1):
            self.runtime.reserve(deadline, retry=index > 0)
            timeout = self.runtime.policy.http_timeout(self.runtime.remaining(deadline))
            try:
                result = self._target.invoke(input, config=config, timeout=timeout, **kwargs)
            except LLMGuardError:
                raise
            except Exception as exc:
                self.runtime.checkpoint(deadline)
                classified = _classify_failure(exc)
                if classified is None:
                    raise
                safe, retryable = classified
                if retryable and index < self.runtime.policy.max_retries:
                    delay = _retry_delay(exc, self.runtime, index)
                    if delay is not None:
                        self.runtime.wait_for_retry(deadline, delay)
                        continue
                raise self.runtime.stop(safe) from exc
            self.runtime.checkpoint(deadline)
            return result
        raise AssertionError("Unreachable retry loop")


class GuardedClient(BaseLLMClient):
    def __init__(self, client: BaseLLMClient, runtime: TaskBudget):
        super().__init__(client.model, client.base_url)
        self._client, self.runtime = client, runtime
        self.provider = client.get_provider_name()

    def get_llm(self) -> GuardedModel:
        try:
            target = self._client.get_llm()
            for name in ("root_client", "root_async_client"):
                sdk = getattr(target, name, None)
                if sdk is not None and sdk.max_retries != 0:
                    raise LLMConfigurationError()
            return GuardedModel(target, self.runtime)
        except BaseException:
            self.close()
            raise

    def close(self):
        """End this task-scoped runtime; never close borrowed HTTP clients."""
        self.runtime.close()

    def validate_model(self) -> bool:
        return self._client.validate_model()
