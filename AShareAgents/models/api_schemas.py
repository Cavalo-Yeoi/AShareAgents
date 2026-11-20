"""HTTP API 的 Pydantic 请求与响应模型。"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from AShareAgents.config import DEFAULT_CONFIG
from AShareAgents.datasource.astock.ticker_input import validate_ticker_input


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisConfig(BaseModel):
    """客户端可选择的分析选项。

    文件系统路径和其他服务端拥有的设置会被有意省略。
    """

    llm_provider: str = DEFAULT_CONFIG["llm_provider"]
    deep_think_llm: str = DEFAULT_CONFIG["deep_think_llm"]
    quick_think_llm: str = DEFAULT_CONFIG["quick_think_llm"]
    backend_url: str | None = DEFAULT_CONFIG["backend_url"]
    max_debate_rounds: int = Field(default=1, ge=1, le=10)
    max_risk_discuss_rounds: int = Field(default=1, ge=1, le=10)
    output_language: str = "Chinese"
    selected_analysts: list[Literal[
    "market",
    "social",
    "news",
    "fundamentals",
    "policy",
    "hot_money",
    "lockup",
]] | None = Field(default=None, min_length=1, max_length=7)

class AnalysisCreateRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=64)
    trade_date: date
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return validate_ticker_input(value)


class AnalysisCreatedResponse(BaseModel):
    task_id: str
    status: TaskStatus


class StageSnapshot(BaseModel):
    id: str
    status: str


class AnalysisStats(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    # Separate conservative permits from LangChain events and reported usage.
    llm_logical_calls: int = 0
    llm_attempts: int = 0
    llm_retries: int = 0
    llm_attempt_limit: int = 0
    llm_stop_reason: str | None = None
    sdk_max_retries: int = 0
    http_requests_observed: int | None = None
    cost_estimate: float | None = None


class AnalysisStatusResponse(BaseModel):
    task_id: str
    ticker: str
    trade_date: str
    status: TaskStatus
    current_stage: str = ""
    completed_stages: list[str] = Field(default_factory=list)
    stages: list[StageSnapshot] = Field(default_factory=list)
    stage_reports: dict[str, str] = Field(default_factory=dict)
    stats: AnalysisStats = Field(default_factory=AnalysisStats)
    elapsed: float = 0.0
    signal: str = ""
    error: str | None = None


class AnalysisResultResponse(BaseModel):
    task_id: str
    ticker: str
    trade_date: str
    signal: str
    elapsed: float
    final_state: dict[str, Any]


class HistoryEntry(BaseModel):
    id: str
    ticker: str
    date: str
    time: str


class HistoryResultResponse(BaseModel):
    id: str
    ticker: str
    trade_date: str
    signal: str
    final_state: dict[str, Any]


class TickerResolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=64)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return validate_ticker_input(value)


class TickerResolveResponse(BaseModel):
    ticker: str


class HealthResponse(BaseModel):
    status: str = "ok"
