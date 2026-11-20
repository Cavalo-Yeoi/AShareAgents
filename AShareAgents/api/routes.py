"""分析任务和已保存报告的 FastAPI 路由。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from AShareAgents.api.history import extract_signal, list_history, load_history
from AShareAgents.api.jobs import registry
from AShareAgents.api.service import start_analysis
from AShareAgents.config import DEFAULT_CONFIG
from AShareAgents.datasource.astock.errors import DataSourceUnavailableError
from AShareAgents.models.api_schemas import (
    AnalysisCreateRequest,
    AnalysisCreatedResponse,
    AnalysisResultResponse,
    AnalysisStatusResponse,
    HealthResponse,
    HistoryEntry,
    HistoryResultResponse,
    TaskStatus,
    TickerResolveRequest,
    TickerResolveResponse,
)

router = APIRouter(prefix="/api/v1")


def _resolve_stock_input(value: str) -> str:
    from AShareAgents.datasource.astock.a_stock import resolve_ticker as resolve

    try:
        return resolve(value)
    except DataSourceUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Stock name service temporarily unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/tickers/resolve",
    response_model=TickerResolveResponse,
)
def resolve_ticker(request: TickerResolveRequest) -> TickerResolveResponse:
    return TickerResolveResponse(ticker=_resolve_stock_input(request.query))


@router.post(
    "/analyses",
    response_model=AnalysisCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_analysis(request: AnalysisCreateRequest) -> AnalysisCreatedResponse:
    ticker = _resolve_stock_input(request.ticker)
    config = DEFAULT_CONFIG.copy()
    config.update(request.config.model_dump())
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }
    job = registry.create(ticker=ticker, trade_date=request.trade_date.isoformat())
    start_analysis(job, config)
    return AnalysisCreatedResponse(task_id=job.task_id, status=job.snapshot().status)


@router.get("/analyses/{task_id}", response_model=AnalysisStatusResponse)
def get_analysis(task_id: str) -> AnalysisStatusResponse:
    job = registry.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    return job.snapshot()


@router.get("/analyses/{task_id}/result", response_model=AnalysisResultResponse)
def get_analysis_result(task_id: str) -> AnalysisResultResponse:
    job = registry.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    snapshot = job.snapshot()
    if snapshot.status == TaskStatus.FAILED:
        raise HTTPException(status_code=409, detail=snapshot.error or "Analysis failed")
    if snapshot.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Analysis is not complete")
    return AnalysisResultResponse(
        task_id=snapshot.task_id,
        ticker=snapshot.ticker,
        trade_date=snapshot.trade_date,
        signal=snapshot.signal,
        elapsed=snapshot.elapsed,
        final_state=job.final_state,
    )


@router.get("/history", response_model=list[HistoryEntry])
def get_history() -> list[HistoryEntry]:
    return [HistoryEntry(**entry) for entry in list_history()]


@router.get("/history/{history_id}", response_model=HistoryResultResponse)
def get_history_result(history_id: str) -> HistoryResultResponse:
    try:
        ticker, trade_date, state = load_history(history_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="History entry not found") from exc
    return HistoryResultResponse(
        id=history_id,
        ticker=ticker,
        trade_date=trade_date,
        signal=extract_signal(state),
        final_state=state,
    )
