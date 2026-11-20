"""FastAPI 后端负责的分析执行服务。"""

from __future__ import annotations

import re
import threading
from typing import Any

from AShareAgents.api.jobs import AnalysisJob, PIPELINE_STAGES
from AShareAgents.llm.guardrails import LLMGuardError, LLMPolicy, TaskBudget
from AShareAgents.datasource.astock.errors import (
    DataSourceUnavailableError,
    MarketDataUnavailableError,
    NoMarketDataError,
)

_REPORT_KEY_TO_STAGE = {
    "market_report": "market",
    "sentiment_report": "social",
    "news_report": "news",
    "fundamentals_report": "fundamentals",
    "policy_report": "policy",
    "hot_money_report": "hot_money",
    "lockup_report": "lockup",
}


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _detect_completed_stages(chunk: dict[str, Any], job: AnalysisJob) -> None:
    for report_key, stage_id in _REPORT_KEY_TO_STAGE.items():
        content = chunk.get(report_key, "")
        if content and job.stage_status(stage_id) != "done":
            job.mark_stage_done(stage_id, _strip_think_tags(str(content)))

    dqs = chunk.get("data_quality_summary", "")
    if dqs and job.stage_status("quality_gate") != "done":
        job.mark_stage_done("quality_gate", str(dqs))

    debate = chunk.get("investment_debate_state")
    if isinstance(debate, dict) and debate.get("judge_decision"):
        if job.stage_status("debate") != "done":
            job.mark_stage_done("debate", str(debate["judge_decision"]))

    trader_plan = chunk.get("trader_investment_plan", "")
    if trader_plan and job.stage_status("trader") != "done":
        job.mark_stage_done("trader", _strip_think_tags(str(trader_plan)))

    risk = chunk.get("risk_debate_state")
    if isinstance(risk, dict) and risk.get("judge_decision"):
        if job.stage_status("risk") != "done":
            job.mark_stage_done("risk", str(risk["judge_decision"]))

    final = chunk.get("final_trade_decision", "")
    if final and job.stage_status("pm") != "done":
        job.mark_stage_done("pm", _strip_think_tags(str(final)))


def _infer_active_stage(job: AnalysisJob) -> None:
    if job.current_stage and job.stage_status(job.current_stage) == "active":
        return
    for stage_id in PIPELINE_STAGES:
        if job.stage_status(stage_id) == "pending":
            job.mark_stage_active(stage_id)
            return


def run_analysis(job: AnalysisJob, config: dict[str, Any]) -> None:
    from AShareAgents.api.stats import StatsCallbackHandler
    from AShareAgents.workflows.trading_graph import AShareAgentsGraph

    job.mark_running()
    stats = StatsCallbackHandler()
    runtime = TaskBudget(
        LLMPolicy.from_config(config), started_at=job.started_monotonic,
        on_change=job.update_llm_guard_stats,
    )
    job.update_llm_guard_stats(runtime.snapshot())
    try:
        runtime.checkpoint()
        #graph = AShareAgentsGraph(debug=True, config=config, callbacks=[stats], llm_runtime=runtime)
        selected_analysts = config.get("selected_analysts")

        graph_kwargs = {}
        if selected_analysts:
            graph_kwargs["selected_analysts"] = selected_analysts

        graph = AShareAgentsGraph(
            debug=True,
            config=config,
            callbacks=[stats],
            llm_runtime=runtime,
            **graph_kwargs,
        )
        graph._resolve_pending_entries(job.ticker)
        runtime.checkpoint()
        past_context = graph.memory_log.get_past_context(job.ticker)
        init_state = graph.propagator.create_initial_state(
            job.ticker, job.trade_date, past_context=past_context
        )
        args = graph.propagator.get_graph_args(callbacks=[stats])
        last_chunk: dict[str, Any] = {}

        for chunk in graph.graph.stream(init_state, **args):
            runtime.checkpoint()
            last_chunk = chunk
            _detect_completed_stages(chunk, job)
            _infer_active_stage(job)
            values = stats.get_stats()
            job.update_stats(
                values["llm_calls"],
                values["tool_calls"],
                values["tokens_in"],
                values["tokens_out"],
            )

        runtime.checkpoint()
        # A graph ending without a decision (or for another request) is not a
        # successful report. Validate before publishing files or memories.
        decision = last_chunk.get("final_trade_decision")
        if not isinstance(decision, str) or not decision.strip():
            raise ValueError("Analysis produced no final decision")
        if last_chunk.get("company_of_interest") != job.ticker or last_chunk.get("trade_date") != job.trade_date:
            raise ValueError("Report identity does not match the analysis request")

        signal = graph.process_signal(decision)
        graph.ticker = job.ticker
        runtime.checkpoint()
        graph._log_state(job.trade_date, last_chunk)
        runtime.checkpoint()
        final_state = graph.log_states_dict[str(job.trade_date)]

        final_decision = last_chunk.get("final_trade_decision", "")
        if final_decision:
            graph.memory_log.store_decision(job.ticker, job.trade_date, final_decision)

        runtime.checkpoint()
        job.mark_complete(final_state, signal)
    finally:
        # An exception need not produce another graph chunk. Preserve last usage.
        values = stats.get_stats()
        job.update_stats(values["llm_calls"], values["tool_calls"], values["tokens_in"], values["tokens_out"])
        runtime.close()


def _public_failure_reason(exc: Exception) -> str:
    """Keep useful categories without exposing raw headers, URLs or config."""
    if isinstance(exc, LLMGuardError):
        return exc.public_message
    if isinstance(exc, NoMarketDataError):
        return "请求区间内没有可用行情数据"
    if isinstance(exc, MarketDataUnavailableError):
        return "行情数据源暂时不可用，请稍后重试"
    if isinstance(exc, DataSourceUnavailableError):
        return "外部数据服务暂时不可用，请稍后重试"
    return f"Analysis failed ({type(exc).__name__})"


def start_analysis(job: AnalysisJob, config: dict[str, Any]) -> threading.Thread | None:
    def _target() -> None:
        try:
            run_analysis(job, config)
        except Exception as exc:
            job.mark_error(_public_failure_reason(exc))

    try:
        thread = threading.Thread(target=_target, daemon=True, name=f"analysis-{job.task_id}")
        thread.start()
    except Exception as exc:
        # The job is already registered. Preserve its queryable handle and do
        # not strand it in pending when the OS cannot start a worker.
        job.mark_error(_public_failure_reason(exc))
        return None
    return thread
