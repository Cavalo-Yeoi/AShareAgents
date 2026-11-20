"""Validate graph tool scope against trusted state before executing any tool I/O."""
from collections.abc import Mapping

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from AShareAgents.datasource.astock.ticker_input import normalize_code

# Industry comparison is anchored to the current company but may return peers.
_COMPANY_ARGUMENT = {
    'get_stock_data': 'symbol', 'get_indicators': 'symbol',
    **dict.fromkeys((
        'get_news', 'get_insider_transactions', 'get_fundamentals',
        'get_balance_sheet', 'get_cashflow', 'get_income_statement',
        'get_profit_forecast', 'get_concept_blocks', 'get_fund_flow',
        'get_dragon_tiger_board', 'get_lockup_expiry', 'get_industry_comparison',
        'search_company_official_documents',
    ), 'ticker'),
}
_MARKET_WIDE = frozenset((
    'get_global_news', 'get_hot_stocks', 'get_northbound_flow',
    'search_policy_industry_knowledge',
))


def _denial(request):
    call = request.tool_call
    name, args = call.get('name'), call.get('args')
    if request.tool is None or not isinstance(args, dict):
        return 'unknown_tool_or_invalid_arguments'
    if name in _MARKET_WIDE:
        return None
    field = _COMPANY_ARGUMENT.get(name)
    if field is None:
        return 'unclassified_tool'
    state = request.state
    target = state.get('company_of_interest') if isinstance(state, Mapping) else getattr(state, 'company_of_interest', None)
    value = args.get(field)
    if not isinstance(target, str) or not isinstance(value, str):
        return 'missing_or_invalid_company'
    try:
        # Syntax-only normalization: never resolves names through a provider.
        if normalize_code(target) != normalize_code(value):
            return 'company_outside_task_scope'
    except ValueError:
        return 'invalid_company_code'
    return None


def _rejected(request, reason):
    return ToolMessage(
        content=f'TOOL_SCOPE_DENIED: {reason}。公司工具仅允许当前任务标的；未执行工具或外部请求。请使用任务标的，市场及行业工具按其既定用途使用。',
        name=request.tool_call['name'], tool_call_id=request.tool_call['id'], status='error',
    )


def _wrap(request, execute):
    reason = _denial(request)
    return _rejected(request, reason) if reason else execute(request)


async def _awrap(request, execute):
    reason = _denial(request)
    return _rejected(request, reason) if reason else await execute(request)


def scoped_tool_node(tools):
    """Keep original tool schemas/results; validate sync and async execution."""
    return ToolNode(tools, wrap_tool_call=_wrap, awrap_tool_call=_awrap)
