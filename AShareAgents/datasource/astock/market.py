"""A 股 OHLCV 行情数据和技术指标适配器。"""

from __future__ import annotations

from datetime import datetime
from numbers import Number
from typing import Annotated
import logging
import math
import os
import re

from dateutil.relativedelta import relativedelta
import pandas as pd

from ..mootdx import get_daily_bars as _get_mootdx_daily_bars
from ..sinaFinance import get_daily_kline as _sina_kline_fallback
from .errors import MarketDataUnavailableError, NoMarketDataError, RECOVERABLE_DATA_SOURCE_ERRORS
from .symbols import _normalize_ticker

logger = logging.getLogger(__name__)

# 行情 OHLCV 数据优先从 mootdx 获取，并以 CSV 缓存。

def _parse_date(value: str) -> pd.Timestamp:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("日期必须为 YYYY-MM-DD 格式")
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))


def _validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Validate provider/cache payload before returning or persisting any rows."""
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("行情响应为空或不是表格")
    if not data.columns.is_unique or not set(columns).issubset(data.columns):
        raise ValueError("行情响应字段不完整或重复")
    frame = data[columns].copy()
    if any(isinstance(value, Number) for value in frame["Date"]):
        raise ValueError("行情日期不能为数值")
    frame["Date"] = pd.to_datetime(frame["Date"], format="ISO8601", errors="raise")
    if frame["Date"].isna().any() or frame["Date"].dt.tz is not None:
        raise ValueError("行情日期无效")
    frame["Date"] = frame["Date"].dt.normalize()
    if frame["Date"].duplicated().any():
        raise ValueError("行情日期重复")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].map(math.isfinite).all():
            raise ValueError("行情包含非有限数值")
    prices = frame[["Open", "High", "Low", "Close"]]
    if (prices <= 0).any().any() or (frame["Volume"] < 0).any():
        raise ValueError("行情价格或成交量无效")
    if ((frame["High"] < prices.max(axis=1)) | (frame["Low"] > prices.min(axis=1))).any():
        raise ValueError("行情高低价关系无效")
    return frame.sort_values("Date").reset_index(drop=True)


def _load_ohlcv_astock(symbol: str, curr_date: str) -> pd.DataFrame:
    """通过 mootdx 获取 OHLCV，缓存至 CSV，按 curr_date 过滤。

    类似 yFinance.stockstats_tools.load_ohlcv，但使用 mootdx 替代 yfinance。

    Returns:
        包含以下列的 DataFrame：Date、Open、High、Low、Close、Volume。
    """
    from ..config import get_config

    code = _normalize_ticker(symbol)
    cutoff = _parse_date(curr_date)
    config = get_config()
    cache_dir = config.get(
        "data_cache_dir", os.path.expanduser("~/.ashareagents/cache")
    )
    cache_file = os.path.join(cache_dir, f"{code}-astock-daily.csv")

    try:
        if os.path.exists(cache_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if mtime.date() == datetime.now().date():
                data = _validate_ohlcv(pd.read_csv(cache_file, encoding="utf-8"))
                return data[data["Date"] <= cutoff].copy()
    except RECOVERABLE_DATA_SOURCE_ERRORS as exc:
        logger.warning("忽略无效行情缓存 %s：%s", code, type(exc).__name__)

    # 适配层对每个供应商调用一次；底层 SDK 的超时/重试仍由其配置控制。
    # 空响应、超时、HTTP/格式错误触发回退。
    failures = []
    for name, provider in (("mootdx", _get_mootdx_daily_bars), ("sina", _sina_kline_fallback)):
        try:
            df = _validate_ohlcv(provider(code))
        except RECOVERABLE_DATA_SOURCE_ERRORS as exc:
            failures.append(f"{name}:{type(exc).__name__}")
            logger.warning("行情供应商失败 %s %s：%s", code, name, type(exc).__name__)
            continue
        if name == "sina":
            logger.info("行情数据源切换：%s 已使用新浪财经", code)
        break
    else:
        raise MarketDataUnavailableError(f"{code} 行情不可用（{', '.join(failures)}）")

    # 缓存是优化，写入失败不应丢弃已经验证的行情。
    try:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(cache_file, index=False, encoding="utf-8")
    except OSError as exc:
        logger.warning("行情缓存写入失败 %s：%s", code, type(exc).__name__)

    # 按 curr_date 过滤，防止前视偏差
    return df[df["Date"] <= cutoff].copy()


# 以下供应商方法必须与 interface.py 的 VENDOR_METHODS 签名保持一致。


# ---- 1. 获取行情数据（get_stock_data）----


def get_stock_data(
    symbol: Annotated[str, "A股代码（如 688017、SH688017）"],
    start_date: Annotated[str, "起始日期，格式 yyyy-mm-dd"],
    end_date: Annotated[str, "结束日期，格式 yyyy-mm-dd"],
) -> str:
    """获取 A 股 OHLCV 行情数据，复用 _load_ohlcv_astock 的缓存层。"""
    code = _normalize_ticker(symbol)

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)
    if start_dt > end_dt:
        raise ValueError("起始日期不能晚于结束日期")
    df = _load_ohlcv_astock(code, end_date)

    # 按日期范围过滤
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

    if df.empty:
        raise NoMarketDataError(f"在 {start_date} 至 {end_date} 期间未找到 A 股 '{code}' 的数据")

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    csv_out = df[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(
        index=False
    )

    header = f"# {code} (A股) 行情数据，{start_date} 至 {end_date}\n"
    header += f"# 总记录数: {len(df)}\n"
    header += (
        f"# 数据获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    return header + csv_out


# ---- 2. 获取技术指标（get_indicators）----

# 支持的技术指标及其说明
_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": "50 日均线：中期趋势指标。",
    "close_200_sma": "200 日均线：长期趋势基准。",
    "close_10_ema": "10 日 EMA：灵敏的短期均线。",
    "macd": "MACD：通过 EMA 差值计算的动量指标。",
    "macds": "MACD 信号线：MACD 线的 EMA 平滑。",
    "macdh": "MACD 柱：MACD 与信号线之间的差值。",
    "rsi": "RSI：超买/超卖动量指标（阈值 70/30）。",
    "boll": "布林带中轨：20 日均线基准线。",
    "boll_ub": "布林带上轨：中轨上方 2 个标准差。",
    "boll_lb": "布林带下轨：中轨下方 2 个标准差。",
    "atr": "ATR：平均真实波幅，衡量波动率。",
    "vwma": "VWMA：成交量加权移动平均。",
    "mfi": "MFI：资金流量指数（成交量 + 价格动量）。",
}


def get_indicators(
    symbol: Annotated[str, "A股代码"],
    indicator: Annotated[
        str, "技术指标（如 rsi、macd、close_50_sma）"
    ],
    curr_date: Annotated[str, "当前交易日，格式 YYYY-mm-dd"],
    look_back_days: Annotated[int, "回顾多少天"],
) -> str:
    """基于 mootdx OHLCV 数据，使用 stockstats 计算技术指标。"""
    from stockstats import wrap

    code = _normalize_ticker(symbol)

    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"不支持的指标 {indicator}。"
            f"请从以下选项中选择：{list(_INDICATOR_DESCRIPTIONS.keys())}"
        )

    # 行情失败必须传递给任务层，不可变成可供 Agent 继续推理的成功文本。
    data = _load_ohlcv_astock(code, curr_date)
    if data.empty:
        raise NoMarketDataError(f"{code} 在 {curr_date} 之前没有行情数据")
    try:
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # 触发 stockstats 计算
        df[indicator]

        # 构建 日期 -> 指标值 的查找表
        ind_dict = {}
        for _, row in df.iterrows():
            d = row["Date"]
            v = row[indicator]
            ind_dict[d] = "N/A" if pd.isna(v) else str(round(float(v), 4))

        # 生成回顾窗口内的输出
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_dt - relativedelta(days=look_back_days)

        lines = []
        dt = curr_dt
        while dt >= before:
            ds = dt.strftime("%Y-%m-%d")
            val = ind_dict.get(ds, "N/A：非交易日（周末或节假日）")
            lines.append(f"{ds}: {val}")
            dt -= relativedelta(days=1)

        result = (
            f"## {code} 的 {indicator} 指标值 "
            f"（{before.strftime('%Y-%m-%d')} 至 {curr_date}）:\n\n"
            + "\n".join(lines)
            + "\n\n"
            + _INDICATOR_DESCRIPTIONS.get(indicator, "")
        )
        return result

    except RECOVERABLE_DATA_SOURCE_ERRORS as e:
        return f"计算 {code} 的 {indicator} 指标时出错：{str(e)}"
