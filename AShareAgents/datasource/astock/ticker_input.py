"""Pure A-share syntax checks; no provider access or listing-membership claims."""

from __future__ import annotations

import re

_CODE = re.compile(r"(?:(SH|SZ|BJ))?([0-9]{6})(?:\.(SH|SZ|BJ))?", re.IGNORECASE)
_NAME = re.compile(r"[\u4e00-\u9fffA-Za-z0-9*()（）·\- \u3000]+")


class AmbiguousStockNameError(ValueError):
    """A syntactically valid name matches more than one stock."""


def stock_exchange(code: str) -> str:
    """Return exchange for supported stock code families, excluding B shares."""
    if not re.fullmatch(r"[0-9]{6}", code):
        raise ValueError("股票代码必须为六位 ASCII 数字")
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8", "920")):
        return "bj"
    raise ValueError("不支持的 A 股代码号段")


def normalize_code(value: str) -> str:
    """Accept bare codes and consistent SH/SZ/BJ prefixes/suffixes."""
    match = _CODE.fullmatch(value.strip())
    if match is None:
        raise ValueError("请输入合法 A 股代码、交易所前后缀代码或中文股票名称")
    prefix, code, suffix = match.groups()
    exchange = stock_exchange(code)
    if any(marker and marker.lower() != exchange for marker in (prefix, suffix)):
        raise ValueError("交易所标记与股票代码不匹配")
    return code


def validate_ticker_input(value: str) -> str:
    """Normalize codes; validate name syntax before business resolution."""
    text = value.strip()
    if not text or len(text) > 64:
        raise ValueError("股票输入长度必须为 1 至 64 个字符")
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        if _NAME.fullmatch(text) is None:
            raise ValueError("股票名称包含不支持的字符")
        return text.replace(" ", "").replace("\u3000", "")
    return normalize_code(text)
