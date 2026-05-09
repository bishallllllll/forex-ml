"""Forex trading indicators module.

Provides SMA, EMA, RSI, MACD, and Bollinger Bands calculations
with input validation, edge case handling, and optional Decimal precision.
"""

from __future__ import annotations

import warnings
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal

import numpy as np
import pandas as pd


def _validate_series(data: pd.Series, name: str = "data") -> None:
    if not isinstance(data, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(data).__name__}")
    if data.empty:
        raise ValueError(f"{name} must not be empty")
    if data.isna().all():
        raise ValueError(f"{name} must not be entirely NaN")
    try:
        is_numeric = np.issubdtype(data.dtype, np.number)
    except TypeError:
        is_numeric = False
    if not is_numeric:
        raise TypeError(f"{name} must be numeric, got {data.dtype}")


def _validate_period(period: int, name: str = "period") -> None:
    if not isinstance(period, int):
        raise TypeError(f"{name} must be an integer, got {type(period).__name__}")
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    _validate_series(high, "high")
    _validate_series(low, "low")
    _validate_series(close, "close")
    _validate_period(period)
    _warn_insufficient(len(close), period, "ATR")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _warn_insufficient(data_len: int, period: int, name: str = "") -> None:
    needed = period
    if data_len < needed:
        warnings.warn(
            f"{name} insufficient data: need {needed} points, got {data_len}",
            RuntimeWarning,
            stacklevel=3,
        )


def sma(data: pd.Series, period: int) -> pd.Series:
    _validate_series(data, "data")
    _validate_period(period)
    _warn_insufficient(len(data), period, "SMA")
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    _validate_series(data, "data")
    _validate_period(period)
    _warn_insufficient(len(data), period, "EMA")
    result = data.ewm(span=period, adjust=False).mean()
    result.iloc[: period - 1] = np.nan
    return result


def rsi(
    data: pd.Series,
    period: int = 14,
    method: Literal["wilder", "ema"] = "wilder",
) -> pd.Series:
    _validate_series(data, "data")
    _validate_period(period)
    _warn_insufficient(len(data), period + 1, "RSI")

    if method not in ("wilder", "ema"):
        raise ValueError(f"method must be 'wilder' or 'ema', got {method!r}")

    deltas = data.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)

    if method == "wilder":
        avg_gain = gains.copy()
        avg_loss = losses.copy()
        avg_gain.iloc[:period] = np.nan
        avg_loss.iloc[:period] = np.nan
        if len(data) > period:
            avg_gain.iloc[period] = gains.iloc[1:period + 1].mean()
            avg_loss.iloc[period] = losses.iloc[1:period + 1].mean()
            alpha = 1.0 / period
            for i in range(period + 1, len(data)):
                avg_gain.iloc[i] = avg_gain.iloc[i - 1] * (1 - alpha) + gains.iloc[i] * alpha
                avg_loss.iloc[i] = avg_loss.iloc[i - 1] * (1 - alpha) + losses.iloc[i] * alpha
    else:
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()
        avg_gain.iloc[:period] = np.nan
        avg_loss.iloc[:period] = np.nan

    zero_loss = avg_loss == 0
    rs = avg_gain / avg_loss.where(~zero_loss)
    rsi_vals = 100 - (100 / (1 + rs))

    rsi_vals = rsi_vals.where(~zero_loss, 100.0)
    rsi_vals.iloc[:period] = np.nan
    return rsi_vals


def macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, pd.Series]:
    _validate_series(data, "data")
    _validate_period(fast_period)
    _validate_period(slow_period)
    _validate_period(signal_period)

    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        )

    macd_line = ema(data, fast_period) - ema(data, slow_period)
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
    ddof: int = 0,
) -> dict[str, pd.Series]:
    _validate_series(data, "data")
    _validate_period(period)
    if std_dev <= 0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")

    middle = sma(data, period)
    std = data.rolling(window=period).std(ddof=ddof)
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {"middle": middle, "upper": upper, "lower": lower}


def rolling_corr(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    _validate_series(series1, "series1")
    _validate_series(series2, "series2")
    _validate_period(period)
    return series1.rolling(window=period).corr(series2)


def rolling_skew(data: pd.Series, period: int = 20) -> pd.Series:
    _validate_series(data, "data")
    _validate_period(period)
    _warn_insufficient(len(data), period, "rolling_skew")
    return data.rolling(window=period).skew()


def rolling_kurt(data: pd.Series, period: int = 20) -> pd.Series:
    _validate_series(data, "data")
    _validate_period(period)
    _warn_insufficient(len(data), period, "rolling_kurt")
    return data.rolling(window=period).kurt()


def lag_feature(data: pd.Series, lag: int = 1) -> pd.Series:
    _validate_series(data, "data")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError(f"lag must be a positive integer, got {lag}")
    return data.shift(lag)


def volatility_regime(
    atr_series: pd.Series,
    low_percentile: float = 0.3,
    high_percentile: float = 0.7,
) -> pd.Series:
    _validate_series(atr_series, "atr_series")
    low_thresh = atr_series.quantile(low_percentile)
    high_thresh = atr_series.quantile(high_percentile)
    regime = pd.Series(1, index=atr_series.index, dtype=float)
    regime[atr_series <= low_thresh] = 0.0
    regime[atr_series >= high_thresh] = 2.0
    regime[atr_series.isna()] = np.nan
    return regime


def rsi_decimal(
    prices: list[Decimal],
    period: int = 14,
    precision: int = 8,
) -> list[Decimal]:
    if not isinstance(prices, list):
        raise TypeError(f"prices must be a list, got {type(prices).__name__}")
    if not all(isinstance(p, Decimal) for p in prices):
        raise TypeError("All prices must be Decimal instances")
    if len(prices) < period + 1:
        raise ValueError(
            f"Insufficient data: need at least {period + 1} prices, got {len(prices)}"
        )
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(-diff)

    avg_gain = Decimal("0")
    avg_loss = Decimal("0")
    alpha = Decimal("1") / Decimal(str(period))

    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= Decimal(str(period))
    avg_loss /= Decimal(str(period))

    rsi_values: list[Decimal] = [Decimal("0")] * (period + 1)

    for i in range(period, len(gains)):
        avg_gain = gains[i] * alpha + avg_gain * (Decimal("1") - alpha)
        avg_loss = losses[i] * alpha + avg_loss * (Decimal("1") - alpha)

        if avg_loss == 0:
            rs = Decimal("Infinity")
        else:
            rs = avg_gain / avg_loss

        if rs == Decimal("Infinity"):
            rsi_val = Decimal("100")
        else:
            rsi_val = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        rsi_values.append(
            rsi_val.quantize(Decimal("0." + "0" * precision), rounding=ROUND_HALF_UP)
        )

    return rsi_values


def calculate_all(data: pd.Series) -> dict[str, Any]:
    return {
        "sma_10": sma(data, 10),
        "sma_20": sma(data, 20),
        "sma_50": sma(data, 50),
        "ema_10": ema(data, 10),
        "ema_20": ema(data, 20),
        "rsi": rsi(data),
        "macd": macd(data),
        "bollinger": bollinger_bands(data),
    }


__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger_bands",
    "atr",
    "rolling_corr",
    "rolling_skew",
    "rolling_kurt",
    "lag_feature",
    "volatility_regime",
    "rsi_decimal",
    "calculate_all",
]
