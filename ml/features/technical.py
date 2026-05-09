import pandas as pd
import pandas_ta as ta

from ml.config import config


def _compute_single_pair(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", None)
    fc = config.features
    out = df.copy()

    out["RSI"] = ta.rsi(close, length=fc.rsi_period)
    macd_df = ta.macd(close, fast=fc.macd_fast, slow=fc.macd_slow, signal=fc.macd_signal)
    if macd_df is not None:
        out["MACD_Hist"] = macd_df.iloc[:, 2]
        out["MACD_Signal"] = macd_df.iloc[:, 1]
        out["MACD"] = macd_df.iloc[:, 0]
    bb_df = ta.bbands(close, length=fc.bb_period, std=fc.bb_std)
    if bb_df is not None:
        out["BB_PctB"] = bb_df.iloc[:, 3]
        out["BB_Width"] = bb_df.iloc[:, 4]
    out["ATR"] = ta.atr(high, low, close, length=fc.atr_period)
    out["SMA_20"] = ta.sma(close, length=20)
    out["SMA_50"] = ta.sma(close, length=50)
    ema_fast = ta.ema(close, length=12)
    ema_slow = ta.ema(close, length=26)
    if ema_slow is not None and (ema_slow != 0).any():
        out["MA_Distance"] = (close - ema_slow) / ema_slow
    out["Body_Ratio"] = (close - out["open"]) / (out["high"] - out["low"]).clip(lower=1e-8)
    out["Upper_Wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / (out["high"] - out["low"]).clip(lower=1e-8)
    out["Lower_Wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / (out["high"] - out["low"]).clip(lower=1e-8)
    out["Log_Returns"] = ta.log_return(close)
    return out


def compute_indicators_multi(df: pd.DataFrame) -> pd.DataFrame:
    if "pair" not in df.columns:
        return _compute_single_pair(df)
    groups = []
    for pair_name, group in df.groupby("pair", sort=False):
        processed = _compute_single_pair(group)
        groups.append(processed)
    result = pd.concat(groups).sort_index()
    return result


compute_indicators = compute_indicators_multi


def shift_features(df: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
    exclude = {"open", "high", "low", "close", "volume", "pair", "pair_id"}
    feature_cols = [c for c in df.columns if c.lower() not in exclude]
    shifted = df[feature_cols].shift(shift)
    for col in ["pair", "pair_id"]:
        if col in df.columns:
            shifted[col] = df[col]
    return shifted.dropna()
