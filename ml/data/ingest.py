import subprocess
from pathlib import Path

import pandas as pd
import yfinance as yf

from ml.config import config


def download_kaggle(dataset_ref: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(target_dir), "--unzip"],
        check=True,
    )
    return target_dir


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    return df.rename(columns={c: c.lower() for c in df.columns if c.lower() in rename})


def fetch_yfinance(pair: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    ticker = f"{pair}=X"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def download_multi_pair() -> pd.DataFrame:
    frames = []
    for pair in config.data.currency_pairs:
        df = fetch_yfinance(pair, period="2y", interval="1d")
        df = _standardize_columns(df)
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            continue
        df["pair"] = pair
        df["pair_id"] = config.data.pair_id_map[pair]
        frames.append(df)
    if not frames:
        raise ValueError("No pairs could be downloaded")
    combined = pd.concat(frames)
    combined = combined.sort_index()
    return combined


def validate_data(df: pd.DataFrame) -> dict:
    issues = []
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair]
        if not sub.index.is_monotonic_increasing:
            issues.append(f"{pair}: index not monotonic")
        nan_streaks = sub["close"].isna().astype(int).groupby(sub["close"].notna().cumsum()).sum()
        if nan_streaks.max() > config.data.max_consecutive_nan:
            issues.append(f"{pair}: NaN streak of {nan_streaks.max()} > {config.data.max_consecutive_nan}")
        if len(sub) < 100:
            issues.append(f"{pair}: only {len(sub)} rows")
    return {"valid": len(issues) == 0, "issues": issues}
