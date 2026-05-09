import numpy as np
import pandas as pd

from ml.config import config


def oos_holdout_split(df: pd.DataFrame, pct: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pct is None:
        pct = config.data.oos_holdout_pct
    cutoff = int(len(df) * (1 - pct))
    train = df.iloc[:cutoff]
    oos = df.iloc[cutoff:]
    return train, oos


def walk_forward_split(
    df: pd.DataFrame,
    train_months: int | None = None,
    test_months: int | None = None,
    stride_months: int | None = None,
):
    train_m = train_months or config.models.walk_forward_train_months
    test_m = test_months or config.models.walk_forward_test_months
    stride_m = stride_months or config.models.walk_forward_stride_months

    days_per_month = 30
    train_size = train_m * days_per_month
    test_size = test_m * days_per_month
    stride = stride_m * days_per_month

    folds = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_end = start + train_size
        test_end = train_end + test_size
        folds.append({
            "train": df.iloc[start:train_end],
            "test": df.iloc[train_end:test_end],
            "train_start": df.index[start],
            "train_end": df.index[train_end - 1],
            "test_start": df.index[train_end],
            "test_end": df.index[test_end - 1],
        })
        start += stride
    return folds


def encode_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pair_id"] = df["pair"].map(config.data.pair_id_map)
    return df


def prepare_labels(df: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    if horizon is None:
        horizon = config.features.target_horizon
    future_close = df["close"].shift(-horizon)
    return (future_close > df["close"]).astype(int)
