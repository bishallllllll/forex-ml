import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ml.config import config


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def adf_test(series: pd.Series) -> tuple[float, float]:
    stat, pval, *_ = adfuller(series.dropna())
    return stat, pval


def align_multisaset(dataframes: dict[str, pd.DataFrame], freq: str = "D") -> pd.DataFrame:
    aligned = []
    for name, df in dataframes.items():
        df = df.resample(freq).last()
        df = df.add_prefix(f"{name}_")
        aligned.append(df)
    return pd.concat(aligned, axis=1)


def handle_missing(df: pd.DataFrame, method: str = "ffill", limit: int = 3) -> pd.DataFrame:
    if method == "ffill":
        return df.ffill(limit=limit).bfill(limit=limit)
    return df.dropna()


def purge_embargo_split(df, train_end, test_start, embargo_days=None):
    if embargo_days is None:
        embargo_days = config.models.embargo_days
    embargo_end = train_end + pd.Timedelta(days=embargo_days)
    if embargo_end > test_start:
        test_start = embargo_end
    return df.loc[:train_end], df.loc[test_start:]


def oos_holdout_split(df: pd.DataFrame, pct: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pct is None:
        pct = config.data.oos_holdout_pct
    cutoff = int(len(df) * (1 - pct))
    train = df.iloc[:cutoff]
    oos = df.iloc[cutoff:]
    return train, oos
