import pandas as pd
import numpy as np
import pytest

from ml.data.preprocess import log_returns, handle_missing, purge_embargo_split


def test_log_returns():
    s = pd.Series([100.0, 101.0, 99.0, 102.0])
    result = log_returns(s)
    assert result.iloc[0] != result.iloc[0]
    assert not result.iloc[1:].isna().any()


def test_handle_missing_ffill():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
    result = handle_missing(df, method="ffill")
    assert result.isna().sum().sum() == 0
    assert result.iloc[1, 0] == 1.0
    assert result.iloc[2, 0] == 1.0


def test_purge_embargo_split():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"close": range(100)}, index=dates)
    train_end = dates[60]
    test_start = dates[80]
    train, test = purge_embargo_split(df, train_end, test_start, embargo_days=5)
    assert len(train) == 61
    assert len(test) == 20
