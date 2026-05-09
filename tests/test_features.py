import pandas as pd
import numpy as np
import pytest

from ml.features.macro import compute_surprise

pandas_ta = pytest.importorskip("pandas_ta", reason="pandas-ta not installed (Kaggle only)")


def test_compute_indicators_adds_columns():
    from ml.features.technical import compute_indicators
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 102,
        "low": np.random.randn(200).cumsum() + 98,
        "close": np.random.randn(200).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 200),
    }, index=dates)
    result = compute_indicators(df)
    expected_cols = {"RSI", "MACD_Hist", "ATR", "SMA_20", "SMA_50", "Log_Returns"}
    assert expected_cols.issubset(result.columns)


def test_shift_features():
    from ml.features.technical import shift_features
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0],
        "RSI": [50.0, 55.0, 60.0],
        "ATR": [1.0, 1.1, 1.2],
    })
    result = shift_features(df, shift=1)
    assert len(result) == 2
    assert result["RSI"].iloc[0] == 50.0
    assert result["ATR"].iloc[0] == 1.0


def test_compute_surprise():
    actual = pd.Series([0.5, 1.2, -0.3])
    forecast = pd.Series([0.3, 1.0, 0.0])
    result = compute_surprise(actual, forecast)
    expected = pd.Series([0.2, 0.2, -0.3])
    pd.testing.assert_series_equal(result, expected)
