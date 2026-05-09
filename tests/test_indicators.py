"""Unit tests for forex_indicators module."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from forex_indicators import (
    bollinger_bands,
    calculate_all,
    ema,
    macd,
    rsi,
    rsi_decimal,
    sma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 100
SEED = 42
rng = np.random.default_rng(SEED)


def _series(values, dtype=float):
    return pd.Series(values, dtype=dtype)


def _uniform_data(n=N, scale=1.0):
    return _series(np.cumsum(rng.uniform(-scale, scale, size=n)) + 100)


# ===================================================================
# SMA
# ===================================================================


class TestSMA:
    def test_basic(self):
        s = _series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        assert result.iloc[2] == 2.0
        assert result.iloc[3] == 3.0
        assert result.iloc[4] == 4.0

    def test_first_n_minus_1_nan(self):
        result = sma(_series([1, 2, 3, 4, 5]), 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_period_larger_than_data(self):
        with pytest.warns(RuntimeWarning, match="insufficient data"):
            result = sma(_series([1, 2, 3]), 5)
        assert result.isna().all()

    def test_constant_series(self):
        result = sma(_series([5] * 10), 3)
        assert (result.dropna() == 5.0).all()

    def test_period_1(self):
        s = _series([10, 20, 30])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s)

    def test_invalid_period_zero(self):
        with pytest.raises(ValueError, match=">= 1"):
            sma(_series([1, 2, 3]), 0)

    def test_invalid_period_negative(self):
        with pytest.raises(ValueError, match=">= 1"):
            sma(_series([1, 2, 3]), -1)

    def test_invalid_period_type(self):
        with pytest.raises(TypeError, match="must be an integer"):
            sma(_series([1, 2, 3]), 3.0)

    def test_empty_series(self):
        with pytest.raises(ValueError, match="not be empty"):
            sma(pd.Series([], dtype=float), 5)

    def test_all_nan_series(self):
        with pytest.raises(ValueError, match="not be entirely NaN"):
            sma(pd.Series([np.nan, np.nan]), 5)

    def test_non_numeric(self):
        with pytest.raises(TypeError, match="must be numeric"):
            sma(pd.Series(["a", "b", "c"]), 3)

    def test_not_series(self):
        with pytest.raises(TypeError, match="must be a pandas Series"):
            sma([1, 2, 3], 3)


# ===================================================================
# EMA
# ===================================================================


class TestEMA:
    def test_basic(self):
        s = _series([1, 2, 3, 4, 5])
        result = ema(s, 3)
        assert not pd.isna(result.iloc[2])
        assert result.iloc[4] > result.iloc[2]

    def test_more_responsive_than_sma(self):
        s = _series([10, 10, 10, 10, 20])
        assert ema(s, 3).iloc[4] > sma(s, 3).iloc[4]

    def test_constant_series(self):
        result = ema(_series([5] * 10), 3)
        assert (result.dropna() == 5.0).all()

    def test_period_1(self):
        s = _series([10, 20, 30])
        result = ema(s, 1)
        pd.testing.assert_series_equal(result, s)

    def test_empty_series(self):
        with pytest.raises(ValueError, match="not be empty"):
            ema(pd.Series([], dtype=float), 5)


# ===================================================================
# RSI
# ===================================================================


class TestRSI:
    def test_range_property(self):
        s = _uniform_data()
        result = rsi(s, 14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_always_increasing_rsi_100(self):
        s = _series(range(20))
        result = rsi(s, 14).dropna()
        assert (result > 99).all()

    def test_always_decreasing_rsi_0(self):
        s = _series(reversed(range(20)))
        result = rsi(s, 14).dropna()
        assert (result < 1).all()

    def test_constant_prices_rsi_100(self):
        s = _series([50.0] * 20)
        result = rsi(s, 14).dropna()
        assert (result == 100.0).all()

    def test_known_rsi_value(self):
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10,
                  45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
                  46.00, 46.03, 46.41, 46.22, 45.64]
        result = rsi(_series(prices), 14)
        assert result.iloc[20] == pytest.approx(59.28, abs=0.01)

    def test_default_period(self):
        s = _series(range(30))
        result_14 = rsi(s)
        result_explicit = rsi(s, 14)
        pd.testing.assert_series_equal(result_14, result_explicit)

    def test_method_ema_is_different(self):
        s = _uniform_data()
        wilder = rsi(s, method="wilder")
        ema_method = rsi(s, method="ema")
        assert not wilder.equals(ema_method)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            rsi(_uniform_data(), method="invalid")  # type: ignore[arg-type]

    def test_insufficient_data_returns_nan(self):
        s = _series([1, 2, 3])
        with pytest.warns(RuntimeWarning, match="insufficient data"):
            result = rsi(s, 14)
        assert result.isna().all()

    def test_all_equal_gains_and_losses(self):
        s = _series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        result = rsi(s, 14).dropna()
        assert len(result) > 0
        assert (result > 0).all()
        assert (result < 100).all()


# ===================================================================
# MACD
# ===================================================================


class TestMACD:
    def test_returns_required_keys(self):
        result = macd(_uniform_data())
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_histogram_property(self):
        result = macd(_uniform_data())
        expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected)

    def test_fast_slow_invariant(self):
        result = macd(_uniform_data())
        valid = result["macd"].notna()
        assert (result["macd"][valid] != 0).any()

    def test_fast_gt_slow_raises(self):
        with pytest.raises(ValueError, match="fast_period"):
            macd(_uniform_data(), fast_period=26, slow_period=12)

    def test_fast_eq_slow_raises(self):
        with pytest.raises(ValueError, match="fast_period"):
            macd(_uniform_data(), fast_period=20, slow_period=20)

    def test_constant_series(self):
        flat = _series([50] * 50)
        result = macd(flat)
        valid = result["histogram"].dropna()
        assert (valid == 0).all()

    def test_empty_series(self):
        with pytest.raises(ValueError, match="not be empty"):
            macd(pd.Series([], dtype=float))


# ===================================================================
# Bollinger Bands
# ===================================================================


class TestBollingerBands:
    def test_returns_required_keys(self):
        result = bollinger_bands(_series(range(30)))
        assert set(result.keys()) == {"middle", "upper", "lower"}

    def test_upper_above_middle(self):
        result = bollinger_bands(_uniform_data(), 20)
        valid = result["middle"].notna()
        assert (result["upper"][valid] >= result["middle"][valid]).all()

    def test_lower_below_middle(self):
        result = bollinger_bands(_uniform_data(), 20)
        valid = result["middle"].notna()
        assert (result["lower"][valid] <= result["middle"][valid]).all()

    def test_middle_is_sma(self):
        s = _uniform_data()
        result = bollinger_bands(s, 20)
        expected = sma(s, 20)
        pd.testing.assert_series_equal(result["middle"], expected)

    def test_constant_series_bands(self):
        s = _series([100] * 50)
        result = bollinger_bands(s, 20)
        valid = result["middle"].notna()
        assert (result["upper"][valid] == 100.0).all()
        assert (result["lower"][valid] == 100.0).all()

    def test_zero_std_dev_raises(self):
        with pytest.raises(ValueError, match="std_dev must be positive"):
            bollinger_bands(_uniform_data(), std_dev=0)

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="std_dev must be positive"):
            bollinger_bands(_uniform_data(), std_dev=-1.0)

    def test_empty_series(self):
        with pytest.raises(ValueError, match="not be empty"):
            bollinger_bands(pd.Series([], dtype=float))


# ===================================================================
# rsi_decimal
# ===================================================================


class TestRSIDecimal:
    def test_decimal_output_type(self):
        prices = [Decimal(str(x)) for x in range(20)]
        result = rsi_decimal(prices)
        assert all(isinstance(v, Decimal) for v in result)
        assert len(result) == len(prices)

    def test_decimal_rsi_range(self):
        prices = [Decimal(str(x)) for x in range(30)]
        result = rsi_decimal(prices)
        valid = result[15:]
        assert all(Decimal("0") <= v <= Decimal("100") for v in valid)

    def test_decimal_insufficient_data(self):
        with pytest.raises(ValueError, match="Insufficient data"):
            rsi_decimal([Decimal("1")], 14)

    def test_decimal_non_decimal_input(self):
        with pytest.raises(TypeError, match="Decimal"):
            rsi_decimal([1.0, 2.0, 3.0])

    def test_decimal_not_list(self):
        with pytest.raises(TypeError, match="must be a list"):
            rsi_decimal("foo", 14)  # type: ignore[arg-type]

    def test_decimal_constant_prices(self):
        prices = [Decimal("50")] * 20
        result = rsi_decimal(prices, 14)
        assert all(v == Decimal("100") for v in result[15:])

    def test_decimal_precision(self):
        prices = [Decimal(str(x)) for x in range(30)]
        result_4 = rsi_decimal(prices, precision=4)
        result_8 = rsi_decimal(prices, precision=8)
        assert len(str(result_4[-1]).split(".")[1]) <= 4
        assert len(str(result_8[-1]).split(".")[1]) <= 8


# ===================================================================
# calculate_all
# ===================================================================


class TestCalculateAll:
    def test_returns_all_keys(self):
        result = calculate_all(_uniform_data())
        expected_keys = {"sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
                         "rsi", "macd", "bollinger"}
        assert set(result.keys()) == expected_keys

    def test_macd_is_dict(self):
        result = calculate_all(_uniform_data())
        assert isinstance(result["macd"], dict)

    def test_bollinger_is_dict(self):
        result = calculate_all(_uniform_data())
        assert isinstance(result["bollinger"], dict)

    def test_rsi_is_series(self):
        result = calculate_all(_uniform_data())
        assert isinstance(result["rsi"], pd.Series)


# ===================================================================
# Integration / edge cases across indicators
# ===================================================================


class TestIntegrationEdgeCases:
    def test_single_element(self):
        s = _series([42.0])
        with pytest.warns(RuntimeWarning):
            assert pd.isna(sma(s, 3).iloc[0])
        with pytest.warns(RuntimeWarning):
            assert pd.isna(rsi(s, 14).iloc[0])
        with pytest.warns(RuntimeWarning):
            bb = bollinger_bands(s, 20)
            assert pd.isna(bb["middle"].iloc[0])

    def test_two_elements_insufficient_for_sma_and_rsi(self):
        s = _series([10, 20])
        with pytest.warns(RuntimeWarning):
            assert sma(s, 3).isna().all()
        with pytest.warns(RuntimeWarning):
            assert rsi(s, 14).isna().all()

    def test_all_indicators_return_same_length(self):
        s = _uniform_data()
        assert len(sma(s, 10)) == len(s)
        assert len(ema(s, 10)) == len(s)
        assert len(rsi(s, 14)) == len(s)
        for v in macd(s).values():
            assert len(v) == len(s)
        for v in bollinger_bands(s).values():
            assert len(v) == len(s)

    def test_rsi_bounds_with_extreme_data(self):
        up = _series([100 + i for i in range(30)])
        down = _series([130 - i for i in range(30)])
        rsi_up = rsi(up, 14).dropna()
        rsi_down = rsi(down, 14).dropna()
        assert (rsi_up >= 50).all()
        assert (rsi_down <= 50).all()
