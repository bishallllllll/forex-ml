import numpy as np
import pandas as pd
import pytest

from ml.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, z_score, check_min_trades, check_z_score


def test_sharpe_ratio_positive():
    returns = pd.Series([0.001] * 100)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 0


def test_sortino_ratio():
    returns = pd.Series([0.001] * 90 + [-0.01] * 10)
    sort = sortino_ratio(returns, periods_per_year=252)
    assert isinstance(sort, float)


def test_max_drawdown():
    equity = pd.Series([100, 110, 105, 95, 80, 85])
    mdd = max_drawdown(equity)
    assert mdd < 0
    assert mdd > -0.5


def test_z_score():
    z = z_score(60, 40)
    assert z > 1.5


def test_z_score_random():
    z = z_score(50, 50)
    assert z < 1.0


def test_check_min_trades_passes():
    assert check_min_trades(50) is True


def test_check_min_trades_fails():
    with pytest.raises(ValueError):
        check_min_trades(5)


def test_check_z_score_fails():
    with pytest.raises(ValueError):
        check_z_score(10, 10)
