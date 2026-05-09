import pandas as pd
import numpy as np

from ml.models.utils import walk_forward_split, oos_holdout_split
from ml.models.sizing import kelly_criterion, fractional_kelly, position_size


def test_walk_forward_split():
    df = pd.DataFrame({"close": np.random.randn(500).cumsum() + 100},
                      index=pd.date_range("2020-01-01", periods=500, freq="D"))
    folds = walk_forward_split(df, train_months=4, test_months=1, stride_months=1)
    assert len(folds) >= 1
    assert "train" in folds[0]
    assert "test" in folds[0]


def test_oos_holdout_split():
    df = pd.DataFrame({"close": range(100)}, index=pd.date_range("2020-01-01", periods=100, freq="D"))
    train, oos = oos_holdout_split(df, pct=0.2)
    assert len(train) == 80
    assert len(oos) == 20


def test_kelly_criterion():
    k = kelly_criterion(0.6, 1.0, 1.0)
    assert 0.1 < k < 0.3


def test_fractional_kelly():
    k = fractional_kelly(0.6, 1.0, 1.0, fraction=0.25)
    assert 0.0 < k < 0.1


def test_position_size():
    size = position_size(10000.0, 0.7, 0.6, 1.0, 1.0, max_risk_pct=0.02)
    assert 0 < size < 200
