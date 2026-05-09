import warnings

import numpy as np
import pandas as pd

from ml.config import config


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / std


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-8
    return np.sqrt(periods_per_year) * excess.mean() / downside_std


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    return gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")


def walk_forward_efficiency(in_sample_sharpe: float, out_of_sample_sharpe: float) -> float:
    return out_of_sample_sharpe / in_sample_sharpe if in_sample_sharpe != 0 else 0.0


def z_score(wins: int, losses: int) -> float:
    n = wins + losses
    if n < 2:
        return 0.0
    p = wins / n
    expected = n * 0.5
    std = np.sqrt(n * 0.5 * 0.5)
    if std == 0:
        return 0.0
    return (wins - expected) / std


def check_min_trades(num_trades: int) -> bool:
    threshold = config.models.min_trades_per_fold
    if num_trades < threshold:
        raise ValueError(f"Only {num_trades} trades, need minimum {threshold}")
    return True


def check_z_score(wins: int, losses: int) -> bool:
    z = z_score(wins, losses)
    threshold = config.models.z_score_threshold
    if z < threshold:
        raise ValueError(f"Z-score {z:.3f} < {threshold} — not statistically significant")
    return True


def check_overfit_warnings(sharpe: float, win_rate: float):
    if sharpe > config.models.max_acceptable_sharpe:
        warnings.warn(f"Sharpe {sharpe:.2f} > {config.models.max_acceptable_sharpe} — possible overfit")
    if win_rate > config.models.max_acceptable_win_rate:
        warnings.warn(f"Win rate {win_rate:.1%} > {config.models.max_acceptable_win_rate:.0%} — possible overfit")
