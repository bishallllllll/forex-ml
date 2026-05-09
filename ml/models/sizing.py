import numpy as np
import pandas as pd

from ml.config import config


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0.0
    r = avg_win / abs(avg_loss)
    kelly = win_rate - (1 - win_rate) / r
    return max(0.0, kelly)


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float | None = None) -> float:
    if fraction is None:
        fraction = config.backtest.kelly_fraction
    full = kelly_criterion(win_rate, avg_win, avg_loss)
    return full * fraction


def position_size(
    account_value: float,
    confidence: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_risk_pct: float | None = None,
) -> float:
    if max_risk_pct is None:
        max_risk_pct = config.backtest.risk_per_trade
    kelly = fractional_kelly(win_rate, avg_win, avg_loss)
    size = account_value * max_risk_pct * kelly * confidence
    return float(size)


def compute_trade_stats(trades: pd.DataFrame) -> dict:
    if len(trades) < 1:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "num_trades": 0}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    return {
        "win_rate": len(wins) / len(trades) if len(trades) > 0 else 0.0,
        "avg_win": wins["pnl"].mean() if len(wins) > 0 else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) > 0 else 0.0,
        "num_trades": len(trades),
    }
