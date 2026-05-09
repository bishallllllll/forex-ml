import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings

from ml.config import config
from ml.backtest.metrics import check_min_trades, check_z_score, check_overfit_warnings


def apply_rollover_filter(df: pd.DataFrame) -> pd.DataFrame:
    if not config.backtest.rollover_filter or not callable(getattr(df.index, "hour", None)):
        return df
    rollover_mask = (df.index.hour >= config.backtest.rollover_start_hour) | (df.index.hour < config.backtest.rollover_end_hour)
    df = df.copy()
    df.loc[rollover_mask, "signal"] = 0
    return df


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    stress_test: bool = False,
) -> vbt.Portfolio:
    bc = config.backtest
    price = df[price_col]
    signals = df[signal_col]

    spread = bc.spread_pips
    if stress_test:
        spread *= bc.stress_test_spread_multiplier

    portfolio = vbt.Portfolio.from_signals(
        price,
        entries=signals == 1,
        short_entries=signals == -1,
        direction="both",
        init_cash=bc.initial_capital,
        size=bc.initial_capital * bc.position_size_pct,
        size_type="value",
        fees=spread / 10000,
        slippage=bc.slippage_pips / 10000,
        freq="D",
    )
    return portfolio


def compute_trade_metrics(portfolio: vbt.Portfolio) -> dict:
    stats = portfolio.stats()
    num_trades = stats.get("Total Trades", 0)
    win_rate = stats.get("Win Rate [%]", 0) / 100
    sharpe = stats.get("Sharpe Ratio", 0)

    check_min_trades(num_trades)
    wins = int(num_trades * win_rate)
    losses = num_trades - wins
    check_z_score(wins, losses)
    check_overfit_warnings(sharpe, win_rate)

    return {
        "total_return": stats.get("Total Return [%]", 0),
        "sharpe_ratio": sharpe,
        "max_drawdown": stats.get("Max Drawdown [%]", 0),
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": stats.get("Profit Factor", 0),
        "z_score": ((wins - num_trades * 0.5) / (np.sqrt(num_trades * 0.5 * 0.5))) if num_trades >= 2 else 0.0,
    }
