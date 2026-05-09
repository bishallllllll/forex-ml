from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
NOTEBOOK_DIR = BASE_DIR / "notebooks"

for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


CURRENCY_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD",
    "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "EURCHF",
]

PAIR_ID_MAP = {pair: i for i, pair in enumerate(CURRENCY_PAIRS)}

PER_PAIR_SPREADS = {
    "EURUSD": 1.0, "GBPUSD": 1.2, "USDJPY": 1.0,
    "USDCAD": 1.8, "AUDUSD": 1.5, "NZDUSD": 2.5,
    "USDCHF": 2.0, "EURGBP": 1.5, "EURJPY": 2.0, "EURCHF": 2.5,
}


@dataclass
class DataConfig:
    kaggle_daily: str = "asaniczka/forex-exchange-rate-since-2004-updated-daily"
    kaggle_hourly: str = "orkunaktas/eurusd-1h-2020-2024-september-forex"
    kaggle_macro: str = "kanchana1990/yahoo-finance-global-markets-intelligence-2026"
    kaggle_calendar: str = "devorvant/economic-calendar"
    currency_pairs: list[str] = field(default_factory=lambda: CURRENCY_PAIRS)
    pair_id_map: dict = field(default_factory=lambda: PAIR_ID_MAP)
    per_pair_spreads: dict = field(default_factory=lambda: PER_PAIR_SPREADS)
    test_size: float = 0.2
    val_size: float = 0.1
    max_consecutive_nan: int = 5
    oos_holdout_pct: float = 0.20


@dataclass
class FeatureConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    lookback_days: int = 90
    target_horizon: int = 1
    min_features: int = 10
    vol_regime_low_pct: float = 0.3
    vol_regime_high_pct: float = 0.7
    rolling_corr_period: int = 20
    rolling_skew_period: int = 20
    rolling_kurt_period: int = 20
    rolling_ret_period: int = 5
    lags: tuple = (1, 2, 3, 5)


@dataclass
class ModelConfig:
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 20,
    })
    hp_tuning_n_iter: int = 30
    hp_tuning_cv_folds: int = 3
    hp_tuning_params: dict = field(default_factory=lambda: {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0, 0.1, 1, 5],
        "reg_lambda": [0, 0.1, 1, 5],
    })
    tft_hidden_size: int = 64
    tft_attention_heads: int = 4
    tft_max_encoder_length: int = 90
    tft_max_prediction_length: int = 1
    tft_batch_size: int = 64
    tft_max_epochs: int = 50
    tft_learning_rate: float = 0.001
    ensemble_weight: float = 0.5
    walk_forward_train_months: int = 4
    walk_forward_test_months: int = 1
    walk_forward_stride_months: int = 1
    purge_days: int = 5
    embargo_days: int = 5
    threshold_min: float = 0.50
    threshold_max: float = 0.90
    threshold_step: float = 0.05
    min_trades_per_fold: int = 20
    z_score_threshold: float = 1.96
    max_acceptable_sharpe: float = 2.5
    max_acceptable_win_rate: float = 0.85
    feature_prune_pct: float = 0.20


@dataclass
class BacktestConfig:
    spread_pips: float = 1.5
    slippage_pips: float = 0.5
    swap_daily: float = -0.00002
    initial_capital: float = 10000.0
    position_size_pct: float = 0.02
    leverage: int = 1
    risk_per_trade: float = 0.02
    kelly_fraction: float = 0.25
    rollover_filter: bool = True
    rollover_start_hour: int = 23
    rollover_end_hour: int = 0
    asymmetric_slippage_stop: float = 0.5
    stress_test_spread_multiplier: float = 3.0


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


config = AppConfig()
