# Forex ML

XGBoost + TFT ensemble with walk-forward backtesting for forex trading.

## ML Pipeline

```bash
# 1. Data ingestion
python -c "from ml.data.ingest import download_daily_ohlc; df = download_daily_ohlc()"

# 2. Feature engineering
python -c "from ml.features.technical import compute_indicators; df = compute_indicators(df)"

# 3. Model training (walk-forward XGBoost baseline)
python -c "from ml.models.xgb import train_xgb, prepare_direction_labels; ..."

# 4. Backtesting
python -c "from ml.backtest.engine import run_backtest; portfolio = run_backtest(df)"

# 5. Daily signal generation
python -c "from ml.models.sizing import position_size; ..."
```

## Kaggle Notebook Pipeline

| # | Notebook | GPU | Input | Output |
|---|----------|-----|-------|--------|
| 1 | `01_Data_Ingestion` | No | yfinance | `daily.parquet` |
| 2 | `02_Feature_Engineering` | No | `daily.parquet` | `features.parquet` |
| 3 | `03_Model_Training` | **Yes** | `features.parquet` | `xgb_model.joblib`, `predictions.parquet`, `optimal_threshold.npy` |
| 4 | `04_Backtesting_Evaluation` | No | `features.parquet` + `predictions.parquet` | metrics, charts |
| 5 | `05_Daily_Signal` | No | `xgb_model.joblib` | `signals.csv` |

## Anti-Overfit Guards

- Min 20 trades — ValueError
- Z-score ≥ 1.96 — ValueError
- Sharpe < 2.5 — warning
- Win rate < 85% — warning
- Feature pruning (bottom 20% removed)
- Purge + embargo (5d each)

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

Run tests:
```bash
python -m pytest tests/ -v
```
