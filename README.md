# Forex ML

Machine learning pipeline for forex trading direction prediction.

**Pipeline:** Data → Features → XGBoost/TFT Ensemble → Walk-Forward Backtest → Signal

Built with XGBoost, scikit-learn, pandas-ta, and vectorbt. Runs on Kaggle with GPU support for model training.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Structure

| Path | Purpose |
|------|---------|
| `ml/data/` | Data ingestion (yfinance) |
| `ml/features/` | Technical indicators (pandas-ta) |
| `ml/models/` | XGBoost + TFT ensemble |
| `ml/backtest/` | Vectorbt backtesting engine |
| `ml/config.py` | Central configuration |
| `ml/notebooks/` | Kaggle-native notebooks (5 stages) |
| `scripts/` | Kaggle push scripts |
| `tests/` | Test suite |
| `forex_indicators/` | Hand-rolled technical indicators |
