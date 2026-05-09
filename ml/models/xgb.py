import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from ml.config import config
from ml.models.utils import prepare_labels, encode_pairs


def prepare_direction_labels(df: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    return prepare_labels(df, horizon=horizon)


def train_val_split(X: pd.DataFrame, y: pd.Series, val_pct: float = 0.2):
    split = int(len(X) * (1 - val_pct))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def hp_tune_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    hp = config.models.hp_tuning_params
    base = config.models.xgb_params.copy()
    base.pop("early_stopping_rounds", None)
    model = xgb.XGBClassifier(
        **base, objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    search = RandomizedSearchCV(
        model, hp, n_iter=config.models.hp_tuning_n_iter,
        cv=config.models.hp_tuning_cv_folds,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_params_


def train_xgb(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> CalibratedClassifierCV:
    params = config.models.xgb_params.copy()
    params.pop("early_stopping_rounds", None)
    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def prune_features(model: xgb.XGBClassifier, feature_names: list[str], pct: float | None = None) -> list[str]:
    if pct is None:
        pct = config.models.feature_prune_pct
    importance = model.feature_importances_
    threshold = np.percentile(importance, pct * 100)
    kept = [f for f, imp in zip(feature_names, importance) if imp > threshold]
    return kept if kept else feature_names


def threshold_scan(
    model: CalibratedClassifierCV, X_val: pd.DataFrame, y_val: pd.Series,
) -> tuple[float, float, pd.DataFrame]:
    probs = model.predict_proba(X_val)[:, 1]
    results = []
    best_sharpe = -1e9
    best_threshold = 0.5
    cfg = config.models
    for t in np.arange(cfg.threshold_min, cfg.threshold_max + 1e-9, cfg.threshold_step):
        preds = (probs >= t).astype(int)
        trades = preds.sum()
        if trades < 5:
            continue
        ret = (2 * preds - 1) * 0.01
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        results.append({"threshold": t, "trades": trades, "sharpe": sharpe})
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = t
    return best_threshold, best_sharpe, pd.DataFrame(results)


def evaluate_xgb(
    model: CalibratedClassifierCV, X_test: pd.DataFrame, y_test: pd.Series,
    threshold: float = 0.5, pair: str | None = None,
) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "trades": int(preds.sum()),
        "predictions": preds,
        "probabilities": probs,
        "pair": pair,
    }


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"open", "high", "low", "close", "volume", "pair", "pair_id", "target"}
    return [c for c in df.columns if c not in exclude]
