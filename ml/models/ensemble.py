import numpy as np
from sklearn.metrics import accuracy_score


def ensemble_probs(xgb_probs: np.ndarray, tft_probs: np.ndarray, weight: float = 0.5) -> np.ndarray:
    return weight * xgb_probs + (1 - weight) * tft_probs


def optimize_weight(
    xgb_probs: np.ndarray, tft_probs: np.ndarray, y_true: np.ndarray,
) -> tuple[float, float]:
    best_w = 0.5
    best_acc = 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blended = ensemble_probs(xgb_probs, tft_probs, w)
        preds = (blended >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_w = w
    return best_w, best_acc


def ensemble_predict(
    xgb_probs: np.ndarray, tft_probs: np.ndarray,
    xgb_threshold: float, ensemble_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    blended = ensemble_probs(xgb_probs, tft_probs, ensemble_weight)
    signals = np.where(blended >= xgb_threshold, 1, np.where(blended < (1 - xgb_threshold), -1, 0))
    return signals, blended
