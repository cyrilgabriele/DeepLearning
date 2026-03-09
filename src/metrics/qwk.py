import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa between two ordinal rating arrays."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(y_cont: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Map continuous predictions to ordinal classes 1-8 using 7 thresholds."""
    return np.digitize(y_cont, thresholds) + 1


def optimize_thresholds(
    y_true: np.ndarray,
    y_cont: np.ndarray,
    num_classes: int = 8,
) -> tuple[np.ndarray, float]:
    """Find optimal rounding thresholds that maximize QWK."""
    initial = np.arange(1.5, num_classes, 1.0)

    def neg_qwk(thresholds):
        t = np.sort(thresholds)
        preds = _apply_thresholds(y_cont, t)
        preds = np.clip(preds, 1, num_classes)
        return -quadratic_weighted_kappa(y_true, preds)

    result = minimize(neg_qwk, initial, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-4})
    best_thresholds = np.sort(result.x)
    best_kappa = -result.fun
    return best_thresholds, best_kappa
