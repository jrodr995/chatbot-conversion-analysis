from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def optimize_threshold(y_true, y_prob, metric: str = "accuracy") -> tuple[float, float]:
    """Find probability threshold in [0,1] that maximizes the chosen metric.

    Returns (best_threshold, best_score).
    """
    thresholds = np.linspace(0.0, 1.0, 201)
    best_t, best_s = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "accuracy":
            s = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            s = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            s = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            s = recall_score(y_true, y_pred, zero_division=0)
        else:
            s = accuracy_score(y_true, y_pred)
        if s > best_s:
            best_s = s
            best_t = t
    return float(best_t), float(best_s)
