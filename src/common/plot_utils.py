from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, title: str, out_dir: str, filename: str) -> None:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=160)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, title: str, out_dir: str, filename: str) -> Optional[float]:
    ensure_dir(out_dir)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    except Exception:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{title}{' (AUC={:.3f})'.format(auc) if auc is not None else ''}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=160)
    plt.close(fig)
    return auc
