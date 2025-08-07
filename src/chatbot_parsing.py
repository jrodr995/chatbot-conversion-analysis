from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
from common.plot_utils import save_confusion_matrix, save_roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr

from common.data_loader import load_dataframe


def compute_and_print_models(df: pd.DataFrame, no_plots: bool = True, save_figures: bool = False, fig_dir: str = "figures") -> None:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["has_appt"] = df["HAS_APPT_SCHEDULED"].astype(int)
    df["has_rfi"] = df["HAS_RFI_SUBMISSION"].astype(int)

    feature_cols = [
        "TOTAL_USER_MESSAGES",
        "AVG_USER_WORDS_PER_MSG",
        "MAX_USER_WORDS_IN_MSG",
        "USER_ENGAGEMENT_DURATION",
        "MESSAGE_COUNT",
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
    ]

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    # Appointments model
    y_appt = df["has_appt"]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_scaled, y_appt, test_size=0.2, random_state=42)
    model_appt = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_appt.fit(X_train_a, y_train_a)
    y_pred_a = model_appt.predict(X_test_a)
    y_prob_a = model_appt.predict_proba(X_test_a)[:, 1]

    print("=== Appointment Scheduling Model ===")
    print(classification_report(y_test_a, y_pred_a))
    if save_figures and not no_plots:
        save_confusion_matrix(y_test_a, y_pred_a, "Appointments Confusion Matrix", fig_dir, "appt_cm.png")
        save_roc_curve(y_test_a, y_prob_a, "Appointments ROC", fig_dir, "appt_roc.png")

    # RFI model
    y_rfi = df["has_rfi"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_rfi, test_size=0.2, random_state=42)
    model_rfi = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_rfi.fit(X_train_r, y_train_r)
    y_pred_r = model_rfi.predict(X_test_r)
    y_prob_r = model_rfi.predict_proba(X_test_r)[:, 1]

    print("\n=== RFI Submission Model ===")
    print(classification_report(y_test_r, y_pred_r))
    if save_figures and not no_plots:
        save_confusion_matrix(y_test_r, y_pred_r, "RFI Confusion Matrix", fig_dir, "rfi_cm.png")
        save_roc_curve(y_test_r, y_prob_r, "RFI ROC", fig_dir, "rfi_roc.png")

    # Correlations
    def corr(title: str, target_col: str, metric: str) -> tuple[float, float]:
        r, p = pointbiserialr(df[target_col], df[metric])
        print(f"{title:<50}: r={r:+.3f}, p={p:.3g}")
        return r, p

    print("\n=== CORRELATIONS: Appointments ===")
    for col in [
        "AVG_USER_WORDS_PER_MSG",
        "USER_ENGAGEMENT_DURATION",
        "MAX_USER_WORDS_IN_MSG",
        "TOTAL_USER_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
        "TOTAL_AGENT_MESSAGES",
        "MESSAGE_COUNT",
    ]:
        corr(f"{col} vs Appointments", "has_appt", col)

    print("\n=== CORRELATIONS: RFI ===")
    for col in [
        "AVG_USER_WORDS_PER_MSG",
        "USER_ENGAGEMENT_DURATION",
        "MAX_USER_WORDS_IN_MSG",
        "TOTAL_USER_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
        "TOTAL_AGENT_MESSAGES",
        "MESSAGE_COUNT",
    ]:
        corr(f"{col} vs RFI", "has_rfi", col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitized chatbot analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots to keep CLI non-interactive")
    parser.add_argument("--save-figures", action="store_true", help="Save ROC and confusion matrix figures to figures/")
    parser.add_argument("--fig-dir", default="figures", help="Directory to save figures")
    args = parser.parse_args()

    df = load_dataframe()
    compute_and_print_models(df, no_plots=args.no_plots, save_figures=args.save_figures, fig_dir=args.fig_dir)


if __name__ == "__main__":
    main()


