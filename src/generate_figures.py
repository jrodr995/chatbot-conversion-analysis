from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.data_loader import load_dataframe


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_lr(df: pd.DataFrame, target_col: str, feature_cols: Sequence[str]) -> tuple[np.ndarray, LogisticRegression]:
    X = df[list(feature_cols)]
    y = df[target_col].astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return np.abs(model.coef_[0]), model


def plot_feature_importance(df: pd.DataFrame, out_dir: str) -> None:
    features = [
        "TOTAL_USER_MESSAGES",
        "AVG_USER_WORDS_PER_MSG",
        "MAX_USER_WORDS_IN_MSG",
        "USER_ENGAGEMENT_DURATION",
        "MESSAGE_COUNT",
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
    ]
    imp_appt, _ = train_lr(df, "HAS_APPT_SCHEDULED", features)
    imp_rfi, _ = train_lr(df, "HAS_RFI_SUBMISSION", features)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(features))
    width = 0.38
    ax.bar(x - width / 2, imp_appt, width, label="Appointments")
    ax.bar(x + width / 2, imp_rfi, width, label="RFI")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("|Coefficient|")
    ax.set_title("Feature Importance: Appointments vs RFI (Logistic Regression)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance_appt_vs_rfi.png"), dpi=160)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # User Engagement Duration
    ax = axes[0, 0]
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 1]["USER_ENGAGEMENT_DURATION"], bins=40, color="green", alpha=0.7, ax=ax, label="Appt=1")
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 0]["USER_ENGAGEMENT_DURATION"], bins=40, color="red", alpha=0.5, ax=ax, label="Appt=0")
    ax.set_title("User Engagement Duration - Appointments")
    ax.legend()

    ax = axes[0, 1]
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 1]["USER_ENGAGEMENT_DURATION"], bins=40, color="blue", alpha=0.7, ax=ax, label="RFI=1")
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 0]["USER_ENGAGEMENT_DURATION"], bins=40, color="orange", alpha=0.5, ax=ax, label="RFI=0")
    ax.set_title("User Engagement Duration - RFI")
    ax.legend()

    # Agent Message Count
    ax = axes[0, 2]
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 1]["TOTAL_AGENT_MESSAGES"], bins=40, color="green", alpha=0.7, ax=ax, label="Appt=1")
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 0]["TOTAL_AGENT_MESSAGES"], bins=40, color="red", alpha=0.5, ax=ax, label="Appt=0")
    ax.set_title("Agent Message Count - Appointments")
    ax.legend()

    ax = axes[1, 0]
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 1]["TOTAL_AGENT_MESSAGES"], bins=40, color="blue", alpha=0.7, ax=ax, label="RFI=1")
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 0]["TOTAL_AGENT_MESSAGES"], bins=40, color="orange", alpha=0.5, ax=ax, label="RFI=0")
    ax.set_title("Agent Message Count - RFI")
    ax.legend()

    # Avg User Words per Message
    ax = axes[1, 1]
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 1]["AVG_USER_WORDS_PER_MSG"], bins=40, color="green", alpha=0.7, ax=ax, label="Appt=1")
    sns.histplot(df[df["HAS_APPT_SCHEDULED"] == 0]["AVG_USER_WORDS_PER_MSG"], bins=40, color="red", alpha=0.5, ax=ax, label="Appt=0")
    ax.set_title("Avg User Words - Appointments")
    ax.legend()

    ax = axes[1, 2]
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 1]["AVG_USER_WORDS_PER_MSG"], bins=40, color="blue", alpha=0.7, ax=ax, label="RFI=1")
    sns.histplot(df[df["HAS_RFI_SUBMISSION"] == 0]["AVG_USER_WORDS_PER_MSG"], bins=40, color="orange", alpha=0.5, ax=ax, label="RFI=0")
    ax.set_title("Avg User Words - RFI")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distributions_grid.png"), dpi=160)
    plt.close(fig)


def plot_pattern_rates(df: pd.DataFrame, out_dir: str) -> None:
    exp = df[df.get("EXPLICIT_APPT_REQUEST", 0) == 1].copy()
    if exp.empty:
        return
    grp = (
        exp.groupby("SEQUENCE_PATTERN")["HAS_APPT_SCHEDULED"]
        .agg([("count", "size"), ("rate", "mean")])
        .sort_values("rate", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=grp, x="SEQUENCE_PATTERN", y="rate", ax=ax, color="#4c78a8")
    ax2 = ax.twinx()
    sns.lineplot(data=grp, x="SEQUENCE_PATTERN", y="count", ax=ax2, color="#f58518", marker="o")
    ax.set_ylabel("Appointment Rate")
    ax2.set_ylabel("Volume")
    ax.set_xlabel("")
    ax.set_title("Explicit Appointment Requests: Appointment Rate by Agent Pattern")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pattern_rates_and_volume.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    out_dir = os.path.abspath(out_dir)
    ensure_dir(out_dir)
    df = load_dataframe()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    plot_feature_importance(df, out_dir)
    plot_distributions(df, out_dir)
    plot_pattern_rates(df, out_dir)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()


