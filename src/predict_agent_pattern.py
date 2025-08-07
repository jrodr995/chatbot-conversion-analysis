from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from common.data_loader import load_dataframe


def main() -> None:
    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()
    exp = df[df.get("EXPLICIT_APPT_REQUEST", 0) == 1].copy()
    if exp.empty:
        print("No explicit appointment requests in dataset; nothing to predict.")
        return

    # Features: messaging and duration stats (no proprietary text)
    feature_cols = [
        "TOTAL_USER_MESSAGES",
        "AVG_USER_WORDS_PER_MSG",
        "MAX_USER_WORDS_IN_MSG",
        "USER_ENGAGEMENT_DURATION",
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
        "MESSAGE_COUNT",
    ]
    X = exp[feature_cols]
    y = exp["SEQUENCE_PATTERN"].astype(str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Multiclass logistic regression (one-vs-rest)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="ovr")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== Multiclass Logistic Regression: Predict Agent Pattern ===")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()


