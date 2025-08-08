from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.data_loader import load_dataframe
from common.metrics_utils import optimize_threshold


def print_header(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def print_section(title: str) -> None:
    print(f"\n{'-'*60}\n{title}\n{'-'*60}")


def main() -> None:
    print_header("PROFESSIONAL ANALYSIS OUTPUT")

    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()

    print_section("Dataset Overview")
    print(f"Rows: {len(df):,}")
    cols = [
        "TOTAL_USER_MESSAGES",
        "AVG_USER_WORDS_PER_MSG",
        "MAX_USER_WORDS_IN_MSG",
        "USER_ENGAGEMENT_DURATION",
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
        "MESSAGE_COUNT",
        "HAS_APPT_SCHEDULED",
        "HAS_RFI_SUBMISSION",
    ]
    print("Columns:", ", ".join(cols))
    appt_rate = df["HAS_APPT_SCHEDULED"].mean()
    rfi_rate = df["HAS_RFI_SUBMISSION"].mean()
    print(f"Appointment prevalence: {appt_rate:.1%}")
    print(f"RFI prevalence: {rfi_rate:.1%}")

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
    print_section("Appointment Model (Logistic Regression, class_weight='balanced')")
    yA = df["has_appt"]
    XA_tr, XA_te, yA_tr, yA_te = train_test_split(X_scaled, yA, test_size=0.2, random_state=42)
    mdlA = LogisticRegression(max_iter=1000, class_weight="balanced")
    mdlA.fit(XA_tr, yA_tr)
    yA_prob = mdlA.predict_proba(XA_te)[:, 1]
    tA, sA = optimize_threshold(yA_te, yA_prob, metric="accuracy")
    yA_pr = (yA_prob >= tA).astype(int)
    print(classification_report(yA_te, yA_pr))
    try:
        print(
            f"Appointments ROC-AUC: {roc_auc_score(yA_te, yA_prob):.3f}; Best threshold={tA:.2f} (acc={sA:.3f})"
        )
    except Exception:
        pass

    # RFI model
    print_section("RFI Model (Logistic Regression, class_weight='balanced')")
    yR = df["has_rfi"]
    XR_tr, XR_te, yR_tr, yR_te = train_test_split(X_scaled, yR, test_size=0.2, random_state=42)
    mdlR = LogisticRegression(max_iter=1000, class_weight="balanced")
    mdlR.fit(XR_tr, yR_tr)
    yR_prob = mdlR.predict_proba(XR_te)[:, 1]
    tR, sR = optimize_threshold(yR_te, yR_prob, metric="accuracy")
    yR_pr = (yR_prob >= tR).astype(int)
    print(classification_report(yR_te, yR_pr))
    try:
        print(
            f"RFI ROC-AUC: {roc_auc_score(yR_te, yR_prob):.3f}; Best threshold={tR:.2f} (acc={sR:.3f})"
        )
    except Exception:
        pass

    # Feature importance (absolute coefficients)
    print_section("Feature Importance (Absolute Coefficients)")
    impA = np.abs(mdlA.coef_[0])
    impR = np.abs(mdlR.coef_[0])
    tbl = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "appointment_importance": impA,
                "rfi_importance": impR,
            }
        )
        .sort_values("appointment_importance", ascending=False)
        .reset_index(drop=True)
    )
    print(tbl.to_string(index=False))

    print_header("END OF REPORT")


if __name__ == "__main__":
    main()
