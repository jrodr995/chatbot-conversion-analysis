from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from common.data_loader import load_dataframe


def print_correlation_table(title: str, df: pd.DataFrame, target: str, metrics: list[str]) -> None:
    print(f"\n=== {title} ===")
    rows: list[tuple[str, float, float]] = []
    for metric in metrics:
        r, p = pointbiserialr(df[target], df[metric])
        rows.append((metric, r, p))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    for metric, r, p in rows:
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
        print(f"{metric:<40}: r={r:+.3f}, p={p:.3g} {sig}")


def main() -> None:
    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()
    df["has_appt"] = df["HAS_APPT_SCHEDULED"].astype(int)
    df["has_rfi"] = df["HAS_RFI_SUBMISSION"].astype(int)

    agent_metrics = [
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
    ]
    user_duration = ["USER_ENGAGEMENT_DURATION"]

    print_correlation_table("AGENT METRICS vs APPOINTMENTS", df, "has_appt", agent_metrics + user_duration)
    print_correlation_table("AGENT METRICS vs RFI SUBMISSIONS", df, "has_rfi", agent_metrics + user_duration)

    print("\n--- SIGNIFICANCE LEGEND ---")
    print("*** p < 0.001 (highly significant)")
    print("**  p < 0.01  (very significant)")
    print("*   p < 0.05  (significant)")
    print("    p >= 0.05 (not significant)")


if __name__ == "__main__":
    main()


