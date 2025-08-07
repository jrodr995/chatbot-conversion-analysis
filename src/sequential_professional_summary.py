from __future__ import annotations

import numpy as np
import pandas as pd

from common.data_loader import load_dataframe


def main() -> None:
    # This is a placeholder summary that operates on available columns without proprietary patterns.
    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()

    print("=" * 80)
    print("SEQUENTIAL BEHAVIOR SUMMARY (Sanitized)")
    print("=" * 80)

    print("Dataset overview")
    print(f"Rows: {len(df):,}")
    print("Available fields: TOTAL_USER_MESSAGES, USER_ENGAGEMENT_DURATION, TOTAL_AGENT_MESSAGES, outcomes")

    # Simple proxy ordering by activity and duration
    df = df.assign(
        activity_score=df["TOTAL_USER_MESSAGES"] + df["TOTAL_AGENT_MESSAGES"] * 0.8,
    )

    print("\nTop conversations by activity score (head)")
    cols = ["ID", "TOTAL_USER_MESSAGES", "TOTAL_AGENT_MESSAGES", "USER_ENGAGEMENT_DURATION", "HAS_APPT_SCHEDULED", "HAS_RFI_SUBMISSION", "activity_score"]
    print(df[cols].sort_values("activity_score", ascending=False).head(10).to_string(index=False))

    print("\nBasic outcome rates by activity terciles")
    terciles = pd.qcut(df["activity_score"], 3, labels=["low", "mid", "high"]).rename("activity_band")
    grp = df.join(terciles).groupby("activity_band")[
        ["HAS_APPT_SCHEDULED", "HAS_RFI_SUBMISSION"]
    ].mean()
    print((grp * 100).round(1).to_string())

    print("\nEnd of summary")


if __name__ == "__main__":
    main()


