from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

from common.data_loader import load_dataframe


def main() -> None:
    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()
    df["has_appt"] = df["HAS_APPT_SCHEDULED"].astype(int)
    df["has_rfi"] = df["HAS_RFI_SUBMISSION"].astype(int)

    metrics = [
        "TOTAL_USER_MESSAGES",
        "AVG_USER_WORDS_PER_MSG",
        "MAX_USER_WORDS_IN_MSG",
        "USER_ENGAGEMENT_DURATION",
        "TOTAL_AGENT_MESSAGES",
        "AVG_AGENT_WORDS_PER_MSG",
        "MAX_AGENT_WORDS_IN_MSG",
        "MESSAGE_COUNT",
    ]

    def corr_table(target: str) -> pd.DataFrame:
        rows = []
        for m in metrics:
            r, p = pointbiserialr(df[target], df[m])
            rows.append({"metric": m, "r": r, "p": p})
        tab = pd.DataFrame(rows).sort_values(by="r", key=lambda s: s.abs(), ascending=False)
        return tab

    print("\n=== APPOINTMENTS: Correlations ===")
    print(corr_table("has_appt").to_string(index=False, formatters={"r": "{:+.3f}".format, "p": "{:.3g}".format}))

    print("\n=== RFI: Correlations ===")
    print(corr_table("has_rfi").to_string(index=False, formatters={"r": "{:+.3f}".format, "p": "{:.3g}".format}))

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05")


if __name__ == "__main__":
    main()


