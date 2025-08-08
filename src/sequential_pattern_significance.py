from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common.data_loader import load_dataframe


def chi_square_by_pattern(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df.get("EXPLICIT_APPT_REQUEST", 0) == 1].copy()
    if exp.empty:
        return pd.DataFrame()

    # Build contingency table: pattern x outcome (appointment yes/no)
    table = pd.crosstab(exp["SEQUENCE_PATTERN"], exp["HAS_APPT_SCHEDULED"])
    chi2, p, dof, expected = chi2_contingency(table)

    out = table.copy()
    out.columns = ["APPT_0", "APPT_1"] if 0 in out.columns else list(out.columns)
    out["APPT_RATE"] = out.get("APPT_1", 0) / (out.sum(axis=1) + 1e-9)
    out = out.sort_values("APPT_RATE", ascending=False)
    out.loc["__GLOBAL__", "APPT_0"] = table.sum(axis=0)[0] if 0 in table.columns else 0
    out.loc["__GLOBAL__", "APPT_1"] = table.sum(axis=0)[1] if 1 in table.columns else 0
    out.loc["__GLOBAL__", "APPT_RATE"] = out["APPT_1"].sum() / (
        out[["APPT_0", "APPT_1"]].sum().sum() + 1e-9
    )
    out.attrs["chi2"] = chi2
    out.attrs["p_value"] = p
    out.attrs["dof"] = dof
    return out


def main() -> None:
    df = load_dataframe().replace([np.inf, -np.inf], np.nan).dropna()
    res = chi_square_by_pattern(df)
    if res.empty:
        print("No explicit appointment requests found in dataset.")
        return
    print("=== PATTERN vs APPOINTMENT: Chi-square Test ===")
    print(res.to_string())
    print(
        f"\nChi2: {res.attrs['chi2']:.3f}, dof: {res.attrs['dof']}, p-value: {res.attrs['p_value']:.3g}"
    )
    print(
        "Legend: Lower p-value indicates significant dependence between pattern and appointment outcome."
    )


if __name__ == "__main__":
    main()
