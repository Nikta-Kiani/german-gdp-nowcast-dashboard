"""Forecast-comparison statistics used by the Significance & calibration tab.

Vendored (not imported) from the thesis pipeline's ``nowcast_utils`` module so
this repository has no dependency on ``statsmodels``/``scikit-learn`` — those
are only needed to *estimate* the models, not to compare their saved forecast
errors. Kept numerically identical to the source implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def _subset_eval_window(
    nowcast_df: pd.DataFrame,
    eval_start: str | None = None,
    eval_end: str | None = None,
    month_in_quarter: int | None = None,
) -> pd.DataFrame:
    """Restrict a nowcast DataFrame to the evaluation window (and optional M1/M2/M3)."""
    sub = nowcast_df.copy()
    qcol = "quarter" if "quarter" in sub.columns else None
    if qcol is not None:
        if eval_start is not None:
            sub = sub.loc[sub[qcol] >= eval_start]
        if eval_end is not None:
            sub = sub.loc[sub[qcol] <= eval_end]
    else:
        if eval_start is not None:
            sub = sub.loc[sub.index >= eval_start]
        if eval_end is not None:
            sub = sub.loc[sub.index <= eval_end]
    if month_in_quarter is not None and "month_in_quarter" in sub.columns:
        sub = sub.loc[sub["month_in_quarter"] == month_in_quarter]
    return sub


def _ensure_quarter_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a string ``quarter`` column (never rely on duplicate index)."""
    out = df.copy()
    if "quarter" not in out.columns:
        out = out.reset_index()
        if "quarter" not in out.columns and out.index.name:
            out = out.rename(columns={out.index.name: "quarter"})
    out["quarter"] = out["quarter"].astype(str)
    return out


def align_forecast_errors(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    month_in_quarter: int | None = 3,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired forecast errors on a common quarter grid for DM tests.

    Always merges on the string quarter key (one row per target quarter at the
    requested ``month_in_quarter``). This avoids accidental row-order alignment
    when DFM DataFrames carry duplicate quarter index labels (M1/M2/M3).
    """
    a = _subset_eval_window(
        df_a, eval_start, eval_end, month_in_quarter=month_in_quarter
    )
    b = _subset_eval_window(
        df_b, eval_start, eval_end, month_in_quarter=month_in_quarter
    )
    a = _ensure_quarter_column(a)
    b = _ensure_quarter_column(b)

    merged = a[["quarter", "error"]].merge(
        b[["quarter", "error"]],
        on="quarter",
        suffixes=("_a", "_b"),
        how="inner",
    )
    ea = merged["error_a"].to_numpy(dtype=float)
    eb = merged["error_b"].to_numpy(dtype=float)
    mask = ~(np.isnan(ea) | np.isnan(eb))
    return ea[mask], eb[mask]


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
    loss: str = "se",
) -> dict[str, float]:
    """Two-sided test of equal predictive accuracy.

    Tests H0: E[L(e_a) - L(e_b)] = 0. Negative DM statistic means model A is
    more accurate. Uses the Harvey-Leybourne-Newbold (1997) small-sample
    correction. ``loss`` in {'se', 'ae'} for squared / absolute error.

    References
    ----------
    Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy.
        Journal of Business and Economic Statistics, 13(3), 253-263.
    Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of
        prediction mean squared errors. International Journal of Forecasting,
        13(2), 281-291.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = a.shape[0]
    if n < 8:
        return {"DM": np.nan, "p_value": np.nan, "n": n}

    if loss == "se":
        d = a ** 2 - b ** 2
    elif loss == "ae":
        d = np.abs(a) - np.abs(b)
    else:
        raise ValueError("loss must be 'se' or 'ae'")

    d_mean = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=0))
    var_d = gamma_0
    for k in range(1, h):
        cov = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
        var_d += 2.0 * cov
    if var_d <= 0:
        return {"DM": np.nan, "p_value": np.nan, "n": n}
    dm = d_mean / np.sqrt(var_d / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * (1 - _st.t.cdf(abs(dm_hln), df=n - 1))
    return {"DM": float(dm_hln), "p_value": float(p), "n": int(n)}
