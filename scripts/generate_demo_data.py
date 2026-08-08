#!/usr/bin/env python3
"""Generate the synthetic demo dataset shipped with this repository.

Every number produced here is fabricated with a seeded random-number
generator — there is no real thesis data anywhere in this script or its
output. The point is schema-fidelity: each file has the exact columns and
dtypes the dashboard's data layer expects (verified against the original
research pipeline), so the app renders every page and never crashes on a
clean clone, while the values themselves carry no information about the
licensed source data behind the real analysis.

Usage:
    python scripts/generate_demo_data.py

Writes into ``data/demo/`` (same layout ``config.py`` expects). Safe to rerun;
it always overwrites its own output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dashboard import config as C  # noqa: E402
from dashboard.stats import align_forecast_errors, diebold_mariano_test  # noqa: E402

RNG = np.random.default_rng(20260808)

REAL_CATEGORIES = [c for c in C.CATEGORY_ORDER if c not in ("Offset", "Baseline")]
# Mirrors the thesis panel's real composition: survey data dominates the
# candidate universe, hard real-activity data is a minority (see the
# dashboard's own "Emphasis over time" callout).
CATEGORY_WEIGHTS = {
    "Surveys": 0.58, "Orders": 0.06, "Turnover": 0.05, "Production": 0.07,
    "Construction": 0.03, "Trade": 0.05, "Prices": 0.04, "Commodities": 0.03,
    "Financial": 0.03, "Global": 0.02, "Misc": 0.04,
}
CATEGORY_PREFIX = {
    "Surveys": "srv", "Orders": "ord", "Turnover": "trn", "Production": "prd",
    "Construction": "con", "Trade": "trd", "Prices": "prc", "Commodities": "cmd",
    "Financial": "fin", "Global": "glb", "Misc": "msc",
}
N_SERIES = 585

QUARTERS = pd.period_range("1991Q1", "2025Q4", freq="Q")
EVAL_QUARTERS = pd.period_range("2011Q1", "2025Q4", freq="Q")
MONTHLY_ORIGINS_SEL = pd.period_range("2011-01", "2025-12", freq="M")
MONTHLY_ORIGINS_CONTRIB = pd.period_range("2017-01", "2025-12", freq="M")


# --------------------------------------------------------------------------- #
# Synthetic universe
# --------------------------------------------------------------------------- #
def build_universe() -> pd.DataFrame:
    """585 fabricated series ids spread across the 11 economic categories."""
    counts = {
        cat: max(1, round(N_SERIES * w)) for cat, w in CATEGORY_WEIGHTS.items()
    }
    # Fix rounding drift so counts sum exactly to N_SERIES.
    drift = N_SERIES - sum(counts.values())
    counts["Surveys"] += drift

    rows = []
    counter = 0
    for cat in REAL_CATEGORIES:
        prefix = CATEGORY_PREFIX[cat]
        for _ in range(counts[cat]):
            counter += 1
            sid = f"{prefix}{counter:04d}"
            rows.append({
                "id": sid,
                "name": f"Synthetic {cat.lower()} indicator #{counter}",
                "category": cat,
            })
    return pd.DataFrame(rows)


def regime_of(quarter: pd.Period) -> str:
    for name, (start, end) in C.REGIMES.items():
        if pd.Period(start, "Q") <= quarter <= pd.Period(end, "Q"):
            return name
    return "pre-COVID"


# --------------------------------------------------------------------------- #
# GDP target
# --------------------------------------------------------------------------- #
def build_gdp_actuals() -> pd.Series:
    """Fabricated quarterly QoQ growth, with a COVID-shaped shock."""
    values = RNG.normal(0.30, 0.45, size=len(QUARTERS))
    covid_shape = {"2020Q1": -2.0, "2020Q2": -9.5, "2020Q3": 8.7, "2020Q4": 0.6}
    out = pd.Series(values, index=QUARTERS, dtype=float)
    for q, v in covid_shape.items():
        if pd.Period(q, "Q") in out.index:
            out.loc[pd.Period(q, "Q")] = v
    return out.round(3)


def write_gdp_target(gdp: pd.Series) -> None:
    df = pd.DataFrame({
        "quarter": [str(q) for q in gdp.index],
        "gdp_qoq_log_growth_first_release": gdp.values,
    })
    path = C.GDP_TARGET_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Selection matrices
# --------------------------------------------------------------------------- #
def _sample_selection(series_ids: list[str], cat_map: dict[str, str],
                       n_target: int, hard_tilt: float) -> np.ndarray:
    weights = np.array([
        3.0 if cat_map[s] in C.HARD_CATEGORIES else
        (0.6 if cat_map[s] in C.SOFT_CATEGORIES else 1.0)
        for s in series_ids
    ])
    weights = weights ** hard_tilt
    weights = weights / weights.sum()
    chosen = RNG.choice(len(series_ids), size=min(n_target, len(series_ids)),
                        replace=False, p=weights)
    mask = np.zeros(len(series_ids), dtype=int)
    mask[chosen] = 1
    return mask


def build_selection_matrix(universe: pd.DataFrame, n_target: int,
                           hard_tilt: float, ensure_per_category: bool = False
                           ) -> pd.DataFrame:
    series_ids = universe["id"].tolist()
    cat_map = dict(zip(universe["id"], universe["category"]))
    rows = {}
    for origin in MONTHLY_ORIGINS_SEL:
        mask = _sample_selection(series_ids, cat_map, n_target, hard_tilt)
        if ensure_per_category:
            by_cat: dict[str, list[int]] = {}
            for i, s in enumerate(series_ids):
                by_cat.setdefault(cat_map[s], []).append(i)
            for idxs in by_cat.values():
                if not any(mask[i] for i in idxs):
                    mask[RNG.choice(idxs)] = 1
        rows[str(origin)] = mask
    mat = pd.DataFrame.from_dict(rows, orient="index", columns=series_ids)
    mat.index.name = "forecast_origin"
    return mat


def write_selection_matrices(universe: pd.DataFrame) -> None:
    en = build_selection_matrix(universe, n_target=55, hard_tilt=1.4)
    bb = build_selection_matrix(universe, n_target=20, hard_tilt=1.1,
                                ensure_per_category=True)
    pls = build_selection_matrix(universe, n_target=30, hard_tilt=0.7)

    C.EN_SELECTION_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    en.to_csv(C.EN_SELECTION_MATRIX_CSV)
    C.SELECTION_MATRICES["Block-balanced (k=20)"].parent.mkdir(parents=True, exist_ok=True)
    bb.to_csv(C.SELECTION_MATRICES["Block-balanced (k=20)"])
    pls.to_csv(C.SELECTION_MATRICES["PLS"])


# --------------------------------------------------------------------------- #
# ifoCAST fixed expert set
# --------------------------------------------------------------------------- #
def write_ifocast_mapping(universe: pd.DataFrame) -> list[str]:
    picks = universe.sample(n=20, random_state=7).reset_index(drop=True)
    rows = []
    for i, r in picks.iterrows():
        in_panel = i != 0  # one deliberately "unmapped" row, like the real set
        rows.append({
            "ifoCAST_indicator": f"ifo_expert_{i + 1:02d}",
            "ifoCAST_group": r["category"],
            "my_id": r["id"],
            "my_name": r["name"],
            "my_category": r["category"],
            "pub_lag": float(RNG.integers(0, 3)),
            "match": "exact",
            "in_panel": in_panel,
            "note": "" if in_panel else "mapped but absent from transformed panel",
        })
    rows.append({
        "ifoCAST_indicator": "gdp_target", "ifoCAST_group": "Target",
        "my_id": "(target)", "my_name": "German GDP (QoQ growth)",
        "my_category": "Target", "pub_lag": np.nan, "match": "exact",
        "in_panel": False, "note": "target series, excluded from predictor set",
    })
    df = pd.DataFrame(rows)
    C.IFOCAST_MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.IFOCAST_MAPPING_CSV, index=False)
    return [r["my_id"] for r in rows if r["in_panel"]]


# --------------------------------------------------------------------------- #
# Nowcast model result files
# --------------------------------------------------------------------------- #
def _monthly_origin(quarter: pd.Period, miq: int) -> str:
    month = (quarter.month - 1) + miq  # quarter.month is the first month
    year = quarter.year
    return f"{year:04d}-{month:02d}"


def build_model_frame(key: str, gdp: pd.Series, noise_scale: float,
                      miq_shrink: float = 0.55) -> pd.DataFrame:
    spec = C.MODELS[key]
    quarters = EVAL_QUARTERS
    rows = []
    for q in quarters:
        actual = float(gdp.loc[q])
        if spec.has_miq:
            for miq in (1, 2, 3):
                shrink = miq_shrink ** (miq - 1)
                nowcast = actual + RNG.normal(0, noise_scale * shrink)
                rows.append({
                    "monthly_origin": _monthly_origin(q, miq),
                    "quarter": str(q),
                    "month_in_quarter": miq,
                    "nowcast": round(nowcast, 4),
                    "actual": round(actual, 4),
                    "error": round(nowcast - actual, 4),
                })
        else:
            nowcast = actual + RNG.normal(0, noise_scale)
            rows.append({
                "quarter": str(q),
                "nowcast": round(nowcast, 4),
                "actual": round(actual, 4),
                "error": round(nowcast - actual, 4),
            })
    return pd.DataFrame(rows)


MODEL_NOISE = {
    "RW": 1.55, "AR1": 1.35,
    "DFM-ifoCAST": 0.92, "DFM-EN": 0.88, "DFM-PLS": 0.98,
    "DFM-BlockBalanced": 0.90, "DFM-TVP": 0.85, "DFM-SV-k2": 0.86,
    "combo_equal": 0.80, "XGB-Full": 1.05, "MLP-Factor": 1.00,
}


def write_model_files(gdp: pd.Series) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key in C.MODEL_ORDER:
        spec = C.MODELS[key]
        df = build_model_frame(key, gdp, MODEL_NOISE[key])

        if spec.has_miq and key not in ("combo_equal",):
            df.insert(3, "n_indicators", RNG.integers(15, 60, size=len(df)))
            df.insert(4, "k_factors", 2)
        if key == "DFM-BlockBalanced":
            df = df.drop(columns=["k_factors"])
            df.insert(4, "fit_tag", "ok")
        if key == "DFM-TVP":
            df["q_ratio"] = RNG.uniform(0.5, 1.5, size=len(df)).round(3)
            df["tvp_intercept"] = RNG.normal(0.2, 0.1, size=len(df)).round(4)
            df["ci_lower_90"] = df["nowcast"] - RNG.uniform(0.8, 1.6, size=len(df))
            df["ci_upper_90"] = df["nowcast"] + RNG.uniform(0.8, 1.6, size=len(df))
            df["tvp_loading_1"] = RNG.normal(0.6, 0.15, size=len(df)).round(4)
            df["tvp_loading_2"] = RNG.normal(0.3, 0.15, size=len(df)).round(4)
        if key == "DFM-SV-k2":
            df["nowcast_baseline"] = (df["nowcast"] + RNG.normal(0, 0.1, len(df))).round(4)
            df["point_shift"] = (df["nowcast"] - df["nowcast_baseline"]).round(4)
            df["ci_lower_90"] = df["nowcast"] - RNG.uniform(0.7, 1.5, size=len(df))
            df["ci_upper_90"] = df["nowcast"] + RNG.uniform(0.7, 1.5, size=len(df))
            df["rel_vol"] = RNG.uniform(0.7, 1.6, size=len(df)).round(3)
            df["rel_vol_target"] = 1.0
            df["sigma_em"] = RNG.uniform(0.5, 1.0, size=len(df)).round(3)
        if key == "combo_equal":
            df.insert(1, "month_in_quarter", df.pop("month_in_quarter"))
        if key == "XGB-Full":
            df.insert(1, "monthly_origin", df.pop("monthly_origin"))
            df["n_features"] = RNG.integers(500, 620, size=len(df))
            df["n_features_pre"] = RNG.integers(1700, 1800, size=len(df))
        if key == "MLP-Factor":
            df.insert(1, "monthly_origin", df.pop("monthly_origin"))
            df["n_features"] = 6

        spec.file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(spec.file, index=False)
        frames[key] = df
    return frames


# --------------------------------------------------------------------------- #
# Aggregate accuracy / significance tables (computed from the generated
# model frames so the numbers are internally consistent, not double-random)
# --------------------------------------------------------------------------- #
def _m3(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["month_in_quarter"] == 3] if "month_in_quarter" in df.columns else df


def write_rmsfe_table(frames: dict[str, pd.DataFrame]) -> None:
    ar1_err = _m3(frames["AR1"])["error"].to_numpy(dtype=float)
    rmsfe_ar1 = float(np.sqrt(np.mean(ar1_err ** 2)))
    rows = []
    for key, df in frames.items():
        err = _m3(df)["error"].to_numpy(dtype=float)
        rmsfe = float(np.sqrt(np.mean(err ** 2)))
        std_actual = float(_m3(df)["actual"].std())
        if key == "AR1":
            p = np.nan
        else:
            ea, eb = align_forecast_errors(df, frames["AR1"], month_in_quarter=3)
            p = diebold_mariano_test(ea, eb)["p_value"]
        rows.append({
            "model": key, "RMSFE_M3": round(rmsfe, 4),
            "NSR": round(rmsfe / std_actual, 4) if std_actual else np.nan,
            "vs_AR1": round(rmsfe / rmsfe_ar1, 4),
            "DM_p_vs_AR1": round(p, 4) if pd.notna(p) else np.nan,
        })
    pd.DataFrame(rows).to_csv(C.RMSFE_ALL_CSV, index=False)


def write_horizon_tables(frames: dict[str, pd.DataFrame]) -> None:
    profile_rows, bv_rows = [], []
    for key, df in frames.items():
        if "month_in_quarter" not in df.columns:
            continue
        work = df.copy()
        work["quarter_p"] = pd.PeriodIndex(work["quarter"], freq="Q")
        work["regime"] = work["quarter_p"].map(regime_of)
        for regime in C.REGIMES:
            for miq in (1, 2, 3):
                sub = work.loc[(work["regime"] == regime) & (work["month_in_quarter"] == miq)]
                if sub.empty:
                    continue
                err = sub["error"].to_numpy(dtype=float)
                rmsfe = float(np.sqrt(np.mean(err ** 2)))
                bias = float(np.mean(err))
                variance = float(np.var(err))
                bias_sq = bias ** 2
                profile_rows.append({"model": key, "regime": regime,
                                     "month_in_quarter": miq, "RMSFE": round(rmsfe, 4)})
                bv_rows.append({
                    "model": key, "regime": regime, "month_in_quarter": miq,
                    "n": int(len(sub)), "bias": round(bias, 4),
                    "bias_sq": round(bias_sq, 4), "variance": round(variance, 4),
                    "RMSFE": round(rmsfe, 4),
                    "bias_sq_share_pct": round(100 * bias_sq / rmsfe ** 2, 2) if rmsfe else np.nan,
                })
    pd.DataFrame(profile_rows).to_csv(C.HORIZON_PROFILE_CSV, index=False)
    pd.DataFrame(bv_rows).to_csv(C.HORIZON_BIAS_VARIANCE_CSV, index=False)


def write_post_covid_table(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for key, df in frames.items():
        m3 = _m3(df).copy()
        m3["quarter_p"] = pd.PeriodIndex(m3["quarter"], freq="Q")
        m3["regime"] = m3["quarter_p"].map(regime_of)
        row = {"model": key}
        for regime in C.REGIMES:
            sub = m3.loc[m3["regime"] == regime, "error"].to_numpy(dtype=float)
            row[f"{regime}_rmsfe"] = round(float(np.sqrt(np.mean(sub ** 2))), 4) if len(sub) else np.nan
            row[f"{regime}_bias"] = round(float(np.mean(sub)), 4) if len(sub) else np.nan
        all_err = m3["error"].to_numpy(dtype=float)
        row["all_rmsfe"] = round(float(np.sqrt(np.mean(all_err ** 2))), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.POST_COVID_CSV, index=False)


def write_dm_matrix(frames: dict[str, pd.DataFrame]) -> None:
    keys = [k for k in C.MODEL_ORDER if k in frames]
    mat = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            if a == b:
                continue
            ea, eb = align_forecast_errors(frames[a], frames[b], month_in_quarter=3)
            mat.loc[a, b] = diebold_mariano_test(ea, eb)["p_value"]
    mat.index.name = "model"
    mat.round(4).to_csv(C.DM_ALL_CSV)


def write_mincer_zarnowitz(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for key, df in frames.items():
        m3 = _m3(df)
        y = m3["actual"].to_numpy(dtype=float)
        x = m3["nowcast"].to_numpy(dtype=float)
        n = len(y)
        X = np.column_stack([np.ones(n), x])
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = beta_hat
        resid = y - X @ beta_hat
        sigma2 = float(np.sum(resid ** 2) / max(n - 2, 1))
        xtx_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
        se_beta = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
        from scipy import stats as _st
        p_alpha_zero = float(2 * (1 - _st.t.cdf(abs(alpha / se_alpha), df=n - 2))) if se_alpha else np.nan
        p_joint = float(2 * (1 - _st.t.cdf(abs((beta - 1) / se_beta), df=n - 2))) if se_beta else np.nan
        rows.append({
            "model": key, "alpha": round(float(alpha), 4), "beta": round(float(beta), 4),
            "se_alpha": round(se_alpha, 4), "se_beta": round(se_beta, 4),
            "p_alpha_zero": round(p_alpha_zero, 4), "p_joint_H0_a0_b1": round(p_joint, 4),
            "n": n,
        })
    pd.DataFrame(rows).to_csv(C.MZ_CSV, index=False)


def write_sv_calibration(frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for key in ("DFM-SV-k2", "DFM-TVP"):
        df = frames[key]
        m3 = _m3(df)
        err = m3["error"].to_numpy(dtype=float)
        rmsfe = float(np.sqrt(np.mean(err ** 2)))
        lo, hi = m3["ci_lower_90"], m3["ci_upper_90"]
        covered = ((m3["actual"] >= lo) & (m3["actual"] <= hi)).mean()
        rows.append({
            "model": key, "RMSFE": round(rmsfe, 4),
            "coverage_empirical": round(float(covered), 4), "coverage_nominal": 0.90,
            "mean_width": round(float((hi - lo).mean()), 4),
            "CRPS": round(rmsfe * 0.55, 4),
        })
    pd.DataFrame(rows).to_csv(C.SV_CALIBRATION_CSV, index=False)


def write_revision_path(frames: dict[str, pd.DataFrame]) -> None:
    df = frames["DFM-EN"]
    piv_e = df.pivot(index="quarter", columns="month_in_quarter", values="error")
    piv_n = df.pivot(index="quarter", columns="month_in_quarter", values="nowcast")
    actual = df.groupby("quarter")["actual"].first()
    out = pd.DataFrame({
        "quarter": piv_e.index,
        "error_M1": piv_e[1].values, "error_M2": piv_e[2].values, "error_M3": piv_e[3].values,
        "nowcast_M1": piv_n[1].values, "nowcast_M2": piv_n[2].values, "nowcast_M3": piv_n[3].values,
        "actual": actual.values,
    })
    out["revision_M1_to_M3"] = out["nowcast_M3"] - out["nowcast_M1"]
    out["abs_revision_M1_to_M3"] = out["revision_M1_to_M3"].abs()
    out.to_csv(C.REVISION_CSV, index=False)


# --------------------------------------------------------------------------- #
# Factor loadings, SHAP, ragged edge, XGB sensitivity
# --------------------------------------------------------------------------- #
def write_factor_loadings() -> None:
    rows = []
    for q in EVAL_QUARTERS:
        origin = f"{q.year:04d}-{q.month + 2:02d}"
        for factor in (1, 2):
            shares = RNG.dirichlet(np.ones(len(C.FACTOR_LOADING_CATEGORIES)))
            for cat, share in zip(C.FACTOR_LOADING_CATEGORIES, shares):
                rows.append({
                    "record_type": "category", "quarter": str(q),
                    "date": pd.Period(q, "Q").to_timestamp(how="end").date().isoformat(),
                    "origin": origin, "n_indicators": int(RNG.integers(20, 60)),
                    "factor": factor, "category": cat, "share": round(float(share), 4),
                    "rank": np.nan, "id": np.nan, "name": np.nan,
                    "loading": np.nan, "abs_loading": np.nan,
                })
    pd.DataFrame(rows).to_csv(C.FACTOR_LOADING_CSV, index=False)


def write_shap(universe: pd.DataFrame) -> None:
    pool = universe.sample(n=120, random_state=3)["id"].tolist()
    rows = []
    for q in EVAL_QUARTERS:
        chosen = RNG.choice(pool, size=60, replace=False)
        importances = RNG.exponential(0.5, size=60)
        for sid, imp, lag in zip(chosen, importances, RNG.integers(0, 3, size=60)):
            rows.append({"quarter": str(q), "feature": f"{sid}__L{lag}",
                        "mean_abs_shap": round(float(imp), 5)})
    pd.DataFrame(rows).to_csv(C.SHAP_CSV, index=False)


def write_ragged_edge(universe: pd.DataFrame) -> None:
    rows = []
    for _, r in universe.iterrows():
        if r["category"] in C.SOFT_CATEGORIES:
            pub_lag = 0
        elif r["category"] in C.HARD_CATEGORIES:
            pub_lag = int(RNG.choice([1, 2], p=[0.4, 0.6]))
        else:
            pub_lag = int(RNG.choice([0, 1, 2]))
        status = "observed" if pub_lag == 0 else "ar_filled"
        rows.append({
            "series": r["id"], "pub_lag": pub_lag,
            "last_obs_month": "2025-12" if pub_lag == 0 else "2025-11",
            "n_obs_q": 3 - pub_lag if pub_lag < 3 else 0,
            "n_filled_q": pub_lag, "n_missing_q": 0, "status": status,
        })
    C.RAGGED_EDGE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.RAGGED_EDGE_CSV, index=False)


def write_xgb_sensitivity() -> None:
    runs = [f"seed_{s}" for s in (42, 1, 2, 3, 4)] + \
           ["max_depth_4", "max_depth_8", "n_estimators_300", "learning_rate_low"]
    rows = []
    for run in runs:
        rows.append({
            "run": run, "seed": 42 if not run.startswith("seed_") else int(run.split("_")[1]),
            "cv_rmse": round(float(RNG.uniform(0.9, 1.2)), 4),
            "rmsfe_pre": round(float(RNG.uniform(0.8, 1.1)), 4),
            "rmsfe_COVID": round(float(RNG.uniform(2.0, 3.5)), 4),
            "rmsfe_post": round(float(RNG.uniform(0.9, 1.3)), 4),
            "rmsfe_full": round(float(RNG.uniform(1.0, 1.4)), 4),
            "param_subsample": 0.8, "param_reg_lambda": 1.0, "param_reg_alpha": 0.1,
            "param_n_estimators": 300 if "n_estimators_300" not in run else 300,
            "param_min_child_weight": 3, "param_max_depth": 6,
            "param_learning_rate": 0.05, "param_colsample_bytree": 0.8,
        })
    C.XGB_SENSITIVITY_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.XGB_SENSITIVITY_SUMMARY_CSV, index=False)

    post_covid_q = pd.period_range("2022Q1", "2025Q4", freq="Q")
    jk_rows = []
    for q in post_covid_q:
        jk_rows.append({
            "dropped_quarter": str(q), "dropped_error": round(float(RNG.normal(0, 1.0)), 4),
            "rmsfe_excl_quarter": round(float(RNG.uniform(0.85, 1.15)), 4),
            "rmsfe_all_16": round(float(RNG.uniform(0.9, 1.1)), 4),
        })
    pd.DataFrame(jk_rows).to_csv(C.XGB_SENSITIVITY_JACKKNIFE_CSV, index=False)

    C.XGB_SENSITIVITY_DM_TXT.write_text(
        f"DM={float(RNG.uniform(-3, -1)):.3f} p_value={float(RNG.uniform(0.01, 0.09)):.3f} n=16\n"
    )


# --------------------------------------------------------------------------- #
# Category / series contribution parquets
# --------------------------------------------------------------------------- #
def write_contributions(universe: pd.DataFrame, gdp: pd.Series,
                        model_key: str, cat_path: Path, series_path: Path) -> None:
    contrib_categories = C.FACTOR_LOADING_CATEGORIES + ["Prices", "Offset"]
    by_cat = {cat: universe.loc[universe["category"] == cat, "id"].tolist()
             for cat in contrib_categories if cat in universe["category"].unique()}
    by_cat.setdefault("Offset", [])

    cat_rows, series_rows = [], []
    for origin in MONTHLY_ORIGINS_CONTRIB:
        q = origin.asfreq("Q")
        miq = origin.month - (q.month - 1)
        actual = float(gdp.loc[q])
        weights = RNG.dirichlet(np.ones(len(contrib_categories)))
        signs = RNG.choice([-1, 1], size=len(contrib_categories), p=[0.35, 0.65])
        raw = weights * signs
        raw = raw / np.sum(np.abs(raw))  # normalise so components sum in proportion
        nowcast = actual + RNG.normal(0, 0.5)
        contribs = raw * nowcast
        for cat, contrib in zip(contrib_categories, contribs):
            cat_rows.append({
                "monthly_origin": str(origin), "quarter": str(q),
                "month_in_quarter": int(miq), "nowcast": round(nowcast, 4),
                "actual": round(actual, 4), "category": cat,
                "contrib_pp": round(float(contrib), 4),
            })
            members = by_cat.get(cat) or ["__offset__"]
            picks = RNG.choice(members, size=min(4, len(members)), replace=False)
            sub_w = RNG.dirichlet(np.ones(len(picks)))
            for sid, sw in zip(picks, sub_w):
                series_rows.append({
                    "monthly_origin": str(origin), "quarter": str(q),
                    "month_in_quarter": int(miq), "series": sid, "category": cat,
                    "contrib_pp": round(float(contrib * sw), 4),
                })
    cat_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cat_rows).to_parquet(cat_path, index=False)
    pd.DataFrame(series_rows).to_parquet(series_path, index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Writing synthetic demo data to: {C.DATA_ROOT}")
    universe = build_universe()

    metadata_path = C.DATA_DICT_CSV
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(metadata_path, index=False)

    gdp = build_gdp_actuals()
    write_gdp_target(gdp)
    write_selection_matrices(universe)
    write_ifocast_mapping(universe)

    frames = write_model_files(gdp)
    write_rmsfe_table(frames)
    write_horizon_tables(frames)
    write_post_covid_table(frames)
    write_dm_matrix(frames)
    write_mincer_zarnowitz(frames)
    write_sv_calibration(frames)
    write_revision_path(frames)

    write_factor_loadings()
    write_shap(universe)
    write_ragged_edge(universe)
    write_xgb_sensitivity()

    write_contributions(universe, gdp, "DFM-EN", C.CONTRIB_PARQUET, C.SERIES_CONTRIB_PARQUET)
    write_contributions(universe, gdp, "DFM-TVP", C.CONTRIB_PARQUET_TVP, C.SERIES_CONTRIB_PARQUET_TVP)
    write_contributions(universe, gdp, "DFM-BlockBalanced",
                        C.CONTRIB_PARQUET_BLOCKBALANCED, C.SERIES_CONTRIB_PARQUET_BLOCKBALANCED)

    n_files = sum(1 for _ in C.DATA_ROOT.rglob("*") if _.is_file())
    print(f"Done — {n_files} synthetic files written under {C.DATA_ROOT}")


if __name__ == "__main__":
    main()
