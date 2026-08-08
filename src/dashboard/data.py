"""Cached data-access layer for the dashboard.

Every reader is wrapped in ``st.cache_data`` so the heavy CSV/parquet loads run
once per session. Nothing here mutates source files — it only reads and reshapes
the existing artefacts (real or demo) into tidy frames the charts consume.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from .stats import align_forecast_errors, diebold_mariano_test

HEADLINE_MIQ = 3


def artifact_mtime() -> float:
    """Latest modification time among dashboard nowcast/selection artefacts."""
    paths: list[Path] = [
        C.GDP_TARGET_CSV,
        C.DATA_DICT_CSV,
        C.RMSFE_ALL_CSV,
        C.HORIZON_PROFILE_CSV,
        C.POST_COVID_CSV,
        C.DM_ALL_CSV,
        C.MZ_CSV,
        C.SV_CALIBRATION_CSV,
        C.REVISION_CSV,
        C.CONTRIB_PARQUET,
        C.SERIES_CONTRIB_PARQUET,
        C.CONTRIB_PARQUET_TVP,
        C.SERIES_CONTRIB_PARQUET_TVP,
        C.CONTRIB_PARQUET_BLOCKBALANCED,
        C.SERIES_CONTRIB_PARQUET_BLOCKBALANCED,
        C.FACTOR_LOADING_CSV,
        C.TVP_RESULTS_CSV,
        C.SHAP_CSV,
        C.IFOCAST_MAPPING_CSV,
        C.RAGGED_EDGE_CSV,
        C.XGB_SENSITIVITY_SUMMARY_CSV,
        C.XGB_SENSITIVITY_JACKKNIFE_CSV,
        C.XGB_SENSITIVITY_DM_TXT,
        *[spec.file for spec in C.MODELS.values()],
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else 0.0


def refresh_cache_if_stale() -> None:
    """Clear ``st.cache_data`` when underlying CSV/parquet files change."""
    mtime = artifact_mtime()
    if st.session_state.get("_artifact_mtime") != mtime:
        st.cache_data.clear()
        st.session_state["_artifact_mtime"] = mtime


# --------------------------------------------------------------------------- #
# Metadata & category mapping
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_category_map() -> dict[str, str]:
    """Map every series id -> economic category from the enriched data dict."""
    md = pd.read_csv(C.DATA_DICT_CSV)
    md = md.drop_duplicates(subset="id")
    return dict(zip(md["id"].astype(str), md["category"].astype(str)))


_SUFFIX = re.compile(r"__L\d+$")


def feature_to_id(feature: str) -> str:
    """Strip lag/transform suffixes (e.g. ``deprod1404__L0`` -> ``deprod1404``)."""
    return _SUFFIX.sub("", str(feature))


def categorize(ids, cat_map: dict[str, str]) -> pd.Series:
    base = pd.Index([feature_to_id(x) for x in ids])
    return pd.Series([cat_map.get(b, "Misc") for b in base], index=range(len(base)))


# --------------------------------------------------------------------------- #
# Quarter helpers
# --------------------------------------------------------------------------- #
def quarter_to_ts(q: pd.Series | pd.Index) -> pd.DatetimeIndex:
    """Quarter string (e.g. ``2020Q1``) -> period-end timestamp."""
    return pd.PeriodIndex(q, freq="Q").to_timestamp(how="end")


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_gdp_target() -> pd.DataFrame:
    df = pd.read_csv(C.GDP_TARGET_CSV)
    df.columns = ["quarter", "gdp"]
    df["date"] = quarter_to_ts(df["quarter"])
    return df


# --------------------------------------------------------------------------- #
# Nowcast results
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_nowcast(key: str) -> pd.DataFrame | None:
    """Load one model's nowcast result frame; None if the file is missing."""
    spec = C.MODELS[key]
    if not spec.file.exists():
        return None
    df = pd.read_csv(spec.file)
    if "quarter" not in df.columns:
        return None
    df["date"] = quarter_to_ts(df["quarter"])
    return df


def m3_slice(df: pd.DataFrame, has_miq: bool) -> pd.DataFrame:
    """Keep one row per quarter: the final (M3) information set when available."""
    if has_miq and "month_in_quarter" in df.columns:
        df = df.loc[df["month_in_quarter"] == 3]
    return df


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    """Headline Part II models (excludes appendix-only comparison specs)."""
    return [
        k for k in C.MODEL_ORDER
        if C.MODELS[k].file.exists()
        and k not in C.APPENDIX_COMPARE_MODELS
    ]


@st.cache_data(show_spinner=False)
def accuracy_models() -> list[str]:
    """Models shown on Accuracy & model paths (includes appendix comparisons)."""
    return [k for k in C.MODEL_ORDER if C.MODELS[k].file.exists()]


@st.cache_data(show_spinner=False)
def rmsfe_by_regime() -> pd.DataFrame:
    """Long frame: model x regime -> M3 RMSFE (+ a 'full' window row)."""
    rows = []
    for key in accuracy_models():
        df = load_nowcast(key)
        if df is None:
            continue
        spec = C.MODELS[key]
        sub = m3_slice(df, spec.has_miq)
        windows = {**C.REGIMES, "full": (C.EVAL_START, C.EVAL_END)}
        for regime, (s, e) in windows.items():
            w = sub.loc[(sub["quarter"] >= s) & (sub["quarter"] <= e)]
            err = w["error"].dropna()
            rmsfe = float(np.sqrt((err ** 2).mean())) if len(err) else np.nan
            rows.append({"model": key, "regime": regime, "rmsfe": rmsfe,
                         "n": int(len(err))})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def full_window_accuracy() -> pd.DataFrame:
    """Full-window M3 accuracy ranking using both RMSE and MAE."""
    rows = []
    for key in accuracy_models():
        df = load_nowcast(key)
        if df is None:
            continue
        spec = C.MODELS[key]
        sub = m3_slice(df, spec.has_miq)
        sub = sub.loc[
            (sub["quarter"] >= C.EVAL_START) & (sub["quarter"] <= C.EVAL_END)
        ]
        err = pd.to_numeric(sub["error"], errors="coerce").dropna()
        if err.empty:
            continue
        rows.append({
            "model": key,
            "family": spec.family,
            "n": int(len(err)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(err.abs().mean()),
            "bias": float(err.mean()),
        })

    cols = ["model", "family", "n", "rmse", "mae", "bias",
            "rmse_rank", "mae_rank"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["rmse_rank"] = out["rmse"].rank(method="min").astype(int)
    out["mae_rank"] = out["mae"].rank(method="min").astype(int)
    return out.sort_values(["rmse_rank", "mae_rank", "model"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def nowcast_timeseries(keys: tuple[str, ...]) -> pd.DataFrame:
    """Tidy frame (date, quarter, model, nowcast) at the M3 information set."""
    frames = []
    for key in keys:
        df = load_nowcast(key)
        if df is None:
            continue
        spec = C.MODELS[key]
        sub = m3_slice(df, spec.has_miq).copy()
        sub = sub[["quarter", "date", "nowcast"]].copy()
        sub["model"] = key
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["quarter", "date", "nowcast", "model"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Summary tables
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_rmsfe_table() -> pd.DataFrame:
    return pd.read_csv(C.RMSFE_ALL_CSV)


@st.cache_data(show_spinner=False)
def load_horizon_profile() -> pd.DataFrame:
    return pd.read_csv(C.HORIZON_PROFILE_CSV)


@st.cache_data(show_spinner=False)
def load_horizon_bias_variance() -> pd.DataFrame:
    """Bias^2 / variance decomposition of the M1-M3 RMSFE, per model x regime."""
    if not C.HORIZON_BIAS_VARIANCE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(C.HORIZON_BIAS_VARIANCE_CSV)


@st.cache_data(show_spinner=False)
def load_post_covid() -> pd.DataFrame:
    return pd.read_csv(C.POST_COVID_CSV)


@st.cache_data(show_spinner=False)
def load_dm_matrix() -> pd.DataFrame:
    """Full-sample pairwise DM matrix (legacy CSV artefact)."""
    df = pd.read_csv(C.DM_ALL_CSV, index_col=0)
    return df


def _dm_pvalue(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    eval_start: str,
    eval_end: str,
) -> float:
    ea, eb = align_forecast_errors(
        df_a, df_b,
        month_in_quarter=HEADLINE_MIQ,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    return float(diebold_mariano_test(ea, eb)["p_value"])


@st.cache_data(show_spinner=False)
def dm_matrix_by_regime(regime: str) -> pd.DataFrame:
    """Symmetric pairwise HLN-corrected DM p-values for one evaluation regime."""
    if regime not in C.REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    start, end = C.REGIMES[regime]
    models = available_models()
    idx = pd.Index(models, name="model")
    mat = pd.DataFrame(np.nan, index=idx, columns=idx.copy())
    loaded = {k: load_nowcast(k) for k in models}
    for a, b in combinations(models, 2):
        if loaded[a] is None or loaded[b] is None:
            continue
        p = _dm_pvalue(loaded[a], loaded[b], start, end)
        mat.loc[a, b] = p
        mat.loc[b, a] = p
    return mat.round(3)


@st.cache_data(show_spinner=False)
def dm_regime_n(regime: str) -> int:
    """Number of M3 evaluation quarters in a regime window."""
    start, end = C.REGIMES[regime]
    gdp = load_gdp_target()
    q = pd.PeriodIndex(gdp["quarter"].astype(str), freq="Q")
    mask = (q >= start) & (q <= end)
    return int(mask.sum())


@st.cache_data(show_spinner=False)
def load_mincer_zarnowitz() -> pd.DataFrame:
    """Mincer–Zarnowitz forecast-efficiency regressions (one row per model)."""
    if not C.MZ_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(C.MZ_CSV)


@st.cache_data(show_spinner=False)
def load_sv_calibration() -> pd.DataFrame:
    """Prediction-interval calibration summary for the DFM-SV model."""
    if not C.SV_CALIBRATION_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(C.SV_CALIBRATION_CSV)


@st.cache_data(show_spinner=False)
def load_xgb_sensitivity() -> pd.DataFrame:
    """XGB-Full post-COVID sensitivity runs: random seeds + hyperparameter

    perturbations, each re-run through the full expanding-window nowcast
    loop. Adds a ``category`` column (Seed / Hyperparameter) and marks the
    headline configuration (``seed_42``) as the baseline run.
    """
    if not C.XGB_SENSITIVITY_SUMMARY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.XGB_SENSITIVITY_SUMMARY_CSV)
    df["category"] = np.where(
        df["run"].str.startswith("seed_"), "Seed", "Hyperparameter",
    )
    df["is_baseline"] = df["run"] == "seed_42"
    return df


@st.cache_data(show_spinner=False)
def load_xgb_sensitivity_jackknife() -> pd.DataFrame:
    """Leave-one-quarter-out jackknife of the post-COVID RMSFE (headline seed)."""
    if not C.XGB_SENSITIVITY_JACKKNIFE_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(C.XGB_SENSITIVITY_JACKKNIFE_CSV)


@st.cache_data(show_spinner=False)
def load_xgb_sensitivity_dm() -> dict | None:
    """Parse the DM test result (XGB-Full seed=42 vs Rolling-AR(1) 40q, post-COVID)."""
    if not C.XGB_SENSITIVITY_DM_TXT.exists():
        return None
    text = C.XGB_SENSITIVITY_DM_TXT.read_text()
    m = re.search(r"DM=(-?[\d.]+)\s+p_value=([\d.]+)\s+n=(\d+)", text)
    if not m:
        return None
    return {
        "DM": float(m.group(1)),
        "p_value": float(m.group(2)),
        "n": int(m.group(3)),
    }


@st.cache_data(show_spinner=False)
def load_revision_path() -> pd.DataFrame:
    """DFM-EN within-quarter nowcast revisions (M1 → M2 → M3 vs actual)."""
    if not C.REVISION_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.REVISION_CSV)
    df["date"] = quarter_to_ts(df["quarter"])
    return df


# --------------------------------------------------------------------------- #
# Indicator selection
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_selection_matrix_cached(label: str, mtime: float) -> pd.DataFrame | None:
    path = C.SELECTION_MATRICES.get(label)
    if path is None or not path.exists():
        return None
    mat = pd.read_csv(path, index_col=0)
    mat.index = pd.PeriodIndex(mat.index, freq="M").to_timestamp()
    return mat.astype(float)


def load_selection_matrix(label: str) -> pd.DataFrame | None:
    path = C.SELECTION_MATRICES.get(label)
    if path is None or not path.exists():
        return None
    return _load_selection_matrix_cached(label, path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def category_share_over_time(label: str) -> pd.DataFrame:
    """Per-origin category mass share (rows sum to 1) for a selection method."""
    mat = load_selection_matrix(label)
    if mat is None:
        return pd.DataFrame()
    cat_map = load_category_map()
    cats = pd.Index([cat_map.get(feature_to_id(c), "Misc") for c in mat.columns])
    grouped = mat.T.groupby(cats).sum().T
    grouped = grouped.reindex(columns=C.CATEGORY_ORDER, fill_value=0.0)
    total = grouped.sum(axis=1).replace(0, np.nan)
    return grouped.div(total, axis=0).fillna(0.0)


def _weights_share_over_time(df: pd.DataFrame, id_col: str, weight_col: str,
                             cat_map: dict[str, str]) -> pd.DataFrame:
    """Per-quarter category importance share from a long importance frame.

    Returns a frame indexed by quarter-end timestamp with one column per
    category (rows sum to 1), matching the layout of
    :func:`category_share_over_time` so the same area chart can render it.
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["category"] = work[id_col].map(
        lambda s: cat_map.get(feature_to_id(s), "Misc"))
    grouped = (work.groupby(["quarter", "category"])[weight_col].sum()
                   .unstack(fill_value=0.0)
                   .reindex(columns=C.CATEGORY_ORDER, fill_value=0.0))
    total = grouped.sum(axis=1).replace(0, np.nan)
    grouped = grouped.div(total, axis=0).fillna(0.0)
    grouped.index = quarter_to_ts(pd.Index(grouped.index))
    return grouped.sort_index()


@st.cache_data(show_spinner=False)
def xgb_share_over_time() -> pd.DataFrame:
    """XGBoost mean|SHAP| category share per quarterly origin."""
    if not C.SHAP_CSV.exists():
        return pd.DataFrame()
    shap = pd.read_csv(C.SHAP_CSV)
    return _weights_share_over_time(shap, "feature", "mean_abs_shap",
                                    load_category_map())


@st.cache_data(show_spinner=False)
def category_share_over_time_any(method: str) -> pd.DataFrame:
    """Per-origin category mass share for any time-varying method.

    Dispatches to the binary (selection-matrix) or XGBoost (SHAP) reader so the
    structural-shift view is method-agnostic. The fixed ifoCAST set has no time
    variation and returns empty.
    """
    kind = C.SELECTION_METHODS.get(method, {}).get("kind")
    if kind == "binary":
        return category_share_over_time(method)
    if kind == "shap":
        return xgb_share_over_time()
    return pd.DataFrame()


SOFT_HARD_SEGMENTS = ["Soft (surveys)", "Hard (real activity)", "Other"]


def _triad(cat: str) -> str:
    """Collapse the 11 categories into the soft / hard / other triad."""
    if cat in C.SOFT_CATEGORIES:
        return SOFT_HARD_SEGMENTS[0]
    if cat in C.HARD_CATEGORIES:
        return SOFT_HARD_SEGMENTS[1]
    return SOFT_HARD_SEGMENTS[2]


@st.cache_data(show_spinner=False)
def soft_hard_share_over_time(method: str) -> pd.DataFrame:
    """Per-origin share of selected mass in soft / hard / other blocks."""
    share = category_share_over_time_any(method)
    if share.empty:
        return share
    out = pd.DataFrame(index=share.index)
    for seg in SOFT_HARD_SEGMENTS:
        cols = [c for c in share.columns if _triad(c) == seg]
        out[seg] = share[cols].sum(axis=1)
    return out


@st.cache_data(show_spinner=False)
def universe_soft_hard() -> pd.Series:
    """Universe base rate of the soft / hard / other triad."""
    uni = universe_category_share()
    g = uni.groupby(uni.index.map(_triad)).sum()
    return g.reindex(SOFT_HARD_SEGMENTS, fill_value=0.0)


@st.cache_data(show_spinner=False)
def regime_soft_hard() -> pd.DataFrame:
    """Soft/hard/other share of selected mass: method x regime (long frame)."""
    long = regime_category_share()
    if long.empty:
        return pd.DataFrame()
    long = long.copy()
    long["segment"] = long["category"].map(_triad)
    out = (long.groupby(["method", "regime", "segment"])["share"].sum()
               .reset_index())
    return out


def _share_from_weights(weights: pd.Series, cat_map: dict[str, str]) -> pd.Series:
    cats = pd.Index([cat_map.get(feature_to_id(i), "Misc") for i in weights.index])
    g = weights.groupby(cats).sum()
    g = g.reindex(C.CATEGORY_ORDER, fill_value=0.0)
    tot = g.sum()
    return g / tot if tot > 0 else g


# --------------------------------------------------------------------------- #
# Unified per-method view (base-rate aware)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def ifocast_membership() -> list[str]:
    """Series ids in the fixed ifoCAST expert set (GDP target excluded).

    Uses ``in_panel == True`` from the supervisor-confirmed mapping when present,
    so the dashboard matches DFM-ifoCAST (19 active predictors; detrad3414 is
    mapped but absent from the transformed panel).
    """
    if not C.IFOCAST_MAPPING_CSV.exists():
        return []
    m = pd.read_csv(C.IFOCAST_MAPPING_CSV)
    pred = m[~m["my_id"].isin(["(target)"]) & m["my_id"].notna()].copy()
    if "in_panel" in pred.columns:
        pred = pred[pred["in_panel"].astype(bool)]
    ids = pred["my_id"].astype(str)
    ids = ids[~ids.isin(["nan", ""])]
    return list(dict.fromkeys(ids))  # order-preserving unique


def _method_series_weights(method: str) -> pd.Series:
    """Total importance mass per series id for one method (index = series id).

    Harmonises the native signals onto a single non-negative weight vector so
    category mass, lift and rankings can be computed identically downstream.
    """
    kind = C.SELECTION_METHODS[method]["kind"]
    if kind == "binary":
        mat = load_selection_matrix(method)
        return pd.Series(dtype=float) if mat is None else mat.sum(axis=0)
    if kind == "shap":
        if not C.SHAP_CSV.exists():
            return pd.Series(dtype=float)
        s = pd.read_csv(C.SHAP_CSV)
        s["id"] = s["feature"].map(feature_to_id)
        return s.groupby("id")["mean_abs_shap"].sum()
    if kind == "fixed":
        ids = ifocast_membership()
        return pd.Series(1.0, index=ids) if ids else pd.Series(dtype=float)
    return pd.Series(dtype=float)


@st.cache_data(show_spinner=False)
def available_methods() -> list[str]:
    """Part I methods whose source artefact is present."""
    return [m for m in C.SELECTION_METHOD_ORDER
            if not _method_series_weights(m).empty]


@st.cache_data(show_spinner=False)
def universe_category_share() -> pd.Series:
    """Share of the predictor *universe* in each category (the base rate)."""
    cat_map = load_category_map()
    counts = pd.Series(
        pd.Index([cat_map.get(str(i), "Misc") for i in cat_map]).value_counts())
    counts = counts.reindex(C.CATEGORY_ORDER, fill_value=0.0)
    return counts / counts.sum()


@st.cache_data(show_spinner=False)
def all_methods_category_share() -> pd.DataFrame:
    """Category mass share per method (columns sum to 1). Index = CATEGORY_ORDER."""
    cat_map = load_category_map()
    cols = {}
    for m in available_methods():
        cols[m] = _share_from_weights(_method_series_weights(m), cat_map)
    return pd.DataFrame(cols).reindex(C.CATEGORY_ORDER).fillna(0.0)


@st.cache_data(show_spinner=False)
def cross_method_agreement() -> pd.DataFrame:
    """Spearman rank-correlation across every method over the union universe.

    Each method's per-series mass is reindexed onto the union of all series and
    0-filled (a not-selected series ranks at the bottom), then Spearman ρ is
    computed pairwise — now including the fixed ifoCAST set.
    """
    from scipy.stats import spearmanr

    methods = available_methods()
    weights = {m: _method_series_weights(m) for m in methods}
    universe = sorted(set().union(*[w.index for w in weights.values()]))
    aligned = pd.DataFrame(
        {m: w.reindex(universe).fillna(0.0) for m, w in weights.items()})
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            out.loc[a, b] = spearmanr(aligned[a], aligned[b]).correlation
    return out.astype(float)


@st.cache_data(show_spinner=False)
def method_overlap_jaccard(top_n: int = 20) -> pd.DataFrame:
    """Jaccard overlap of each method's top-``top_n`` indicators (set agreement)."""
    methods = available_methods()
    tops = {}
    for m in methods:
        w = _method_series_weights(m).sort_values(ascending=False)
        tops[m] = set(w.head(top_n).index) if len(w) > top_n else set(w.index)
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            ua, ub = tops[a], tops[b]
            out.loc[a, b] = len(ua & ub) / len(ua | ub) if (ua | ub) else np.nan
    return out.astype(float)


@st.cache_data(show_spinner=False)
def regime_category_share() -> pd.DataFrame:
    """Category mass share by regime for EN / XGBoost(SHAP).

    Returns long frame: method, regime, category, share.
    """
    cat_map = load_category_map()
    out = []

    for label, method_name in [
        ("EN (raw)", "EN"),
        ("Block-balanced (k=20)", "Block-balanced"),
        ("PLS", "PLS"),
    ]:
        mat = load_selection_matrix(label)
        if mat is None:
            continue
        idx_q = pd.PeriodIndex(mat.index, freq="Q").astype(str)
        for regime, (s, e) in C.REGIMES.items():
            mask = (idx_q >= s) & (idx_q <= e)
            if mask.sum() == 0:
                continue
            counts = mat.loc[mask].sum(axis=0)
            share = _share_from_weights(counts, cat_map)
            for cat, val in share.items():
                out.append({"method": method_name, "regime": regime,
                            "category": cat, "share": val})

    # XGBoost — mean |SHAP| aggregated within each regime window
    if C.SHAP_CSV.exists():
        shap = pd.read_csv(C.SHAP_CSV)
        shap["q"] = shap["quarter"].astype(str)
        for regime, (s, e) in C.REGIMES.items():
            w = shap.loc[(shap["q"] >= s) & (shap["q"] <= e)]
            if w.empty:
                continue
            agg = w.groupby("feature")["mean_abs_shap"].sum()
            share = _share_from_weights(agg, cat_map)
            for cat, val in share.items():
                out.append({"method": "XGBoost (SHAP)", "regime": regime,
                            "category": cat, "share": val})

    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def input_set_composition() -> pd.DataFrame:
    """Economic composition (% of total selections) of each candidate input set.

    Covers every set a DFM can consume — the data-driven screens plus the fixed
    ifoCAST expert set — so the structural make-up can be compared directly.
    """
    cat_map = load_category_map()
    cols = {}
    for label, path in C.INPUT_SET_MATRICES.items():
        if not path.exists():
            continue
        mat = pd.read_csv(path, index_col=0)
        share = _share_from_weights(mat.sum(axis=0), cat_map)
        cols[label] = 100 * share
    ids = ifocast_membership()
    if ids:
        cols["ifoCAST (fixed)"] = 100 * _share_from_weights(
            pd.Series(1.0, index=ids), cat_map)
    return pd.DataFrame(cols).reindex(C.CATEGORY_ORDER).fillna(0.0)


# --------------------------------------------------------------------------- #
# Category contributions (DFM-EN decomposition)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_en_input_matrix_cached(mtime: float) -> pd.DataFrame | None:
    path = C.INPUT_SET_MATRICES.get("EN only")
    if path is None or not path.exists():
        return None
    mat = pd.read_csv(path, index_col=0)
    mat.index = pd.PeriodIndex(mat.index, freq="M").to_timestamp()
    return mat.astype(float)


def load_en_input_matrix() -> pd.DataFrame | None:
    """Monthly EN-only selection matrix fed to DFM-EN."""
    path = C.INPUT_SET_MATRICES.get("EN only")
    if path is None or not path.exists():
        return None
    return _load_en_input_matrix_cached(path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_series_names() -> dict[str, str]:
    """Map series id -> full metadata name."""
    md = pd.read_csv(C.DATA_DICT_CSV)
    md = md.drop_duplicates(subset="id")
    return dict(zip(md["id"].astype(str), md["name"].astype(str)))


_NAME_TRIM = re.compile(
    r",\s*(?:SA \(.*\)|Calendar Adjusted.*|Constant Prices.*|Balance.*)$"
)


def series_short_label(series_id: str, names: dict[str, str] | None = None) -> str:
    """Compact display label for hovers and notes."""
    sid = feature_to_id(series_id)
    if sid in C.MISC_SERIES_LABELS:
        return C.MISC_SERIES_LABELS[sid]
    name = (names or load_series_names()).get(sid)
    if not name:
        return sid
    short = name.removeprefix("Germany, ").strip()
    short = _NAME_TRIM.sub("", short)
    if len(short) > 44:
        short = short[:43] + "…"
    return short


def origin_category_hovers(
    start: str, end: str, top_n: int = C.DECOMP_HOVER_TOP_N,
    series_parquet: "Path | None" = None,
) -> dict[tuple[pd.Timestamp, str], str]:
    """Per-origin, per-category hover: top series by |contribution| (pp).

    ``series_parquet`` defaults to the DFM-EN series-contribution cache; pass
    ``C.SERIES_CONTRIB_PARQUET_TVP`` for the DFM-TVP decomposition.
    """
    parquet_path = series_parquet or C.SERIES_CONTRIB_PARQUET
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["monthly_origin"] + "-01")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return {}
    names = load_series_names()
    out: dict[tuple[pd.Timestamp, str], str] = {}
    for (dt, cat), grp in df.groupby(["date", "category"], sort=False):
        n = len(grp)
        top = grp.loc[grp["contrib_pp"].abs().nlargest(min(top_n, n)).index]
        lines = "<br>".join(
            f"{series_short_label(sid, names)} ({sid}): {val:+.3f} pp"
            for sid, val in zip(top["series"], top["contrib_pp"])
        )
        suffix = f"<br>+{n - top_n} more" if n > top_n else ""
        cat_display = C.CATEGORY_DISPLAY.get(cat, cat)
        out[(pd.Timestamp(dt), cat)] = (
            f"<i>{cat_display} category — top {min(n, top_n)} series by |contribution| "
            f"({n} total):</i><br>{lines}{suffix}"
        )
    return out


def misc_en_selected(start: str, end: str) -> list[tuple[str, str]]:
    """Misc-series (id, label) pairs EN selects at least once in ``[start, end]``."""
    mat = load_en_input_matrix()
    if mat is None:
        return []
    cat_map = load_category_map()
    sub = mat.loc[(mat.index >= start) & (mat.index <= end)]
    out: list[tuple[str, str]] = []
    for sid, label in C.MISC_SERIES_LABELS.items():
        if sid not in mat.columns or cat_map.get(sid) != "Misc":
            continue
        if (sub[sid] > 0).any():
            out.append((sid, label))
    return out


def _deprod2001_method_note(start: str, end: str) -> str:
    """One-line cross-method context when capacity utilisation is an EN driver."""
    if pd.Timestamp(end) < pd.Timestamp("2020-07-01"):
        return ""
    if pd.Timestamp(start) >= pd.Timestamp("2023-01-01"):
        bb = "block-balanced includes it throughout"
    elif pd.Timestamp(end) >= pd.Timestamp("2023-01-01"):
        bb = "block-balanced from 2023 onward (not in 2022)"
    else:
        bb = "block-balanced only from 2023"
    return (
        f" <i>Note:</i> <code>deprod2001</code> is not in the ifoCAST fixed set "
        f"or PLS; {bb} — an EN-led signal the expert set omits."
    )


def misc_other_callout(start: str, end: str) -> str:
    """HTML note for the decomposition chart — only lists EN-selected Misc drivers."""
    selected = misc_en_selected(start, end)
    if not selected:
        return (
            "<b>Other</b> (<i>Misc</i>) — catch-all category for indicators outside "
            "the main blocks. <b>No Misc series is EN-selected in this period</b>, "
            "so the Other bar should be empty or negligible. Hover any category "
            "bar for the top contributors by |contribution| at that origin."
        )
    parts = [f"<b>{label}</b> (<code>{sid}</code>)" for sid, label in selected]
    drivers = ", ".join(parts)
    if len(selected) == 1:
        tail = "that series is what drives the Other bar here."
    else:
        tail = "these series are what drive the Other bar here."
    return (
        f"<b>Other</b> (<i>Misc</i>) — catch-all category. In this window EN selects "
        f"{drivers}; {tail} Hover any category bar for the top "
        f"{C.DECOMP_HOVER_TOP_N} contributors by |contribution| (pp) at that origin."
        + (_deprod2001_method_note(start, end)
           if any(sid == "deprod2001" for sid, _ in selected) else "")
    )


@st.cache_data(show_spinner=False)
def load_contributions() -> pd.DataFrame:
    if not C.CONTRIB_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(C.CONTRIB_PARQUET)
    df["date"] = pd.to_datetime(df["monthly_origin"] + "-01")
    return df


@st.cache_data(show_spinner=False)
def load_contributions_tvp() -> pd.DataFrame:
    """DFM-TVP category contributions (factor-bridge attribution)."""
    if not C.CONTRIB_PARQUET_TVP.exists():
        return pd.DataFrame()
    df = pd.read_parquet(C.CONTRIB_PARQUET_TVP)
    df["date"] = pd.to_datetime(df["monthly_origin"] + "-01")
    return df


@st.cache_data(show_spinner=False)
def load_contributions_blockbalanced() -> pd.DataFrame:
    """DFM-BlockBalanced category contributions (same attribution as DFM-EN)."""
    if not C.CONTRIB_PARQUET_BLOCKBALANCED.exists():
        return pd.DataFrame()
    df = pd.read_parquet(C.CONTRIB_PARQUET_BLOCKBALANCED)
    df["date"] = pd.to_datetime(df["monthly_origin"] + "-01")
    return df


@st.cache_data(show_spinner=False)
def load_factor_loading_categories() -> pd.DataFrame:
    """M3 category shares of |indicator→factor loadings| (Stage 1, long format)."""
    if not C.FACTOR_LOADING_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.FACTOR_LOADING_CSV, parse_dates=["date"])
    df = df[df["record_type"] == "category"].copy()
    df["category"] = df["category"].replace({"Other": "Misc"})
    return df.sort_values("date")


@st.cache_data(show_spinner=False)
def load_tvp_m3_bridge() -> pd.DataFrame:
    """M3 DFM-TVP bridge coefficients (factor → GDP, Stage 2)."""
    if not C.TVP_RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.TVP_RESULTS_CSV)
    df = df[df["month_in_quarter"] == 3].copy()
    df["date"] = pd.PeriodIndex(df["quarter"].astype(str), freq="Q").to_timestamp(
        how="end",
    )
    return df.sort_values("date")


# --------------------------------------------------------------------------- #
# Ragged edge / data coverage
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_ragged_edge() -> pd.DataFrame:
    """Per-series ragged-edge snapshot at the forecast origin, with category."""
    if not C.RAGGED_EDGE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.RAGGED_EDGE_CSV)
    cat_map = load_category_map()
    df["category"] = df["series"].map(lambda s: cat_map.get(feature_to_id(s), "Misc"))
    df["pub_lag"] = pd.to_numeric(df["pub_lag"], errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False)
def publag_category_matrix() -> pd.DataFrame:
    """Count of series by publication lag (rows) x category (cols)."""
    df = load_ragged_edge()
    if df.empty:
        return pd.DataFrame()
    mat = (df.groupby(["pub_lag", "category"]).size()
             .unstack(fill_value=0)
             .reindex(columns=C.CATEGORY_ORDER, fill_value=0))
    mat.index = [int(i) for i in mat.index]
    return mat
