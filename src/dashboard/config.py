"""Central configuration for the nowcasting dashboard.

Holds canonical paths, the unified colour system, model registry and economic
regime definitions. Everything visual flows from here so palettes stay tonal
and consistent across Part I (indicator selection) and Part II (nowcasting).

Data resolution
---------------
Three ways to point the dashboard at a data directory, checked in order:

1. ``DASHBOARD_DATA_DIR`` environment variable — an absolute path to a
   directory laid out like ``data/demo`` (see ``data/README.md``).
2. ``data/real/`` next to this repository, if it exists. This is the
   conventional local drop-in spot for your own (gitignored) results.
3. ``data/demo/`` — the small, synthetic sample shipped with the repo so the
   dashboard always runs out of the box.

``IS_DEMO_DATA`` tells the UI whether option 3 is active, so it can show a
one-line disclosure banner.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DATA = REPO_ROOT / "data" / "demo"
_REAL_DATA = REPO_ROOT / "data" / "real"

_env_dir = os.environ.get("DASHBOARD_DATA_DIR")
if _env_dir:
    DATA_ROOT = Path(_env_dir).expanduser().resolve()
elif _REAL_DATA.exists():
    DATA_ROOT = _REAL_DATA
else:
    DATA_ROOT = _DEMO_DATA

IS_DEMO_DATA = DATA_ROOT == _DEMO_DATA

OUT_SELECTION = DATA_ROOT / "indicator_selection"
OUT_NOWCAST = DATA_ROOT / "nowcasting"
SEL_FIG = OUT_SELECTION / "figures"
NC_FIG = OUT_NOWCAST / "figures"

DATA_DICT_CSV = DATA_ROOT / "metadata" / "data_dict.csv"
GDP_TARGET_CSV = OUT_SELECTION / "gdp_target.csv"

# Canonical Elastic Net selection (COVID down-weighting + max_selected=60 cap).
# Same matrix feeds Part I emphasis plots and DFM-EN.
EN_SELECTION_MATRIX_CSV = (
    OUT_SELECTION / "dfm_input_sets" / "en_only_selection_matrix.csv"
)

# Selection matrices (binary masks, origin x series)
SELECTION_MATRICES = {
    "EN (raw)": EN_SELECTION_MATRIX_CSV,
    "Block-balanced (k=20)": OUT_SELECTION / "selection_matrix_blockbalanced_k20.csv",
    "PLS": OUT_SELECTION / "selection_matrix_pls.csv",
}

# Selection matrices that feed the DFM in Part II
INPUT_SET_MATRICES = {
    "EN only": EN_SELECTION_MATRIX_CSV,
    "Block-balanced (k=20)": OUT_SELECTION / "selection_matrix_blockbalanced_k20.csv",
    "PLS": OUT_SELECTION / "selection_matrix_pls.csv",
}

# ifoCAST fixed expert set — membership comes from the published mapping table
# (one row per ifoCAST indicator, resolved to this panel's series ids).
IFOCAST_MAPPING_CSV = OUT_NOWCAST / "ifocast_indicator_mapping.csv"

# --------------------------------------------------------------------------- #
# Unified Part I method registry
# --------------------------------------------------------------------------- #
# Every indicator-selection signal mapped to a common interface so the whole of
# Part I can be read side by side. ``kind`` drives how the raw artefact is turned
# into a per-series importance score in the data layer:
#   binary  -> selection_matrix (0/1, origin x series); score = mean selection rate
#   shap    -> XGBoost mean|SHAP| per (feature, quarter)
#   fixed   -> ifoCAST expert set (constant membership, no time variation)
SELECTION_METHODS: dict[str, dict] = {
    "EN (raw)":              {"kind": "binary", "color": "#2D6CB3", "data": True},
    "Block-balanced (k=20)": {"kind": "binary", "color": "#8F3D58", "data": True},
    "PLS":                   {"kind": "binary", "color": "#7FB7A6", "data": True},
    "XGBoost (SHAP)":        {"kind": "shap",   "color": "#3E9B73", "data": True},
    "ifoCAST (fixed)":       {"kind": "fixed",  "color": "#C9617F", "data": False},
}
SELECTION_METHOD_ORDER = list(SELECTION_METHODS)

# Nowcasting tables
RMSFE_ALL_CSV = OUT_NOWCAST / "rmsfe_table_all_models.csv"
HORIZON_PROFILE_CSV = OUT_NOWCAST / "horizon_profile_table.csv"
HORIZON_BIAS_VARIANCE_CSV = OUT_NOWCAST / "horizon_bias_variance_table.csv"
POST_COVID_CSV = OUT_NOWCAST / "post_covid_benchmarks_table.csv"
DM_ALL_CSV = OUT_NOWCAST / "diebold_mariano_table_all_models.csv"
CONTRIB_PARQUET = OUT_NOWCAST / "category_contribs_en_2017_2025.parquet"
SERIES_CONTRIB_PARQUET = OUT_NOWCAST / "series_contribs_en_2017_2025.parquet"
CONTRIB_PARQUET_TVP = OUT_NOWCAST / "category_contribs_tvp_2017_2025.parquet"
SERIES_CONTRIB_PARQUET_TVP = OUT_NOWCAST / "series_contribs_tvp_2017_2025.parquet"
CONTRIB_PARQUET_BLOCKBALANCED = (
    OUT_NOWCAST / "category_contribs_blockbalanced_2017_2025.parquet"
)
SERIES_CONTRIB_PARQUET_BLOCKBALANCED = (
    OUT_NOWCAST / "series_contribs_blockbalanced_2017_2025.parquet"
)
FACTOR_LOADING_CSV = OUT_NOWCAST / "factor_loading_m3_panel.csv"
TVP_RESULTS_CSV = OUT_NOWCAST / "nowcast_results_dfm_tvp.csv"
SHAP_CSV = OUT_NOWCAST / "xgb_shap_importance.csv"
RAGGED_EDGE_CSV = OUT_NOWCAST / "ragged_edge_diagnostics" / "info_set_summary.csv"
MZ_CSV = OUT_NOWCAST / "mincer_zarnowitz_table.csv"
SV_CALIBRATION_CSV = OUT_NOWCAST / "sv_interval_calibration_table.csv"
REVISION_CSV = OUT_NOWCAST / "dfm_en_forecast_revision.csv"

# XGB-Full post-COVID robustness check (seed + hyperparameter sensitivity,
# leave-one-quarter-out jackknife, DM vs. Rolling-AR(1) 40q).
XGB_SENSITIVITY_DIR = OUT_NOWCAST / "_scratch"
XGB_SENSITIVITY_SUMMARY_CSV = XGB_SENSITIVITY_DIR / "xgb_sensitivity_summary.csv"
XGB_SENSITIVITY_JACKKNIFE_CSV = (
    XGB_SENSITIVITY_DIR / "xgb_sensitivity_jackknife_postcovid.csv"
)
XGB_SENSITIVITY_DM_TXT = XGB_SENSITIVITY_DIR / "xgb_sensitivity_dm_vs_rolling_ar1.txt"

CONTRIB_PERIODS = {
    "Full window (2017–2025)": ("2017-01-01", "2025-12-31"),
    "Pre-COVID (2017–2019)": ("2017-01-01", "2019-12-31"),
    "COVID (2020–2021)": ("2020-01-01", "2021-12-31"),
    "post-COVID (2022–2025)": ("2022-01-01", "2025-12-31"),
}

# --------------------------------------------------------------------------- #
# Brand palette
# --------------------------------------------------------------------------- #
INK = "#1A2332"
SUBTLE = "#64748B"
GRID = "#E5E9F0"
PAPER = "#FFFFFF"
PANEL = "#F7F9FC"
ACCENT = "#2D6CB3"

FONT_FAMILY = "Inter, 'Source Sans 3', Helvetica, Arial, sans-serif"

# --------------------------------------------------------------------------- #
# Economic regimes (inclusive quarter bounds)
# --------------------------------------------------------------------------- #
REGIMES: dict[str, tuple[str, str]] = {
    "pre-COVID": ("2011Q1", "2019Q4"),
    "COVID": ("2020Q1", "2021Q4"),
    "post-COVID": ("2022Q1", "2025Q4"),
}
REGIME_COLORS = {
    "pre-COVID": "#7FB7A6",
    "COVID": "#E2899B",
    "post-COVID": "#9F8AD1",
}
EVAL_START, EVAL_END = "2011Q1", "2025Q4"

# --------------------------------------------------------------------------- #
# Indicator categories (11 economic blocks)
# --------------------------------------------------------------------------- #
CATEGORY_ORDER = [
    "Surveys", "Orders", "Turnover", "Production", "Construction",
    "Trade", "Prices", "Commodities", "Financial", "Global", "Misc",
    "Offset", "Baseline",
]
CATEGORY_DISPLAY = {"Misc": "Other", "Surveys": "Surveys (soft)",
                    "Baseline": "Baseline (intercept)",
                    "Offset": "Cross-category offset"}

# Factor-loading stacked areas (Stage 1) — subset of categories with stable shares
FACTOR_LOADING_CATEGORIES = [
    "Production", "Turnover", "Orders", "Surveys", "Trade", "Global", "Misc",
]
FACTOR_COLORS = ["#8E44AD", "#4A6FA5"]
# Empirical basis (factor_loading_m3_panel.csv, full sample, capped60 EN set):
# Factor 1 loading mass is ~90% hard data (Turnover 39%, Production 25%,
# Orders 23%) — a clean real-activity composite. Factor 2 is genuinely mixed:
# Surveys is its single largest category (~31%) but Turnover (26%) + Orders
# (22%) + Production (17%) together still account for ~two-thirds of its
# loading mass, so it is NOT a "surveys" factor — label it as the secondary,
# more balanced composite instead.
FACTOR_LABELS = [
    "Factor 1 — real activity (turnover & production)",
    "Factor 2 — mixed demand & sentiment (no single category dominates)",
]
FACTOR_SHORT = ["Factor 1: real activity", "Factor 2: mixed signal"]

# Misc / "Other" — short labels for dashboard notes and chart hovers.
MISC_SERIES_LABELS: dict[str, str] = {
    "deprod2001": "Manufacturing capacity utilisation",
    "deepunnewsindex": "Germany policy uncertainty (news-based EPU)",
    "euepunnewsindex": "Europe policy uncertainty (news-based EPU)",
    # Synthetic pseudo-series id used by the leverage-capped attribution in
    # nowcast_plots._contrib_frame / tvp_dfm._tvp_contrib_frame — see
    # CONTRIB_MAX_LEVERAGE docstring there.
    "__offset__": "Cross-category offset (indicator cancellation)",
}
DECOMP_HOVER_TOP_N = 3

CATEGORY_COLORS = {
    "Surveys": "#E2899B",
    "Orders": "#F0B97D",
    "Turnover": "#F2D58A",
    "Production": "#7FB7A6",
    "Construction": "#A9D2C4",
    "Trade": "#8FB8DE",
    "Prices": "#B7A4DE",
    "Commodities": "#D2AE86",
    "Financial": "#92AED4",
    "Global": "#C7AEC9",
    "Misc": "#C9C2B4",
    "Offset": "#8D95A3",
    "Baseline": "#B8C0CC",
}

SOFT_CATEGORIES = {"Surveys"}
HARD_CATEGORIES = {"Orders", "Turnover", "Production", "Construction", "Trade"}

PUBLAG_LABELS = {
    0: "lag 0 — same month (timely)",
    1: "lag 1 month",
    2: "lag 2 months (hard data)",
}
PUBLAG_COLORS = {
    0: "#E2899B",
    1: "#F2D58A",
    2: "#8FB8DE",
}

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
_NC = OUT_NOWCAST


class ModelSpec:
    __slots__ = ("key", "label", "color", "family", "file", "has_miq")

    def __init__(self, key, label, color, family, file, has_miq):
        self.key = key
        self.label = label
        self.color = color
        self.family = family
        self.file = file
        self.has_miq = has_miq


# Model colours — "twilight garden" harmony: warm rose→orchid DFM arc, cerulean
# counterpoint, amber bridge, teal + indigo ML accents. Same energy level
# across models so multi-line charts feel cohesive, not neon.
MODELS: dict[str, ModelSpec] = {
    "RW": ModelSpec("RW", "Random Walk", "#AEC0D8", "Baselines",
                    _NC / "nowcast_results_rw.csv", False),
    "AR1": ModelSpec("AR1", "AR(1)", "#6E88A8", "Baselines",
                     _NC / "nowcast_results_ar1.csv", False),
    "DFM-ifoCAST": ModelSpec("DFM-ifoCAST", "DFM · ifoCAST fixed set", "#E06B82",
                             "DFM (A-CD-TPN)", _NC / "nowcast_results_dfm_ifocast.csv", True),
    "DFM-EN": ModelSpec("DFM-EN", "DFM · EN inputs", "#EC6F8E", "DFM (A-CD-TPN)",
                        _NC / "nowcast_results_actpn_en_only.csv", True),
    # Appendix / Accuracy-tab comparison only (PLS+VIP top-30 input set).
    # Kept out of DM heatmap and other Part II surfaces via APPENDIX_COMPARE_MODELS.
    "DFM-PLS": ModelSpec("DFM-PLS", "DFM · PLS inputs", "#D96B85", "DFM (A-CD-TPN)",
                         _NC / "nowcast_results_actpn_pls_only.csv", True),
    "DFM-BlockBalanced": ModelSpec("DFM-BlockBalanced", "DFM · block-balanced k=20",
                                   "#C44E7A", "DFM (A-CD-TPN)",
                                   _NC / "nowcast_results_dfm_blockbalanced.csv", True),
    "DFM-TVP": ModelSpec("DFM-TVP", "DFM-TVP (COVID-robust)", "#A855C8", "DFM-TVP",
                         _NC / "nowcast_results_dfm_tvp.csv", True),
    "DFM-SV-k2": ModelSpec("DFM-SV-k2", "DFM-SV (k=2, integrated, EN)", "#4A8FE7", "DFM-SV",
                           _NC / "nowcast_results_actpn_sv_integrated_k2.csv", True),
    "combo_equal": ModelSpec("combo_equal", "Equal-weight combo", "#F0AD4E", "Ensemble",
                             _NC / "nowcast_path_combo_equal.csv", True),
    "XGB-Full": ModelSpec("XGB-Full", "XGBoost (Full)", "#2DB896", "Machine learning",
                          _NC / "nowcast_results_xgb_full.csv", True),
    "MLP-Factor": ModelSpec("MLP-Factor", "MLP · factor-augmented", "#6B5CE7",
                            "Machine learning",
                            _NC / "nowcast_results_mlp_factor.csv", True),
}

# Light badge tints paired with each accent (bg, fg).
MODEL_BADGE: dict[str, tuple[str, str]] = {
    "DFM-EN": ("#FDE8EE", "#B83562"),
    "DFM-TVP": ("#F3E5F9", "#7B2D9E"),
    "DFM-SV-k2": ("#E6F1FD", "#2563B8"),
    "XGB-Full": ("#E0F7F2", "#0D7A62"),
    "MLP-Factor": ("#EDEAFE", "#4C3DB8"),
}

MODEL_ORDER = [
    "RW", "AR1",
    "DFM-ifoCAST", "DFM-EN", "DFM-PLS", "DFM-BlockBalanced",
    "DFM-TVP", "DFM-SV-k2", "combo_equal",
    "XGB-Full", "MLP-Factor",
]

# Surfaced only on Accuracy & model paths (RMSE-by-regime, paths, ranking).
# Excluded from DM / specs / headline horse-race surfaces.
APPENDIX_COMPARE_MODELS = frozenset({"DFM-PLS"})

FAMILY_ORDER = ["Baselines", "DFM (A-CD-TPN)", "DFM-TVP", "DFM-SV",
                "Ensemble", "Machine learning"]

ACTUAL_COLOR = INK


def model_color(key: str) -> str:
    spec = MODELS.get(key)
    return spec.color if spec else SUBTLE


def model_badge(key: str) -> tuple[str, str]:
    return MODEL_BADGE.get(key, (PANEL, SUBTLE))


def model_label(key: str) -> str:
    spec = MODELS.get(key)
    return spec.label if spec else key
