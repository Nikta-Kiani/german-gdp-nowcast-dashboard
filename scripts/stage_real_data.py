"""Stage the real thesis outputs into data/real/ for local use or private deployment.

This copies *only* the specific files the dashboard reads (see data/README.md
for the full expected layout) out of the research pipeline's outputs/ tree —
never the whole tree, and never the licensed source workbook or raw panel.

Nothing this script touches is committed to this repository: data/real/ is
gitignored, and the source directory lives outside this repo entirely.

Usage
-----
    python scripts/stage_real_data.py --source /path/to/Project_files

``--source`` should point at the ``Project_files`` directory of the
german-gdp-nowcasting pipeline (the one containing ``outputs/`` and
``data/metadata/``). Defaults to the ``SOURCE_ROOT`` environment variable if
``--source`` is omitted.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "real"

# (relative source path under Project_files, relative dest path under data/real/)
FILES = [
    ("data/metadata/data_dict_enriched.csv", "metadata/data_dict.csv"),
    ("outputs/indicator_selection/gdp_target.csv", "indicator_selection/gdp_target.csv"),
    (
        "outputs/indicator_selection/dfm_input_sets/en_only_selection_matrix.csv",
        "indicator_selection/dfm_input_sets/en_only_selection_matrix.csv",
    ),
    (
        "outputs/indicator_selection/selection_matrix_blockbalanced_k20.csv",
        "indicator_selection/selection_matrix_blockbalanced_k20.csv",
    ),
    (
        "outputs/indicator_selection/selection_matrix_pls.csv",
        "indicator_selection/selection_matrix_pls.csv",
    ),
    ("outputs/nowcasting/ifocast_indicator_mapping.csv", "nowcasting/ifocast_indicator_mapping.csv"),
    ("outputs/nowcasting/rmsfe_table_all_models.csv", "nowcasting/rmsfe_table_all_models.csv"),
    ("outputs/nowcasting/horizon_profile_table.csv", "nowcasting/horizon_profile_table.csv"),
    ("outputs/nowcasting/horizon_bias_variance_table.csv", "nowcasting/horizon_bias_variance_table.csv"),
    ("outputs/nowcasting/post_covid_benchmarks_table.csv", "nowcasting/post_covid_benchmarks_table.csv"),
    (
        "outputs/nowcasting/diebold_mariano_table_all_models.csv",
        "nowcasting/diebold_mariano_table_all_models.csv",
    ),
    (
        "outputs/nowcasting/category_contribs_en_2017_2025.parquet",
        "nowcasting/category_contribs_en_2017_2025.parquet",
    ),
    (
        "outputs/nowcasting/series_contribs_en_2017_2025.parquet",
        "nowcasting/series_contribs_en_2017_2025.parquet",
    ),
    (
        "outputs/nowcasting/category_contribs_tvp_2017_2025.parquet",
        "nowcasting/category_contribs_tvp_2017_2025.parquet",
    ),
    (
        "outputs/nowcasting/series_contribs_tvp_2017_2025.parquet",
        "nowcasting/series_contribs_tvp_2017_2025.parquet",
    ),
    (
        "outputs/nowcasting/category_contribs_blockbalanced_2017_2025.parquet",
        "nowcasting/category_contribs_blockbalanced_2017_2025.parquet",
    ),
    (
        "outputs/nowcasting/series_contribs_blockbalanced_2017_2025.parquet",
        "nowcasting/series_contribs_blockbalanced_2017_2025.parquet",
    ),
    ("outputs/nowcasting/factor_loading_m3_panel.csv", "nowcasting/factor_loading_m3_panel.csv"),
    ("outputs/nowcasting/nowcast_results_dfm_tvp.csv", "nowcasting/nowcast_results_dfm_tvp.csv"),
    ("outputs/nowcasting/xgb_shap_importance.csv", "nowcasting/xgb_shap_importance.csv"),
    (
        "outputs/nowcasting/ragged_edge_diagnostics/info_set_summary.csv",
        "nowcasting/ragged_edge_diagnostics/info_set_summary.csv",
    ),
    ("outputs/nowcasting/mincer_zarnowitz_table.csv", "nowcasting/mincer_zarnowitz_table.csv"),
    ("outputs/nowcasting/sv_interval_calibration_table.csv", "nowcasting/sv_interval_calibration_table.csv"),
    ("outputs/nowcasting/dfm_en_forecast_revision.csv", "nowcasting/dfm_en_forecast_revision.csv"),
    (
        "outputs/nowcasting/_scratch/xgb_sensitivity_summary.csv",
        "nowcasting/_scratch/xgb_sensitivity_summary.csv",
    ),
    (
        "outputs/nowcasting/_scratch/xgb_sensitivity_jackknife_postcovid.csv",
        "nowcasting/_scratch/xgb_sensitivity_jackknife_postcovid.csv",
    ),
    (
        "outputs/nowcasting/_scratch/xgb_sensitivity_dm_vs_rolling_ar1.txt",
        "nowcasting/_scratch/xgb_sensitivity_dm_vs_rolling_ar1.txt",
    ),
    ("outputs/nowcasting/nowcast_results_rw.csv", "nowcasting/nowcast_results_rw.csv"),
    ("outputs/nowcasting/nowcast_results_ar1.csv", "nowcasting/nowcast_results_ar1.csv"),
    ("outputs/nowcasting/nowcast_results_dfm_ifocast.csv", "nowcasting/nowcast_results_dfm_ifocast.csv"),
    ("outputs/nowcasting/nowcast_results_actpn_en_only.csv", "nowcasting/nowcast_results_actpn_en_only.csv"),
    ("outputs/nowcasting/nowcast_results_actpn_pls_only.csv", "nowcasting/nowcast_results_actpn_pls_only.csv"),
    (
        "outputs/nowcasting/nowcast_results_dfm_blockbalanced.csv",
        "nowcasting/nowcast_results_dfm_blockbalanced.csv",
    ),
    (
        "outputs/nowcasting/nowcast_results_actpn_sv_integrated_k2.csv",
        "nowcasting/nowcast_results_actpn_sv_integrated_k2.csv",
    ),
    ("outputs/nowcasting/nowcast_path_combo_equal.csv", "nowcasting/nowcast_path_combo_equal.csv"),
    ("outputs/nowcasting/nowcast_results_xgb_full.csv", "nowcasting/nowcast_results_xgb_full.csv"),
    ("outputs/nowcasting/nowcast_results_mlp_factor.csv", "nowcasting/nowcast_results_mlp_factor.csv"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to the research pipeline's Project_files directory.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEST,
        help=f"Where to stage the curated files (default: {DEST}).",
    )
    args = parser.parse_args()

    source = args.source
    if source is None:
        import os

        env_source = os.environ.get("SOURCE_ROOT")
        if env_source:
            source = Path(env_source)
    if source is None:
        parser.error("Pass --source /path/to/Project_files or set SOURCE_ROOT.")
    source = source.expanduser().resolve()

    missing: list[str] = []
    copied = 0
    for rel_src, rel_dst in FILES:
        src_path = source / rel_src
        dst_path = args.dest / rel_dst
        if not src_path.exists():
            missing.append(rel_src)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"Staged {copied}/{len(FILES)} files into {args.dest}")
    if missing:
        print("Missing (skipped):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
