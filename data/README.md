# Data directory

The dashboard picks a data directory in this order (see
[`src/dashboard/config.py`](../src/dashboard/config.py)):

1. **`DASHBOARD_DATA_DIR`** — an absolute path with the same layout as `demo/`
   below.
2. **`data/real/`** — gitignored drop-in for your own results.
3. **`data/demo/`** — the synthetic sample bundled with the repo. Used if
   neither of the above exists, so a clean clone always runs.

## Layout expected under `real/` (or `DASHBOARD_DATA_DIR`)

```text
<data_dir>/
├── metadata/
│   └── data_dict.csv
├── indicator_selection/
│   ├── gdp_target.csv
│   ├── selection_matrix_blockbalanced_k20.csv
│   ├── selection_matrix_pls.csv
│   └── dfm_input_sets/
│       └── en_only_selection_matrix.csv
└── nowcasting/
    ├── ifocast_indicator_mapping.csv
    ├── rmsfe_table_all_models.csv
    ├── horizon_profile_table.csv
    ├── horizon_bias_variance_table.csv
    ├── post_covid_benchmarks_table.csv
    ├── diebold_mariano_table_all_models.csv
    ├── model_confidence_set_table.csv
    ├── mincer_zarnowitz_table.csv
    ├── sv_interval_calibration_table.csv
    ├── dfm_en_forecast_revision.csv
    ├── release_block_counterfactual_states.csv
    ├── factor_loading_m3_panel.csv
    ├── xgb_shap_importance.csv
    ├── nowcast_results_*.csv
    ├── nowcast_path_combo_equal.csv
    ├── category_contribs_*.parquet
    ├── series_contribs_*.parquet
    ├── ragged_edge_diagnostics/info_set_summary.csv
    └── _scratch/
        ├── xgb_sensitivity_summary.csv
        ├── xgb_sensitivity_jackknife_postcovid.csv
        └── xgb_sensitivity_dm_vs_rolling_ar1.txt
```

## Rebuild the demo sample

```bash
python3 scripts/generate_demo_data.py
```

This always writes to `data/demo/`, even if `data/real/` exists. See
[`docs/DATA.md`](../docs/DATA.md) for why the real results are not in git.

## Stage real outputs locally

Point the staging script at the pipeline root (the folder that contains
`outputs/` and `data/metadata/`):

```bash
python3 scripts/stage_real_data.py --source /path/to/german-gdp-nowcasting
```

## Deploy with real data without committing it

[`src/dashboard/bootstrap.py`](../src/dashboard/bootstrap.py) can fill
`data/real/` at startup from a private data-only GitHub repo. Authenticate
with a fine-grained, read-only token via Streamlit secrets — see
[`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example).
If no secrets are set, the app stays in demo mode.
