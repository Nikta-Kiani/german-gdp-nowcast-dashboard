# Data directory

The dashboard resolves its data directory in this order (see
[`src/dashboard/config.py`](../src/dashboard/config.py)):

1. **`DASHBOARD_DATA_DIR` environment variable** — an absolute path to a
   directory with the same layout as `demo/` below.
2. **`data/real/`** — if you have your own results, drop them here. This
   folder is gitignored on purpose; it never gets committed.
3. **`data/demo/`** — the synthetic sample bundled with the repo. Used
   automatically if neither of the above exists, so the app always runs.

## Layout expected under `real/` (or your `DASHBOARD_DATA_DIR` target)

```text
<data_dir>/
├── metadata/
│   └── data_dict.csv                  # id, name, category
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
    ├── nowcast_results_*.csv          # one file per model, see config.py MODELS
    ├── nowcast_path_combo_equal.csv
    ├── category_contribs_*.parquet
    ├── series_contribs_*.parquet
    ├── ragged_edge_diagnostics/info_set_summary.csv
    └── _scratch/
        ├── xgb_sensitivity_summary.csv
        ├── xgb_sensitivity_jackknife_postcovid.csv
        └── xgb_sensitivity_dm_vs_rolling_ar1.txt
```

## Regenerating the demo sample

```bash
python scripts/generate_demo_data.py
```

This rebuilds `data/demo/` from a seeded random-number generator — no real
data required. See [`docs/DATA.md`](../docs/DATA.md) for why the real results
aren't in this repository.

## Staging real data locally

Point the staging script at the research pipeline root containing `outputs/`
and `data/metadata/`:

```bash
python scripts/stage_real_data.py --source /path/to/pipeline-root
```

## Deploying with real data (without committing it anywhere public)

`src/dashboard/bootstrap.py` can sync `data/real/` at process startup from a
*separate, private, data-only* GitHub repo, authenticated with a
fine-grained, read-only token supplied via Streamlit secrets — see
[`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example). The
real data and its access token never touch this codebase; only the
deployment's own secrets store sees them. No secrets configured → no-op →
demo mode, exactly like a plain `git clone`.
