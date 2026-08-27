"""Part II — Nowcasting results."""

from __future__ import annotations

import numpy as np
import streamlit as st

from .. import charts
from .. import config as C
from .. import data as D
from .. import theme as T


def render() -> None:
    T.eyebrow("Part II")
    st.markdown("# Nowcasting results")
    st.markdown(
        "All models are evaluated out-of-sample on first-release German GDP "
        "growth at the **final (M3) information set** of each quarter, over "
        "2011Q1–2025Q4. Accuracy is summarised by the root mean squared forecast "
        "error (RMSFE, in percentage points). Publication lags are enforced, but "
        "historical predictor revisions are not reconstructed."
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Accuracy & model paths",
        "Within-quarter accrual",
        "Benchmarks",
        "Interpretation & decomposition",
        "Significance & calibration",
        "Model specifications",
    ])
    with tab1:
        _accuracy_and_paths()
    with tab2:
        _horizon()
    with tab3:
        _benchmarks()
    with tab4:
        _decomposition()
    with tab5:
        _significance()
    with tab6:
        _model_specs()


def _accuracy_and_paths() -> None:
    rmsfe = D.rmsfe_by_regime()
    avail = D.accuracy_models()
    full_acc = D.full_window_accuracy()
    full_window_order = full_acc["model"].tolist()

    st.markdown("### Predictive accuracy by economic regime")
    regimes = st.multiselect(
        "Regimes", list(C.REGIMES), default=list(C.REGIMES),
        help="Compare how the model ranking changes from calm to crisis.",
    )
    models_acc = st.multiselect(
        "Models", avail, default=avail,
        format_func=C.model_label,
    )
    if regimes and models_acc:
        st.plotly_chart(
            charts.rmsfe_regime_bars(
                rmsfe, models_acc, regimes, full_window_order,
            ),
            width='stretch',
        )
    T.callout(
        "Model rankings vary by regime; the COVID window inflates squared-error "
        "metrics for all models. Compare models within each regime panel "
        "rather than treating a single ordering as stable across subsamples. "
        "**DFM · PLS inputs** is an appendix input-set sensitivity check "
        "(PLS+VIP top-30), not part of the headline horse-race."
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### Compare model nowcast paths")
    st.markdown(
        "Pick any set of models to overlay their quarterly nowcasts against "
        "realised GDP growth. Colours stay tied to each model family."
    )
    default_paths = [
        m for m in [
            "DFM-ifoCAST", "DFM-EN", "DFM-PLS", "DFM-TVP",
            "XGB-Full", "MLP-Factor", "AR1",
        ]
        if m in avail
    ]
    models_ts = st.multiselect(
        "Models to overlay", avail, default=default_paths,
        format_func=C.model_label, key="ts_models",
    )
    log_y = st.toggle(
        "Log y-axis", value=False,
        help="Signed-log compression — keeps negative growth readable while "
             "taming the 2020 spike.",
    )
    if models_ts:
        ts = D.nowcast_timeseries(tuple(models_ts))
        gdp = D.load_gdp_target()
        fig = charts.nowcast_timeseries(ts, models_ts, gdp, log_y=log_y)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Select at least one model to plot its nowcast path.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("### Full-window accuracy ranking")
    st.markdown(
        '<div class="rank-caption">Aggregate M3 performance over 2011Q1–2025Q4 '
        "(60 quarters). Lower RMSE and MAE indicate better accuracy; RMSE "
        "penalises the large 2020 misses more heavily than MAE. "
        "<b>Bias</b> is the mean signed error (positive = systematic "
        "over-prediction of growth).</div>",
        unsafe_allow_html=True,
    )
    if not full_acc.empty:
        rows = [
            {
                "rank": int(r["rmse_rank"]),
                "model": C.model_label(r["model"]),
                "family": r["family"],
                "rmse": float(r["rmse"]),
                "mae": float(r["mae"]),
                "bias": float(r["bias"]),
                "n": int(r["n"]),
            }
            for _, r in full_acc.iterrows()
        ]
        T.accuracy_ranking_table(rows)
    else:
        st.info("No model accuracy data available.")


def _horizon() -> None:
    st.markdown("### Within-quarter information accrual (M1 → M2 → M3)")
    st.markdown(
        "As the quarter progresses more monthly indicators are released. This "
        "traces RMSFE across the three monthly information sets, per regime."
    )
    df = D.load_horizon_profile()
    models = sorted(df["model"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        regime = st.radio("Regime", list(C.REGIMES), index=0)
    with c2:
        picked = st.multiselect("Models", models, default=models, key="hz_models")
    if picked:
        st.plotly_chart(charts.horizon_profile(df, picked, regime),
                        width='stretch')
    T.callout(
        "Within-quarter information helps before COVID and is decisive during "
        "COVID. After 2022, every displayed DFM has a higher average M3 than M1 "
        "RMSFE. These are window averages over 16 quarters, not evidence that "
        "every individual release or quarter is harmful."
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Why? Bias–variance decomposition of the M1 → M3 RMSFE")
    st.markdown(
        "RMSFE² splits into bias² (systematic over/under-prediction) plus "
        "variance (dispersion of errors around that bias). This uses the "
        "same regime and model selection as the chart above."
    )
    bv = D.load_horizon_bias_variance()
    if bv.empty:
        st.info(
            "Bias–variance table unavailable — run "
            "`scripts/run_horizon_bias_variance.py`."
        )
    elif picked:
        st.plotly_chart(charts.bias_variance_decomposition(bv, picked, regime),
                        width='stretch')
        T.callout(
            "After 2022, squared bias falls from M1 to M3 for every displayed "
            "DFM, but error variance rises by more. The decomposition identifies "
            "the statistical mechanism; it does not identify which releases "
            "caused it. Before and during COVID, error variance instead falls "
            "as the quarter progresses."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Which release block carries the post-2022 change?")
    block = D.load_release_block_states()
    if block.empty:
        st.info("Release-block counterfactual unavailable.")
    else:
        st.plotly_chart(
            charts.release_block_counterfactual(block),
            width="stretch",
        )
        T.callout(
            "<b>DFM-EN accounting result.</b> Updating the non-hard complement "
            "alone leaves RMSFE close to the M1-frozen value; allowing hard "
            "activity to update reproduces nearly all of the observed M2/M3 "
            "increase. This is a fitted-model counterfactual over 16 quarters, "
            "not a causal claim. The thesis' bootstrap intervals include zero, "
            "and three quarters account for much of the M3 increase."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Quarter-by-quarter revision path (DFM-EN)")
    rev = D.load_revision_path()
    if rev.empty:
        st.info("Revision diagnostics unavailable.")
    else:
        log_y = st.toggle(
            "Log y-axis", value=False, key="rev_log",
            help="Signed-log compression keeps the 2020 quarters from "
                 "flattening the rest of the sample.",
        )
        st.plotly_chart(charts.revision_band(rev, log_y=log_y),
                        width='stretch')
        med_rev = float(rev["abs_revision_M1_to_M3"].median())
        T.callout(
            "The shaded band spans the first (M1) and final (M3) nowcast of "
            "each quarter — its width is the within-quarter revision. The median "
            f"|M1→M3| revision is <b>{med_rev:.2f} pp</b> over the evaluation "
            "window."
        )


def _benchmarks() -> None:
    st.markdown("### Benchmark comparison by regime")
    st.markdown(
        "A focused comparison of the headline models against strong classical "
        "baselines and the equal-weight DFM combination. Lower bars are point "
        "estimates, not automatically significant differences."
    )
    df = D.load_post_covid()
    regime_cols = [c for c in df.columns if c.endswith("_rmsfe") and c != "all_rmsfe"]
    pretty = {c: c.replace("_rmsfe", "") for c in regime_cols}
    regime_pick = st.radio(
        "Regime", regime_cols, format_func=lambda c: pretty[c], horizontal=True,
    )
    st.plotly_chart(charts.post_covid_bars(df, regime_pick),
                    width='stretch')

    with st.expander("Full benchmark table (RMSFE & bias by regime)"):
        st.dataframe(df.set_index("model").round(3), width='stretch')

    st.markdown("<hr/>", unsafe_allow_html=True)
    _xgb_sensitivity(df)


def _xgb_sensitivity(post_covid_df) -> None:
    st.markdown("### Robustness of XGB-Full's post-COVID point estimate")
    st.markdown(
        "The headline seed gives XGB-Full the lowest post-COVID RMSFE among "
        "non-autoregressive models. The same loop is checked across five seeds, "
        "six one-at-a-time hyperparameter changes, a leave-one-quarter-out "
        "jackknife, and a Diebold–Mariano comparison with Rolling-AR(1) 40q."
    )

    sens = D.load_xgb_sensitivity()
    jk = D.load_xgb_sensitivity_jackknife()
    dm = D.load_xgb_sensitivity_dm()

    if sens.empty:
        st.info(
            "Sensitivity cache unavailable — run "
            "`outputs/nowcasting/_scratch/xgb_sensitivity.py`."
        )
        return

    dfm_keys = [
        "DFM-EN", "DFM-ifoCAST", "DFM-PLS", "DFM-BlockBalanced",
        "DFM-TVP", "DFM-SV-k2",
    ]
    regime_rmsfe = D.rmsfe_by_regime()
    dfm_row = regime_rmsfe[
        regime_rmsfe["model"].isin(dfm_keys)
        & regime_rmsfe["regime"].eq("post-COVID")
    ]
    dfm_vals = dfm_row.set_index("model")["rmsfe"]
    dfm_range = (float(dfm_vals.min()), float(dfm_vals.max()))
    best_dfm = (str(dfm_vals.idxmin()), float(dfm_vals.min()))
    ar1_row = post_covid_df[post_covid_df["model"] == "Rolling-AR(1) 40q"]
    rolling_ar1 = float(ar1_row["post-COVID_rmsfe"].iloc[0]) if not ar1_row.empty else np.nan

    seed_vals = sens.loc[sens["category"] == "Seed", "rmsfe_post"]
    hp_vals = sens.loc[sens["category"] == "Hyperparameter", "rmsfe_post"]
    cards = [
        (f"{seed_vals.min():.2f}–{seed_vals.max():.2f} pp",
         "post-COVID RMSFE range across 5 random seeds"),
        (f"{hp_vals.min():.2f}–{hp_vals.max():.2f} pp",
         "range across 6 hyperparameter perturbations"),
    ]
    if not jk.empty:
        cards.append((
            f"{jk['rmsfe_excl_quarter'].min():.2f}–{jk['rmsfe_excl_quarter'].max():.2f} pp",
            "jackknife range (drop any one of 16 quarters)",
        ))
    if dm is not None:
        cards.append((f"p = {dm['p_value']:.2f}",
                      "DM test vs. Rolling-AR(1) 40q (post-COVID)"))
    T.stat_cards(cards)

    if not np.isnan(rolling_ar1):
        st.plotly_chart(
            charts.xgb_sensitivity_bars(sens, dfm_range, best_dfm, rolling_ar1),
            width='stretch',
        )

    dm_text = ""
    if dm is not None:
        dm_text = (
            f" A Diebold–Mariano test of the headline run against "
            f"<b>Rolling-AR(1) 40q</b> over the same 16 post-COVID quarters "
            f"gives DM = {dm['DM']:.2f}, p = {dm['p_value']:.2f} — the accuracy "
            "gap over that naive benchmark is <b>not statistically "
            "significant</b>."
        )
    T.callout(
        "<b>Cautious reading.</b> Across seeds, post-COVID RMSFE ranges from "
        f"{seed_vals.min():.2f} to {seed_vals.max():.2f} pp; hyperparameter "
        f"changes span {hp_vals.min():.2f}–{hp_vals.max():.2f} pp. "
        "The point estimate is therefore seed-sensitive."
        f"{dm_text} "
        "XGB-Full is a useful non-linear benchmark, not a demonstrated "
        "post-COVID winner."
    )


def _decomposition() -> None:
    st.markdown("### How the DFM-EN nowcast is built and attributed")
    st.markdown(
        "The charts on this tab follow one chain: **which indicators are selected** "
        "(Part I) → **what the latent factors represent** (Stage 1) → **how factors "
        "link to GDP** (Stage 2) → **which categories moved the forecast** "
        "(decomposition). Each step uses the same economic category palette but "
        "answers a different question."
    )
    T.dfm_interpretation_flow()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### Stage 1 — Factor content")
    st.markdown(
        "At each M3 origin the DFM is re-fitted on the Elastic-Net-selected "
        "indicators only. The stacked areas show the **share of absolute "
        "indicator→factor loadings** within each factor (shares sum to 1). "
        "**Factor 1** (purple) is a clean real-activity composite — Turnover "
        "and Production together carry ~65% of its loading mass, hard data "
        "overall ~90%. **Factor 2** (blue) is a genuinely **mixed** factor: "
        "Surveys is the largest single category on average, but hard-data "
        "categories combined still dominate its loadings (see callout below)."
    )
    cat_df = D.load_factor_loading_categories()
    if cat_df.empty:
        st.warning(
            "Factor-loading cache unavailable. Run "
            "`python scripts/run_factor_loading_figure.py` in "
            "`04_nowcasting_dfm` to build `factor_loading_m3_panel.csv`."
        )
    else:
        st.plotly_chart(
            charts.factor_loading_category_stacks(cat_df),
            width="stretch",
        )
        T.callout(
            "These are estimated loading shares, not economic effects or "
            "contributions to a particular nowcast. Factor 1 is mainly hard "
            "activity. Factor 2 is mixed: surveys are its largest single category "
            "on average, while production, turnover and orders are larger in "
            "combination. Factors are identified only up to rotation and sign."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### Stage 2 — Factor → GDP transmission")
    st.markdown(
        "The DFM-TVP bridge maps the two latent factors to GDP growth. Loadings "
        "follow a **random walk** across quarters; COVID observations are "
        "down-weighted so the pandemic does not dominate the drift estimate."
    )
    tvp_df = D.load_tvp_m3_bridge()
    if tvp_df.empty:
        st.warning("TVP bridge cache unavailable (`nowcast_results_dfm_tvp.csv`).")
    else:
        st.plotly_chart(charts.tvp_bridge_loadings(tvp_df), width="stretch")
        T.callout(
            "The real-activity coefficient is positive at most origins and is "
            "lower on average after 2022; the second coefficient changes sign "
            "and averages near zero. Because Stage 1 is re-estimated recursively, "
            "read the sequence for broad level and drift—not quarter-to-quarter "
            "structural change or a standalone survey effect."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### DFM-TVP nowcast decomposition — intercept and remainder")
    st.markdown(
        "The **DFM-TVP** nowcast is `nowcast = intercept + λ₁·f₁ + λ₂·f₂`. "
        "The upper panel shows that split as levels: realised GDP, the nowcast, "
        "and the time-varying bridge intercept. The lower panel allocates only "
        "the remainder (`nowcast − intercept`) across indicator categories, in "
        "proportion to each factor's Stage-1 |loadings|. This is a descriptive "
        "allocation of the *fitted nowcast*, not a sequential news decomposition. "
        "Ifo's public ifoCAST, by contrast, attributes the *change between two "
        "successive forecasts* to incoming data groups."
    )
    df_tvp = D.load_contributions_tvp()
    if df_tvp.empty:
        st.warning(
            "DFM-TVP contribution cache unavailable. Run "
            "`python scripts/run_all_thesis_figures.py --rebuild-contrib-tvp` "
            "in `04_nowcasting_dfm` to build "
            "`category_contribs_tvp_2017_2025.parquet`."
        )
    else:
        tc1, tc2 = st.columns([2, 1])
        with tc1:
            tvp_period = st.selectbox(
                "Period", list(C.CONTRIB_PERIODS),
                help="The DFM-TVP decomposition cache spans 2017–2025.",
                key="tvp_decomp_period",
            )
        with tc2:
            tvp_view = st.radio(
                "View", ["Absolute (pp)", "Relative (%)"], horizontal=True,
                help="Absolute (pp): intercept, nowcast and GDP as levels in the "
                     "upper panel; lower-panel bars sum to nowcast − intercept. "
                     "Relative (%): rescales only the factor-driven remainder so "
                     "upward and downward category shares each sum to ±100 %.",
                key="tvp_decomp_view",
            )
        tvp_mode = "pct" if tvp_view.startswith("Relative") else "pp"
        tvp_start, tvp_end = C.CONTRIB_PERIODS[tvp_period]
        tvp_hovers = D.origin_category_hovers(
            tvp_start, tvp_end, top_n=C.DECOMP_HOVER_TOP_N,
            series_parquet=C.SERIES_CONTRIB_PARQUET_TVP,
        )
        st.plotly_chart(
            charts.contributions_tvp_bridge(
                df_tvp, tvp_start, tvp_end, mode=tvp_mode,
                origin_hovers=tvp_hovers,
            ),
            width="stretch",
        )
        T.callout(
            "The bridge intercept is an estimated forecast level, not an economic "
            "category. After 2022 it remains near +0.26 pp while realised growth "
            "averages near +0.03 pp; the TVP gain mainly comes from transmitting "
            "less factor variation, not fully learning the new level. Lower-panel "
            "bars allocate only the factor-driven remainder. Offset records "
            "large cross-category cancellation and keeps the display readable."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### DFM-EN nowcast decomposition — what moved the forecast?")
    st.markdown(
        "The EN-selected DFM nowcast is additively decomposed into the contribution "
        "(in percentage points) of each economic category at every monthly "
        "origin — revealing *which* blocks of indicators pushed the forecast up "
        "or down and by how much. This is the **forecast-attribution** step: "
        "it uses predicted indicator levels, not the Stage-1 loading shares above."
    )
    df = D.load_contributions()
    if df.empty:
        st.warning("Contribution cache unavailable.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        period = st.selectbox(
            "Period", list(C.CONTRIB_PERIODS),
            help="The decomposition cache spans 2017–2025.",
        )
    with c2:
        view = st.radio(
            "View", ["Absolute (pp)", "Relative (%)"], horizontal=True,
            help="Absolute (pp): signed category contributions in percentage "
                 "points; bars sum to the nowcast and GDP lines are shown. "
                 "Relative (%): each origin rescaled so upward and downward "
                 "contributions each sum to ±100 % — composition only, no "
                 "magnitude or GDP lines.",
        )
    mode = "pct" if view.startswith("Relative") else "pp"
    start, end = C.CONTRIB_PERIODS[period]
    origin_hovers = D.origin_category_hovers(start, end, top_n=C.DECOMP_HOVER_TOP_N)
    st.plotly_chart(
        charts.contributions_stacked(
            df, start, end, mode=mode, origin_hovers=origin_hovers,
        ),
        width='stretch',
    )
    if mode == "pp":
        T.callout(
            "Bars <b>above zero</b> push the nowcast higher; bars <b>below zero</b> "
            "pull it down. The black line is the resulting DFM-EN nowcast; the dotted "
            "line is realised GDP. Hover any bar to see the top contributing series "
            "within that category. Note: a bar labelled <b>Global</b> (purple) shows "
            "international indicators such as global climate surveys or world trade "
            "indices — their names may contain words like 'survey' or 'orders' because "
            "they are the international equivalents, not the domestic German categories. "
            "<br><br>"
            "A slate <b>Offset</b> bar appears only when selected indicators point in "
            "strongly opposite directions (their predicted moves largely cancel), which "
            "can otherwise force every category bar to be inflated by the same explosive "
            "factor to still reconcile to a small nowcast — e.g. ±80 pp bars for a 7 pp "
            "forecast. We cap any single category at 5× the nowcast in that situation and "
            "report the unattributable remainder as <b>Offset</b>, so bars stay on an "
            "economically readable scale while still summing exactly to the nowcast."
        )
    else:
        T.callout(
            "Positive shares sum to +100% and negative shares to −100% at each "
            "origin. This shows composition, not magnitude: a 40% slice is 40% "
            "of the same-sign fitted contribution, not 0.40 pp of GDP. It is "
            "forecast attribution—not causal inference, release news or selection "
            "mass. Use the absolute view for levels in percentage points."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### DFM-BlockBalanced nowcast decomposition — what moved the forecast?")
    st.markdown(
        "The **DFM-BlockBalanced** (k=20, ≥1 indicator per category) nowcast uses "
        "the identical fixed-loading DFM and predicted-level attribution as "
        "DFM-EN above, so there is no intercept line. The difference from DFM-EN "
        "is entirely in **which indicators are selected**: a parsimonious, "
        "category-balanced k=20 set rather than the unconstrained, larger EN set."
    )
    df_bb = D.load_contributions_blockbalanced()
    if df_bb.empty:
        st.warning(
            "DFM-BlockBalanced contribution cache unavailable. Run "
            "`python scripts/run_all_thesis_figures.py --rebuild-contrib-bb` "
            "in `04_nowcasting_dfm` to build "
            "`category_contribs_blockbalanced_2017_2025.parquet`."
        )
    else:
        bc1, bc2 = st.columns([2, 1])
        with bc1:
            bb_period = st.selectbox(
                "Period", list(C.CONTRIB_PERIODS),
                help="The DFM-BlockBalanced decomposition cache spans 2017–2025.",
                key="bb_decomp_period",
            )
        with bc2:
            bb_view = st.radio(
                "View", ["Absolute (pp)", "Relative (%)"], horizontal=True,
                help="Absolute (pp): signed category contributions in percentage "
                     "points; bars sum to the nowcast and GDP lines are shown. "
                     "Relative (%): each origin rescaled so upward and downward "
                     "contributions each sum to ±100 %.",
                key="bb_decomp_view",
            )
        bb_mode = "pct" if bb_view.startswith("Relative") else "pp"
        bb_start, bb_end = C.CONTRIB_PERIODS[bb_period]
        bb_hovers = D.origin_category_hovers(
            bb_start, bb_end, top_n=C.DECOMP_HOVER_TOP_N,
            series_parquet=C.SERIES_CONTRIB_PARQUET_BLOCKBALANCED,
        )
        st.plotly_chart(
            charts.contributions_stacked(
                df_bb, bb_start, bb_end, mode=bb_mode,
                origin_hovers=bb_hovers, model_label="DFM-BlockBalanced",
            ),
            width="stretch",
        )
        T.callout(
            "Compare this chart with DFM-EN directly above: same DFM, same "
            "attribution method, different input set. Differences in category "
            "mix therefore show selection sensitivity, not competing explanations "
            "of the economy. Offset has the same display-only role as above."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### What the three decompositions actually show")
    st.markdown(
        "DFM-EN and DFM-BlockBalanced allocate each fitted nowcast in proportion "
        "to predicted indicator levels. DFM-TVP allocates only the factor-driven "
        "remainder, in proportion to Stage-1 |loadings| times the drifting bridge. "
        "None of the three is a sequential news decomposition: Ifo's ifoCAST bars "
        "explain the revision between two successive forecasts, whereas these "
        "bars explain a nowcast *level* at one origin. Comparing the three on "
        "the same months shows how much of the 'story' is the economy, and how "
        "much is the attribution method and the selected input set."
    )
    T.callout(
        "<b>Bounded interpretation.</b> Category allocations change with the "
        "selected set and the attribution rule. They show what the fitted model "
        "used to construct a nowcast level; they do not identify sectors that "
        "caused GDP growth. The contribution window begins in 2017, not at the "
        "2011 evaluation start."
    )


def _significance() -> None:
    st.markdown("### Diebold–Mariano equal-accuracy tests")
    st.markdown(
        "Pairwise tests of whether two models' squared-error losses differ "
        "significantly within each evaluation regime. Low p-values (dark green) "
        "flag a statistically detectable difference; compare models **within** the selected "
        "regime rather than pooling the full 2011–2025 window."
    )
    regime_labels = {
        "pre-COVID": "2011Q1–2019Q4",
        "COVID": "2020Q1–2021Q4",
        "post-COVID": "2022Q1–2025Q4",
    }
    regime = st.radio(
        "Regime",
        list(C.REGIMES),
        format_func=lambda r: f"{r} ({regime_labels[r]})",
        horizontal=True,
        key="dm_regime",
    )
    n_q = D.dm_regime_n(regime)
    df = D.dm_matrix_by_regime(regime)
    st.plotly_chart(
        charts.dm_heatmap(
            df,
            title=f"Diebold–Mariano equal-accuracy test — {regime}",
            subtitle=(
                f"{regime_labels[regime]} · {n_q} quarters at M3 · "
                "HLN-corrected two-sided test on squared-error loss"
            ),
        ),
        width='stretch',
    )
    T.callout(
        "A low p-value is evidence against equal squared-error accuracy; a high "
        "p-value does not prove equality. Regime splits are short, especially "
        f"The COVID panel has only <b>{D.dm_regime_n('COVID')}</b> quarters, "
        "so read its tests and rankings as episode-specific evidence."
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Multiple-model comparison")
    mcs = D.load_model_confidence_set()
    if mcs.empty:
        st.info("Model confidence set unavailable.")
    else:
        st.plotly_chart(charts.model_confidence_set(mcs), width="stretch")
        retained = mcs["in_MCS"].astype(str).str.lower().eq("true")
        retained_count = int(retained.sum())
        if retained_count == len(mcs):
            mcs_result = "cannot eliminate any candidate as inferior"
        else:
            mcs_result = (
                f"retains {retained_count} of {len(mcs)} candidates"
            )
        T.callout(
            f"At the 10% elimination level, the procedure {mcs_result}. "
            "This multiplicity-aware result "
            "shows low precision; it does not imply equal population accuracy."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Forecast efficiency (Mincer–Zarnowitz)")
    st.markdown(
        "The Mincer–Zarnowitz regression — **actual GDP = α + β · nowcast + ε** — "
        "tests whether a model's predictions are *efficient*: an efficient forecast "
        "has an intercept of zero (no systematic bias) and a slope of one "
        "(correct scale). With this convention, **β > 1** means actual GDP moves "
        "more than the forecast, so the nowcast is too compressed; **β < 1** means "
        "the nowcast varies too much relative to the actual outcome. The forest plot shows each model's "
        "estimated α and β with conventional OLS 95% confidence bands. "
        "A **dark ring** marks models where the joint test H₀: α = 0, β = 1 is "
        "rejected at the 5% level — those models are statistically biased or "
        "miscalibrated in magnitude, regardless of their RMSFE ranking."
    )
    mz = D.load_mincer_zarnowitz()
    if mz.empty:
        st.info("Mincer–Zarnowitz table unavailable.")
    else:
        st.plotly_chart(charts.mz_forest(mz), width="stretch")
        T.callout(
            "Only the equal-weight combination avoids rejection of the joint "
            "efficiency null in the full 60-quarter sample. Most slopes above one "
            "reflect forecasts that under-reacted to the pandemic extremes. When "
            "the eight COVID quarters are removed, every reported slope falls "
            "below one, so the combination's apparent calibration is not stable "
            "across regimes. Efficiency and RMSFE answer different questions."
        )
        with st.expander("Full regression table"):
            st.dataframe(mz.set_index("model").round(4), width='stretch')

    sv = D.load_sv_calibration()
    if not sv.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Prediction-interval calibration (DFM-SV, integrated k=2)")
        row = sv.iloc[0]
        T.stat_cards([
            (f"{row['coverage_empirical']:.0%}",
             f"empirical coverage (nominal {row['coverage_nominal']:.0%})"),
            (f"{row['mean_width']:.2f} pp", "mean 90% interval width"),
            (f"{row['CRPS']:.3f}", "CRPS (lower is better)"),
        ])
        regime_coverage = D.sv_calibration_by_regime()
        coverage_note = ""
        if not regime_coverage.empty:
            st.plotly_chart(
                charts.sv_coverage_by_regime(regime_coverage),
                width="stretch",
            )
            by_regime = regime_coverage.set_index("regime")
            covid = by_regime.loc["COVID"]
            post = by_regime.loc["post-COVID"]
            coverage_note = (
                f" COVID coverage is {covid['coverage']:.1%} "
                f"({int(covid['covered'])}/{int(covid['n'])}); post-COVID "
                f"coverage is {post['coverage']:.1%} "
                f"({int(post['covered'])}/{int(post['n'])})."
            )
        T.callout(
            "The stochastic-volatility layer feeds back into the Kalman smoother, "
            "so the point nowcast can differ slightly from plain DFM-EN while the "
            "bands use the model's SV-consistent predictive standard deviation. "
            f"Pooled coverage is <b>{row['coverage_empirical']:.1%}</b>, close to "
            f"90% nominal, but the regime split is the relevant qualification."
            f"{coverage_note} Near-nominal pooled coverage can therefore conceal "
            "poor shock coverage and overly wide calm-period intervals."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Headline accuracy table")
    rmsfe_tbl = D.load_rmsfe_table().set_index("model")
    st.dataframe(rmsfe_tbl.round(4), width='stretch')
    T.callout(
        "<b>NSR</b> = noise-to-signal ratio (RMSFE relative to the GDP series' "
        "own volatility); <b>vs_AR1</b> rebases each RMSFE to the AR(1) "
        "baseline (&lt;1 means a lower point estimate). Over the full M3 window, "
        "<b>combo_equal</b> records the lowest RMSFE; its DM test against AR(1) "
        "does not reject equal accuracy."
    )


def _model_specs() -> None:
    st.markdown("### How each model is built")
    st.markdown(
        '<div class="spec-intro">Reference cards for every model family in the '
        "comparison: design choices, fixed hyperparameters, and — for the "
        "machine-learning benchmarks — exactly which specifications are reported "
        "and why. All models share one pseudo-real-time protocol below.</div>",
        unsafe_allow_html=True,
    )

    T.spec_protocol_panel("Shared real-time protocol", [
        ("Sample", "Monthly panel 1991M01–2025M12; evaluation 2011Q1–2025Q4 "
                   "(60 quarters)."),
        ("Information set", "Publication-lag mask on the transformed monthly "
                            "panel; unreleased months in Q(<i>t</i>) AR(<i>p</i>)-filled "
                            "(BIC lag selection) on the transformed series, "
                            "then back-transformed to raw levels before aggregation."),
        ("Target", "German real GDP, QoQ log-growth, <b>first release</b>."),
        ("Vintage scope", "Predictor publication lags are enforced, but historical "
                          "predictor revisions are not reconstructed."),
        ("Aggregation", "Back-transform the AR-filled panel to raw monthly "
                        "levels → compute quarterly <b>mean</b> of the three "
                        "months → re-transform to stationary growth rates "
                        "(identity for level-stationary series, Δln for "
                        "log-growth series)."),
        ("Headline horizon", "M3 (final within-quarter set); DFM also at M1/M2."),
    ])

    fam1, fam_tvp, fam2, fam3, fam4, fam5 = st.tabs([
        "DFM (A-CD-TPN)",
        "DFM-TVP",
        "DFM-SV",
        "XGBoost",
        "MLP-Factor",
        "Baselines",
    ])

    with fam1:
        _dfm_bg, _dfm_fg = C.model_badge("DFM-EN")
        T.spec_card(
            title="Dynamic Factor Model",
            family="DFM (A-CD-TPN)",
            accent=C.model_color("DFM-EN"),
            badge="Point forecast",
            badge_bg=_dfm_bg,
            badge_fg=_dfm_fg,
            description=(
                "Mixed-frequency EM-Kalman dynamic factor model "
                "(`statsmodels DynamicFactorMQ`) following the approximate, "
                "coordinate-descent, targeted-predictor-nowcasting (A-CD-TPN) "
                "design of Franjic &amp; Schweikert (2025): <b>A</b> = arithmetic "
                "aggregation, <b>CD</b> = coordinate descent (Elastic Net "
                "pre-selection), <b>TPN</b> = targeted-predictor nowcasting. "
                "Factors summarise the pre-selected panel; the Kalman smoother "
                "delivers the GDP nowcast and a model-consistent predictive SD."
            ),
            rows=[
                ("Factors", "r = 2 common factors (headline)."),
                ("Factor dynamics", "VAR / AR(2) on the latent factors."),
                ("Idiosyncratic", "AR(1) components, internally standardised."),
                ("Estimation", "Expectation-Maximisation, cap of 200 iterations; "
                               "Kalman smoother for the state."),
            ],
            variants=[
                ("DFM-ifoCAST", "Fixed 19-series operational reference set."),
                ("DFM-EN", "Elastic Net selection — primary data-driven configuration."),
                ("DFM-BlockBalanced", "EN with structural breadth: ≥1 per category, cap 20."),
                ("DFM-PLS", "Top 30 indicators by PLS variable importance."),
            ],
            variant_label="Four indicator sets",
        )

    with fam_tvp:
        _tvp_bg, _tvp_fg = C.model_badge("DFM-TVP")
        T.spec_card(
            title="Time-varying-parameter DFM",
            family="DFM-TVP",
            accent=C.model_color("DFM-TVP"),
            badge="Post-COVID break remedy",
            badge_bg=_tvp_bg,
            badge_fg=_tvp_fg,
            description=(
                "DFM-TVP keeps the <b>same Stage-1 factor extraction as DFM-EN</b> "
                "but replaces the fixed factor→GDP mapping with a <b>time-varying "
                "bridge</b>. The motivation is post-2022 stagnation: if the "
                "real-activity→GDP transmission weakens after the energy shock, "
                "a fixed-loading DFM stays anchored to pre-break coefficients. "
                "Here only the bridge drifts; factors are still extracted with the "
                "standard EM-Kalman DFM on EN-selected indicators. COVID quarters "
                "are down-weighted when estimating the bridge (Lenza &amp; "
                "Primiceri 2022), so the loadings are not pulled by the 2020 "
                "swings. The two-step design follows Bates et al. (2013): factor "
                "estimates remain usable even when the GDP transmission drifts."
            ),
            rows=[
                ("Stage 1", "Identical to DFM-EN: r = 2 factors, mixed-frequency "
                            "EM-Kalman on the EN input set at each origin."),
                ("Stage 2", "Quarterly bridge from current-quarter factor averages "
                            "f<sub>q</sub> to GDP growth; intercept and loadings "
                            "follow a random walk."),
                ("Drift speed", "Profile MLE chooses q_ratio ∈ [10<sup>−6</sup>, 1] "
                                "each origin; q_ratio → 0 recovers a fixed bridge."),
                ("COVID robustness", "Observation variance inflated 100× for "
                                    "2020Q1–2021Q4 in Stage 2 only."),
                ("Nowcast", "Uses the random-walk forecast of bridge coefficients "
                            "times the current-quarter factor vector."),
                ("Empirical role", "Lower post-COVID point error than DFM-EN, but "
                                   "worse during COVID and over the full sample."),
            ],
            variants=[
                ("DFM-TVP", "EN inputs · random-walk bridge · COVID-down-weighted "
                            "Stage 2 (reported variant)."),
            ],
            variant_label="Reported variant",
        )
        T.spec_formula_panel("Model equations", [
            (
                "Stage 1 — factor extraction (same as DFM-EN)",
                "x<sub>t</sub> = Λ f<sub>t</sub> + u<sub>t</sub>, &nbsp; "
                "f<sub>t</sub> ~ VAR/AR(2), &nbsp; u<sub>t</sub> ~ AR(1)",
                "Monthly indicators x<sub>t</sub> are summarised by k = 2 latent "
                "factors using the same real-time information set as DFM-EN.",
            ),
            (
                "Stage 2 — measurement equation (time-varying bridge)",
                "y<sub>q</sub> = α<sub>q</sub> + λ<sub>1,q</sub> F<sub>1,q</sub> "
                "+ λ<sub>2,q</sub> F<sub>2,q</sub> + ε<sub>q</sub>, &nbsp; "
                "ε<sub>q</sub> ~ N(0, σ² R<sub>q</sub>)",
                "y<sub>q</sub> is quarterly GDP growth; F<sub>1,q</sub>, F<sub>2,q</sub> "
                "are current-quarter averages of the smoothed monthly factors. "
                "R<sub>q</sub> = 100 during COVID, otherwise 1.",
            ),
            (
                "Stage 2 — state equation (random-walk coefficients)",
                "β<sub>q</sub> = [α<sub>q</sub>, λ<sub>1,q</sub>, λ<sub>2,q</sub>]′, "
                "&nbsp; β<sub>q</sub> = β<sub>q−1</sub> + η<sub>q</sub>, &nbsp; "
                "η<sub>q</sub> ~ N(0, σ² Q*)",
                "Q* = q_ratio · I controls how fast the bridge can adapt. "
                "q_ratio is chosen by concentrated maximum likelihood at each origin.",
            ),
            (
                "Nowcast at origin t",
                "ŷ<sub>q|t</sub> = z<sub>q|t</sub>′ β<sub>q|t−1</sub>, &nbsp; "
                "z<sub>q|t</sub> = [1, F<sub>1,q|t</sub>, F<sub>2,q|t</sub>]′",
                "The bridge coefficients are forecast forward with the random walk "
                "(= last filtered state), then applied to the current-quarter factors.",
            ),
        ])
        T.callout(
            "<b>How this differs from DFM-EN:</b> DFM-EN fixes the factor→GDP "
            "transmission inside the state-space model. DFM-TVP extracts the same "
            "factors but lets only the <i>bridge</i> drift, targeting breaks in "
            "the GDP mapping after 2022. The Stage-1 factor-loading plots in the "
            "Interpretation tab describe what each factor <i>is</i>; the Stage-2 "
            "TVP plot shows how strongly each factor <i>transmits</i> into GDP."
        )

    with fam2:
        _sv_bg, _sv_fg = C.model_badge("DFM-SV-k2")
        T.spec_card(
            title="Integrated DFM with stochastic volatility",
            family="DFM-SV",
            accent=C.model_color("DFM-SV-k2"),
            badge="Point + intervals",
            badge_bg=_sv_bg,
            badge_fg=_sv_fg,
            description=(
                "DFM-EN augmented with a Bayesian stochastic-volatility (SV) layer "
                "(VAR on the factors, AR(1) log-volatility per factor innovation, "
                "estimated by NUTS), <b>fed back into the Kalman smoother</b> via a "
                "time-varying factor-innovation covariance (Doz, Giannone &amp; "
                "Reichlin 2011; Marcellino, Porqueddu &amp; Venditti 2016). Unlike a "
                "two-stage SV layer that only rescales the prediction band, the "
                "volatility here reaches the Kalman gain, so the "
                "<b>point nowcast can differ from plain DFM-EN</b> — mostly in "
                "high-volatility episodes (2008-09, 2020) — while calibrating "
                "prediction intervals from the model's own SV-consistent "
                "predictive standard deviation."
            ),
            rows=[
                ("Base", "EN-only input set; iterated two-step EM-DFM + SV "
                         "re-smoothing (Doz-Giannone-Reichlin 2011)."),
                ("Factors", "k = 2 common factors, matching the DFM-EN headline "
                            "specification."),
                ("Volatility", "Univariate AR(1) log-volatility per factor "
                               "innovation; relative-variance path scales only "
                               "the factor block of the state covariance."),
                ("Feedback", "Time-varying Q_t re-injected into the Kalman "
                             "smoother, so the common factor reweights "
                             "observations in turbulent months."),
                ("Inference", "Hamiltonian Monte-Carlo (NUTS)."),
            ],
        )

    with fam3:
        _xgb_bg, _xgb_fg = C.model_badge("XGB-Full")
        T.spec_card(
            title="XGBoost — XGB-Full",
            family="Machine learning",
            accent=C.model_color("XGB-Full"),
            badge="Non-linear benchmark",
            badge_bg=_xgb_bg,
            badge_fg=_xgb_fg,
            description=(
                "Gradient-boosted regression trees (Chen &amp; Guestrin 2016) on a "
                "wide lag-expanded feature matrix. The workflow below details "
                "how the full panel, SHAP pruning, and GDP lags are combined "
                "under the shared pseudo-real-time protocol."
            ),
            rows=[
                ("Learner", "XGBRegressor, squared-error loss, histogram trees."),
                ("Tuning", "Once on 1991Q1–2010Q4 (core set + GDP lags); fixed "
                           "for all 60 origins."),
                ("Features", "≈580 series × lags L0–L2; SHAP screen every 4 origins."),
                ("Intervals", "Empirical 90% bands from past forecast errors."),
            ],
        )
        T.xgb_flow()

    with fam4:
        _mlp_bg, _mlp_fg = C.model_badge("MLP-Factor")
        T.spec_card(
            title="Factor-augmented MLP",
            family="Machine learning",
            accent=C.model_color("MLP-Factor"),
            badge="Non-linearity test",
            badge_bg=_mlp_bg,
            badge_fg=_mlp_fg,
            description=(
                "A deliberately small neural benchmark that isolates "
                "<b>non-linearity</b>. The panel is first compressed to the two "
                "estimated factors of the headline DFM-EN (read-only refit at each "
                "origin); a one-hidden-layer MLP then maps those factors (plus "
                "lags) to GDP. If it cannot beat the linear DFM, the factor→GDP "
                "link is effectively linear at this sample size. See the XGBoost "
                "tab for the tree-based benchmark on the wide panel."
            ),
            rows=[
                ("Learner", "MLPRegressor · 1 hidden layer, 16 tanh units, "
                            "L-BFGS · heavy L2 (alpha = 10)."),
                ("Tuning", "Once on ≤2010Q4 (TimeSeriesSplit grid over hidden "
                           "units × alpha); fixed for all 60 origins."),
                ("Features", "DFM-EN factors F1, F2 at lags L0–L2 (6 features); "
                             "no GDP lags."),
                ("Prediction", "Seed-averaged over 5 random initialisations."),
            ],
            variants=[
                ("MLP · factor-augmented", "Factors-only inputs — headline variant."),
            ],
            variant_label="Reported variant",
        )
        T.mlp_flow()

    with fam5:
        st.markdown(
            f"<div style='color:{C.SUBTLE};font-size:0.9rem;line-height:1.55;"
            "margin-bottom:0.85rem'>Classical benchmarks and the equal-weight "
            "combination, all evaluated on the same pseudo-real-time grid.</div>",
            unsafe_allow_html=True,
        )
        T.spec_baseline_grid([
            ("RW", "Random walk: ŷ_q = y_{q-1} (Stock &amp; Watson 2002)."),
            ("AR(1) expanding", "Direct AR(1) on GDP growth, OLS-refit on an "
                                "expanding 1991Q1+ window every origin."),
            ("Rolling-AR(1)", "AR(1) on a rolling 40-quarter window — "
                              "break-robust alternative (post-COVID figure)."),
            ("AR(1) + IC", "Expanding AR(1) with recursive intercept correction."),
            ("combo_equal", "Equal-weight average of DFM-ifoCAST, DFM-EN and "
                             "DFM-BlockBalanced at each origin."),
        ])

    T.callout(
        "Every model is evaluated on the <b>same</b> first-release GDP target "
        "and publication-lag protocol. Comparisons remain pseudo-real-time because "
        "predictor revisions are not simulated."
    )
