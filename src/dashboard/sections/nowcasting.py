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
        "error (RMSFE, in percentage points)."
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Accuracy & model paths",
        "Within-quarter accrual",
        "Benchmark horse-race",
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
            charts.rmsfe_regime_bars(rmsfe, models_acc, regimes),
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
    full_acc = D.full_window_accuracy()
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
        "Within-quarter accrual does not improve accuracy uniformly across "
        "regimes. In <b>pre-COVID</b> and <b>COVID</b>, most DFM variants show "
        "RMSFE falling from M1 to M3; in <b>post-COVID</b>, several DFM models "
        "record higher M3 than M1 RMSFE — inspect the selected models in the "
        "chart above."
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
            "Post-COVID, bias² shrinks towards zero from M1 to M3 for every "
            "DFM variant — incoming data keeps correcting the AR-anchored "
            "over-prediction, exactly as expected. But the <b>variance</b> "
            "component more than doubles over the same window, and it is "
            "this variance inflation — not the bias — that pushes M3 RMSFE "
            "above M1. Late-arriving, lag-2 hard-data releases are volatile "
            "relative to the near-flat post-COVID growth rate, so replacing "
            "their AR-bridge forecasts (Section on ragged-edge filling) with "
            "the true releases adds noise rather than signal. In "
            "<b>pre-COVID</b> and <b>COVID</b>, by contrast, variance falls "
            "sharply from M1 to M3, which is why RMSFE improves there."
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
    st.markdown("### Benchmark horse-race by regime")
    st.markdown(
        "A focused comparison of the headline models against strong classical "
        "baselines and the inverse-MSE ensemble, regime by regime."
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
    st.markdown("### Robustness check — how stable is XGB-Full's post-COVID edge?")
    st.markdown(
        "The chart shows **XGB-Full** recording the lowest post-COVID "
        "RMSFE of any model at its headline configuration (`seed = 42`). Before "
        "reading that as \"XGBoost wins post-COVID\", the same expanding-window "
        "nowcast loop was re-run **11 times**: under 5 random seeds (identical "
        "hyperparameter search, different tie-breaking / tree randomness) and "
        "6 one-at-a-time hyperparameter perturbations around the tuned values "
        "(`max_depth`, `learning_rate`, `n_estimators`), plus a **16-fold "
        "leave-one-quarter-out jackknife** of the headline run and a fresh "
        "**Diebold–Mariano** test against the strongest naive benchmark, "
        "Rolling-AR(1) 40q."
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

    dfm_keys = ["DFM-EN", "DFM-ifoCAST", "DFM-BlockBalanced", "DFM-TVP"]
    dfm_row = post_covid_df[post_covid_df["model"].isin(dfm_keys)]
    dfm_vals = dfm_row.set_index("model")["post-COVID_rmsfe"]
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
        "<b>Seed sensitivity is the real story here.</b> With every other "
        "choice held fixed, one alternative seed (<code>seed = 1</code>) pushes "
        "the post-COVID RMSFE from 0.25 pp up to 0.62 pp — worse than every DFM "
        "variant and close to the worst model on this tab. Hyperparameter "
        "perturbations are far milder: all 6 stay inside "
        f"{hp_vals.min():.2f}–{hp_vals.max():.2f} pp, comfortably below the "
        "DFM range shown as the shaded band. The 16-fold jackknife also stays "
        "tight (no single quarter drives the headline number), so the result "
        "is not an artefact of one lucky/unlucky observation."
        f"{dm_text} "
        "<br><br>"
        "<b>Honest reading:</b> XGB-Full's post-COVID accuracy is "
        "<i>suggestive, not decisive</i> — it depends materially on the "
        "specific random seed used for its (already-fixed) hyperparameters, "
        "and it does not statistically beat a naive rolling AR(1). Treat it as "
        "one promising non-linear benchmark result among several regime "
        "comparisons on this page, not as a headline finding that XGBoost "
        "dominates in the post-COVID regime."
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
            "These are <b>structural</b> loadings — they describe what each factor "
            "<i>is</i>, not how much it moved the nowcast. Category shares can "
            "spike when very few series are selected at a given origin. "
            "<b>Factor 2 is a mixed factor:</b> across the full sample, Surveys is its single largest category (≈31% of "
            "|loading| mass on average) but Turnover (≈26%), Orders (≈22%) and "
            "Production (≈17%) together still make up roughly two-thirds of its "
            "loadings — Surveys is only the largest single category in half of "
            "the M3 origins shown. Read Factor 2 as a secondary, more balanced "
            "demand/sentiment composite rather than a soft-data-only factor. "
            "Shaded band: COVID window; dashed line: 2022 stagnation onset."
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
            "Before 2020, <b>λ<sub>1</sub></b> (Factor 1, real activity) carried "
            "most of the GDP transmission (mean ≈0.22 pre-COVID). The post-2022 "
            "period shows a weaker but still positive Factor-1 link (mean ≈0.16) "
            "— consistent with the stagnation regime. <b>λ<sub>2</sub></b> "
            "(Factor 2, the mixed demand/sentiment factor) hovers near zero on "
            "average in every regime with comparatively high volatility, so it "
            "is best read as a conditional, second-order bridge coefficient — "
            "not a standalone survey effect, since Factor 2 itself is not "
            "survey-dominated (see Stage 1 above)."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### DFM-TVP nowcast decomposition — what moved the forecast?")
    st.markdown(
        "The **DFM-TVP** nowcast is decomposed the same way as DFM-EN below, but "
        "through the time-varying bridge: `nowcast = intercept + λ₁·f₁ + λ₂·f₂`. "
        "Each factor's contribution (λ·f) is split across economic categories in "
        "proportion to that factor's Stage-1 |loadings| — so the bars reflect the "
        "post-COVID **loading drift** shown above. The grey <b>Baseline</b> bar is "
        "the bridge intercept, so all bars still sum to the nowcast line.",
        unsafe_allow_html=True,
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
                help="Absolute (pp): signed category contributions in percentage "
                     "points; bars (incl. Baseline) sum to the nowcast and GDP "
                     "lines are shown. Relative (%): each origin rescaled so "
                     "upward and downward contributions each sum to ±100 %.",
                key="tvp_decomp_view",
            )
        tvp_mode = "pct" if tvp_view.startswith("Relative") else "pp"
        tvp_start, tvp_end = C.CONTRIB_PERIODS[tvp_period]
        tvp_hovers = D.origin_category_hovers(
            tvp_start, tvp_end, top_n=C.DECOMP_HOVER_TOP_N,
            series_parquet=C.SERIES_CONTRIB_PARQUET_TVP,
        )
        st.plotly_chart(
            charts.contributions_stacked(
                df_tvp, tvp_start, tvp_end, mode=tvp_mode,
                origin_hovers=tvp_hovers, model_label="DFM-TVP",
            ),
            width="stretch",
        )
        T.callout(
            "This is the <b>DFM-TVP</b> counterpart of the DFM-EN decomposition "
            "below. Because the factor→GDP loadings drift (Stage 2 above), the "
            "category mix here can differ from DFM-EN even though both use the same "
            "Elastic-Net indicator set and Stage-1 factors. The grey <b>Baseline "
            "(intercept)</b> bar captures the level the bridge assigns before any "
            "factor movement; category bars plus Baseline sum to the black nowcast "
            "line, the dotted line is realised GDP. Hover any bar for the top "
            "contributing series within that category. "
            "<br><br>"
            "A slate <b>Offset</b> bar appears only in a handful of origins where "
            "λ₁·f₁ and λ₂·f₂ are individually large but largely cancel — e.g. one "
            "factor pulling +40 pp while the other pulls −38 pp for a +2 pp "
            "nowcast. Splitting such a small net effect cleanly across categories "
            "in proportion to each factor's loadings would produce category bars "
            "many times the size of the nowcast itself, which is not economically "
            "meaningful. We therefore cap any single category at 5× the nowcast "
            "and book the rest as <b>Offset</b> — a transparent flag for \"large, "
            "mostly self-cancelling factor swings\" rather than an inflated, "
            "misleading category bar."
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
            "<b>What this plot shows:</b> At each monthly forecast origin, the "
            "DFM-EN nowcast is split into category-level contributions in "
            "percentage points (pp). This <b>relative</b> view rescales those "
            "contributions <i>within each origin</i>: all categories pushing the "
            "nowcast <b>up</b> are re-expressed as shares that sum to "
            "<b>+100 %</b>; all categories pulling it <b>down</b> sum to "
            "<b>−100 %</b>. Bar height is therefore a <i>composition</i>, not a "
            "magnitude — a 40 % Surveys slice means surveys account for 40 % of "
            "the upward (or downward) push at that date, not that surveys added "
            "0.40 pp to GDP. "
            "<br><br>"
            "<b>How to read it economically:</b> Use this view to ask "
            "<i>which information blocks are driving the forecast mix?</i> "
            "Compare bar colours across time: if pink (Surveys) grows in the "
            "positive stack in calm quarters, timely sentiment is carrying more "
            "of the upward revision; if yellow/green (Turnover, Production, Orders) "
            "expand in the negative stack during COVID, hard-activity releases "
            "are pulling the nowcast down as the contraction becomes visible. "
            "Because magnitudes are normalised away, this is the right view for "
            "regime shifts and stress episodes when one category would otherwise "
            "dominate the absolute (pp) chart. "
            "<br><br>"
            "<b>What it is not:</b> This is <i>forecast attribution</i> through "
            "the DFM — not causal inference and not indicator selection mass "
            "(see Part I for that). It also does not show the level of the "
            "nowcast or realised GDP; switch to <b>Absolute (pp)</b> to read "
            "how large the forecast is and how much each category moved it in "
            "percentage points. Hover any bar for the top contributing series "
            "within that category at that origin."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### DFM-BlockBalanced nowcast decomposition — what moved the forecast?")
    st.markdown(
        "The **DFM-BlockBalanced** (k=20, ≥1 indicator per category) nowcast uses "
        "the identical fixed-loading DFM and predict()-based attribution as "
        "DFM-EN above — so there is no intercept/Baseline bar here, unlike the "
        "DFM-TVP bridge. The difference from DFM-EN is entirely in **which "
        "indicators are selected**: a parsimonious, category-balanced k=20 set "
        "rather than the unconstrained, larger EN set. This is the specification "
        "that is most competitive in the **post-2022 stagnation** regime."
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
            "attribution method, only the input set differs. A more balanced "
            "category mix here (every block guaranteed ≥1 indicator) tends to "
            "damp any single category's dominance relative to DFM-EN, which is "
            "part of why block-balanced selection is more robust once the EN "
            "set drifts hard-data-heavy in the post-2022 stagnation regime. With "
            "only k=20 indicators, offsetting predictions concentrate in fewer "
            "series, so the slate <b>Offset</b> bar described above (5× leverage "
            "cap) triggers here somewhat more often than for DFM-EN — most "
            "visibly around the 2020 V-shaped rebound, where some indicators had "
            "already rebounded while others were still deeply negative. "
            "Hover any bar for the top contributing series within that category."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("#### Key economic findings from the decomposition")
    st.markdown(
        "The three panels above attribute the *same* economic reality through three "
        "different lenses — DFM-EN and DFM-BlockBalanced split the nowcast in "
        "proportion to each indicator's predicted level (ifoCAST-style), while DFM-TVP "
        "splits it in proportion to each factor's *structural* loadings. Comparing them "
        "side by side, on the same monthly origins, surfaces findings that a single "
        "decomposition would hide."
    )
    T.callout(
        "<b>1. Surveys dominate calm periods — but only under level-based attribution.</b> "
        "Pre-COVID, Surveys carry <b>86%</b> of DFM-EN's and <b>79%</b> of "
        "DFM-BlockBalanced's total attributed magnitude (magnitude-weighted share of "
        "|contribution| across 2017–2019): timely sentiment data moves more than hard "
        "data in normal times, and both models attribute by how much each indicator's "
        "predicted level moves. DFM-TVP tells a different story for the identical "
        "period: its structural Baseline (bridge intercept) absorbs <b>81%</b> of the "
        "mass, because pre-COVID growth sits close to trend and the loading-based "
        "residual splits mainly across Turnover/Production/Orders — hard data, not "
        "surveys. Which category \"explains\" calm-period growth is therefore as much a "
        "property of the attribution method as of the economy. "
        "<br><br>"
        "<b>2. COVID-19 Q2 2020: the same −10.6&nbsp;pp collapse, three different "
        "reaction speeds.</b> At the M1 origin (April 2020) all three nowcasts were "
        "still near zero (+0.36 EN, +0.23 BlockBalanced, +0.33 TVP) — none saw the "
        "shock a full quarter ahead, before any April survey or hard data had been "
        "released. By M2 (May), EN had moved to only −1.6&nbsp;pp (Surveys −0.6, "
        "Turnover −1.1), while BlockBalanced had already swung to −4.9&nbsp;pp, driven "
        "almost entirely by one selected survey collapsing (−8.2&nbsp;pp gross, partly "
        "reconciled by the leverage-capped Offset bar) — a direct illustration of how a "
        "k=20 input set concentrates the impact of a single indicator shock. TVP barely "
        "moved (+0.31&nbsp;pp): its random-walk factor loadings adjust too slowly to "
        "react within one month. By M3 (June), once May Turnover data arrived, EN "
        "reached −8.9&nbsp;pp (Turnover −6.5) and BlockBalanced overshot to "
        "−14.6&nbsp;pp (Turnover −10.6, past the actual −10.6&nbsp;pp print), while TVP "
        "reached only −3.6&nbsp;pp (Turnover just −1.5). This ordering — EN closest, "
        "BlockBalanced overshooting, TVP undershooting — matches the measured COVID "
        "RMSFE exactly: <b>1.92&nbsp;pp (EN) &lt; 2.13&nbsp;pp (BlockBalanced) &lt; "
        "3.23&nbsp;pp (TVP)</b>. The COVID-regime bias also flips sign: EN and TVP "
        "nowcasts average <b>+0.46&nbsp;pp</b> and <b>+1.11&nbsp;pp</b> too high "
        "(both still short of the crash on average), while BlockBalanced averages "
        "<b>−0.36&nbsp;pp</b> too low — its concentrated indicator set overshoots the "
        "downside as much as it misses elsewhere, and the errors net negative. "
        "<br><br>"
        "<b>3. Post-2022: which category \"explains\" the stagnation nowcast depends on "
        "which indicators were selected, not just on the DFM.</b> For DFM-EN, Surveys "
        "remain the largest block (48% of attributed magnitude, 2022–2025) but a new "
        "driver appears — <b>Global</b> (17%) — traced almost entirely to a single "
        "series, the <b>US ISM Services PMI</b>, consistent with Germany's "
        "export-sector exposure to US demand momentum through the energy-crisis and "
        "stagnation years. DFM-BlockBalanced shows a similar Surveys share (30%) but a "
        "<i>different</i> second driver: <b>Financial</b> (31%), traced to the German "
        "<b>VDAX-NEW volatility index</b> — a domestic risk-sentiment story, not a "
        "global-trade one — plus a Misc contribution from manufacturing capacity "
        "utilisation; Global barely registers (well under 5%). DFM-TVP again assigns "
        "most of the post-2022 mass to its Baseline intercept (61%), reflecting that "
        "near-zero trend growth is mostly a level shift the bridge absorbs rather than "
        "a factor-driven movement. Two economically plausible post-2022 narratives — "
        "external (US) demand sensitivity vs. domestic risk sentiment — are both "
        "visible in the data, but each surfaces only under one particular indicator-"
        "selection method, a caution against treating either single decomposition as "
        "\"the\" explanation. "
        "<br><br>"
        "<b>4. Hard data offsets surveys in about a third of months, not most of "
        "them.</b> Across all 108 DFM-EN monthly origins (2017–2025), the combined "
        "hard-activity contribution (Turnover + Orders + Production) has the "
        "<b>opposite sign</b> from the Surveys contribution in <b>34%</b> of origins "
        "(correlation ≈ −0.24) — a genuine but minority pattern of hard data "
        "correcting an initial survey-driven read, not the dominant dynamic; in the "
        "other two-thirds of origins hard and soft data agree in direction."
    )


def _significance() -> None:
    st.markdown("### Diebold–Mariano equal-accuracy tests")
    st.markdown(
        "Pairwise tests of whether two models' squared-error losses differ "
        "significantly within each evaluation regime. Low p-values (dark green) "
        "flag a meaningful accuracy gap; compare models **within** the selected "
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
        "Regime splits matter because the full sample mixes calm, crisis and "
        "stagnation periods. A large RMSFE gap against RW can still be "
        "statistically insignificant if almost all of it comes from a few "
        "extreme COVID quarters (high variance in the loss differential). "
        "Closer models such as <b>combo_equal</b> vs <b>DFM-EN</b> can reject "
        "more easily when the improvement is more consistent within the window. "
        f"The COVID panel has only <b>{D.dm_regime_n('COVID')}</b> quarters, "
        "so treat it as directional evidence rather than a definitive ranking."
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
        "estimated α and β with 95% confidence bands (HAC-robust standard errors). "
        "A **dark ring** marks models where the joint test H₀: α = 0, β = 1 is "
        "rejected at the 5% level — those models are statistically biased or "
        "miscalibrated in magnitude, regardless of their RMSFE ranking."
    )
    mz = D.load_mincer_zarnowitz()
    if mz.empty:
        st.info("Mincer–Zarnowitz table unavailable.")
    else:
        st.plotly_chart(charts.mz_forest(mz), width='stretch')
        T.callout(
            "<b>What to read from this plot:</b> The left panel tests systematic "
            "bias (α = 0); the right panel tests whether the model's predictions "
            "scale correctly with the outcome (β = 1). Because the regression is "
            "<i>actual on nowcast</i>, slopes above one indicate compressed "
            "forecasts; slopes below one indicate forecasts that are too volatile. "
            "<br><br>"
            "<b>Passing models (joint test not rejected at 5%):</b> "
            "<b>combo_equal</b> (β ≈ 1.12, α ≈ −0.03, joint p ≈ 0.12) is the only "
            "model that clears the 5% threshold — it has the lowest full-sample "
            "RMSFE (0.68 pp) <i>and</i> the closest efficiency to the ideal (α "
            "nearest zero, β nearest one) of any model shown. "
            "<br><br>"
            "<b>Borderline (joint test rejected, but not by much):</b> "
            "<b>DFM-EN</b> (β ≈ 1.20, α ≈ −0.07, joint p ≈ 0.018) and "
            "<b>DFM-SV-k2</b> (β ≈ 1.20, α ≈ −0.07, joint p ≈ 0.021) are both "
            "mildly compressed relative to realised GDP and now reject the joint "
            "test at 5% (they did not under the pre-cap EN selection), but are "
            "far closer to efficient than the models below and are less accurate "
            "in point terms than the combination (RMSFE ≈ 0.78 pp vs 0.68 pp). "
            "<br><br>"
            "<b>Failing models (joint test rejected):</b> "
            "<b>DFM-ifoCAST</b> (β ≈ 1.51) is <i>too compressed</i> — realised GDP "
            "moves more than its nowcasts, so the deviations from the mean would "
            "need to be scaled up. <b>DFM-BlockBalanced</b> (β ≈ 0.78) goes in the "
            "opposite direction: its nowcasts are <i>over-dispersed</i>, consistent "
            "with the breadth constraint adding categories whose movements do not "
            "always map one-for-one into GDP. "
            "<b>DFM-TVP</b> (β ≈ 2.04, α ≈ −0.45) also rejects strongly — "
            "forecasts are too compressed and slightly biased downward on the "
            "full sample. This reflects a deliberate trade-off: the drifting "
            "bridge and COVID down-weighting help post-COVID point accuracy "
            "(RMSFE 0.30) but worsen COVID-window performance and full-sample "
            "calibration. "
            "The two non-linear benchmarks show comparable miscalibration: "
            "<b>XGB-Full</b> (β ≈ 2.06, joint p ≈ 0.0002) and "
            "<b>MLP-Factor</b> (β ≈ 2.30, joint p < 0.0001) both have slopes more "
            "than twice the efficient value — their forecasts are strongly "
            "compressed relative to realised GDP movements. This is consistent with "
            "the learners smoothing or missing the extreme COVID tail rather than "
            "solving it with non-linearity. Notably the factor-augmented MLP, fed "
            "only the two DFM factors, fails in almost the same way as XGB on the "
            "wide panel. "
            "<br><br>"
            "<b>Key insight:</b> A model can rank well on RMSFE yet still fail the "
            "efficiency test — and vice versa. The equal-weight combination is the "
            "cleanest overall result because it combines the lowest RMSFE with a "
            "slope close to one and no rejection of forecast efficiency."
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
            (f"{row['RMSFE']:.3f} pp", "point RMSFE (SV-integrated point nowcast)"),
        ])
        T.callout(
            "The stochastic-volatility layer feeds back into the Kalman smoother, "
            "so the point nowcast can differ slightly from plain DFM-EN while the "
            "prediction intervals are calibrated from the model's own "
            "SV-consistent predictive standard deviation. "
            f"Empirical 90% coverage is <b>{row['coverage_empirical']:.0%}</b> "
            f"(nominal {row['coverage_nominal']:.0%}); mean interval width "
            f"<b>{row['mean_width']:.2f} pp</b>; CRPS <b>{row['CRPS']:.3f}</b>."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Headline accuracy table")
    rmsfe_tbl = D.load_rmsfe_table().set_index("model")
    st.dataframe(rmsfe_tbl.round(4), width='stretch')
    T.callout(
        "<b>NSR</b> = noise-to-signal ratio (RMSFE relative to the GDP series' "
        "own volatility); <b>vs_AR1</b> rebases each RMSFE to the AR(1) "
        "baseline (&lt;1 means it beats AR(1)). Over the full M3 window, "
        "<b>combo_equal</b> records the lowest RMSFE among the models listed."
    )


def _model_specs() -> None:
    st.markdown("### How each model is built")
    st.markdown(
        '<div class="spec-intro">Reference cards for every model family in the '
        "horse-race: design choices, fixed hyperparameters, and — for the "
        "machine-learning benchmarks — exactly which specifications are reported "
        "and why. All models share one real-time protocol below.</div>",
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
                ("DFM-ifoCAST", "ifo's published 21-indicator expert set (supervisor-confirmed mapping: 20 unique predictors)."),
                ("DFM-EN", "Elastic Net selection — primary data-driven configuration."),
                ("DFM-BlockBalanced", "EN with structural breadth: ≥1 per category, cap 20."),
            ],
            variant_label="Three indicator sets",
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
                ("Horse-race role", "Post-COVID RMSFE 0.30 pp vs 0.45 for DFM-EN, "
                                    "but worse in COVID and on the full sample — "
                                    "a regime-specific remedy, not the headline model."),
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
                "under the shared real-time protocol."
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
            "margin-bottom:0.85rem'>Classical benchmarks and the inverse-MSE "
            "combination, all evaluated on the same real-time grid.</div>",
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
        "and real-time information set. Differences in RMSFE therefore reflect "
        "modelling choices — not data handling."
    )
