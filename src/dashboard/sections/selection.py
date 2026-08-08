"""Part I — Indicator selection views.

Every method exposes a different native signal (binary inclusion for the linear
screens, mean |SHAP| for XGBoost, fixed membership for ifoCAST). They are
harmonised onto a common per-series importance mass and
read two ways: raw *category mass share* (composition of the selected set) and
the *deviation from availability* — selected share minus the category's share
of the candidate universe, in percentage points. The deviation view matters
because the universe is extremely survey-heavy, so raw shares partly echo what
exists rather than what a method prefers.
"""

from __future__ import annotations

import streamlit as st

from .. import charts
from .. import config as C
from .. import data as D
from .. import theme as T


def render() -> None:
    T.eyebrow("Part I")
    st.markdown("# Indicator selection")
    st.markdown(
        "Four time-varying selection signals and one fixed expert benchmark are "
        "compared on a common footing: **Elastic Net**, **block-balanced EN**, "
        "**PLS**, **XGBoost** mean&nbsp;|SHAP| and **ifoCAST**. Each native signal "
        "is converted to per-series importance mass and aggregated to economic "
        "categories."
    )
    tab1, tab2, tab3, tab4 = st.tabs([
        "Emphasis over time",
        "Regime shifts",
        "Method agreement",
        "Ragged edge & timeliness",
    ])

    with tab1:
        _structural_shift()
    with tab2:
        _regime_switching()
    with tab3:
        _consensus()
    with tab4:
        _ragged_edge()


# --------------------------------------------------------------------------- #
# Tab 1 — structural shift through time
# --------------------------------------------------------------------------- #
def _structural_shift() -> None:
    st.markdown("### Time-varying category emphasis")
    methods = [m for m in D.available_methods()
               if C.SELECTION_METHODS[m]["kind"] != "fixed"]
    method = st.selectbox(
        "Method", methods,
        index=methods.index("EN (raw)") if "EN (raw)" in methods else 0,
        help="Elastic Net is the primary screen (COVID-weighted, capped at 60 "
             "indicators per origin); XGBoost cross-checks it with non-linear "
             "importances.",
    )

    share = D.category_share_over_time_any(method)
    if share.empty:
        st.warning("Time-varying selection unavailable for this method.")
        return

    soft_hard = D.soft_hard_share_over_time(method)
    st.plotly_chart(
        charts.soft_hard_lines(
            soft_hard, D.universe_soft_hard(),
            title=f"{method} — soft vs hard data emphasis over time"),
        width='stretch')
    T.callout(
        "Solid lines show the share of the method's selected mass in <b>soft "
        "survey data</b> vs <b>hard real-activity data</b> (Orders, Turnover, "
        "Production, Construction, Trade). Dotted lines mark each block's share "
        "of the candidate universe. A solid line above its dotted line indicates "
        "over-selection relative to availability."
    )

    with st.expander("Full 11-category composition (stacked area)"):
        st.plotly_chart(charts.structural_shift_area(share), width='stretch')
        st.caption(
            "Raw composition of the selected set (bands sum to 100%). The wide "
            "rose band mainly reflects that Surveys are about two-thirds of the "
            "universe — the soft/hard chart above nets that base rate out."
        )

    with st.expander("Non-linear cross-check: XGBoost"):
        for m, note in [("XGBoost (SHAP)", "mean |SHAP| mass")]:
            if m == method:
                continue
            sh = D.soft_hard_share_over_time(m)
            if sh.empty:
                st.info(f"{m} importances unavailable.")
                continue
            st.plotly_chart(
                charts.soft_hard_lines(
                    sh, D.universe_soft_hard(),
                    title=f"{m} — soft vs hard emphasis ({note})", height=340),
                width='stretch')


# --------------------------------------------------------------------------- #
# Tab 2 — regime switching
# --------------------------------------------------------------------------- #
def _regime_switching() -> None:
    st.markdown("### Regime-level category emphasis")
    T.callout(
        "Does selection emphasis change between pre-COVID, COVID and post-COVID "
        "windows? The headline chart condenses the answer to the decision that "
        "matters for nowcasting: how strongly each method rotates toward "
        "<b>hard real-activity data</b> in each regime."
    )
    long = D.regime_soft_hard()
    if long.empty:
        st.warning("Regime data unavailable.")
    else:
        st.plotly_chart(
            charts.regime_rotation_bars(long, D.universe_soft_hard()),
            width='stretch',
        )
        T.callout(
            "Hard real-activity categories are over-weighted relative to the "
            "survey-dominated panel in every regime shown. The magnitude and "
            "direction of regime-to-regime shifts differ by method — compare "
            "the bars across panels and against the dotted availability lines."
        )

    st.markdown("#### Full category composition by regime")
    full = D.regime_category_share()
    if full.empty:
        st.info("Regime composition unavailable.")
    else:
        methods = full["method"].unique().tolist()
        st.plotly_chart(charts.regime_share_bars(full, methods), width='stretch')
        st.caption(
            "Shares of selected mass before availability adjustment — read "
            "shifts across panels (regimes) rather than the absolute level of "
            "the survey bar, which reflects the survey-heavy universe."
        )


# --------------------------------------------------------------------------- #
# Tab 3 — cross-method agreement
# --------------------------------------------------------------------------- #
def _consensus() -> None:
    st.markdown("### Method agreement diagnostics")
    n_ifo = len(D.ifocast_membership())
    ifo_note = (
        f" Includes <b>ifoCAST (fixed)</b> — {n_ifo} supervisor-confirmed "
        "predictors from the latest mapping table (live, not the static "
        "selcmp_task2 CSV)."
        if n_ifo
        else ""
    )
    T.callout(
        "Both heatmaps are computed on the fly from the current selection "
        "artefacts and "
        f"<code>ifocast_indicator_mapping.csv</code>.{ifo_note}"
    )
    metric = st.radio(
        "Agreement metric",
        ["Jaccard overlap (top-20 sets)", "Spearman ρ (whole universe)"],
        horizontal=True, label_visibility="collapsed",
    )
    if metric.startswith("Jaccard"):
        df = D.method_overlap_jaccard(top_n=20)
        st.plotly_chart(
            charts.agreement_heatmap(
                df, "Cross-method agreement — Jaccard overlap of top-20 indicators",
                "Jaccard"),
            width='stretch')
        T.callout(
            "Jaccard = shared ÷ union of each pair's <b>top-20</b> indicators. This "
            "is the most concrete agreement diagnostic because it asks whether methods "
            "select the same high-priority series, not only similar category masses."
        )
    else:
        df = D.cross_method_agreement()
        st.plotly_chart(
            charts.agreement_heatmap(
                df, "Cross-method agreement — Spearman rank correlation",
                "Spearman ρ"),
            width='stretch')
        T.callout(
            "<b>Spearman ρ</b> ranks the full 585-series universe, with zero mass for "
            "series a method does not use. Treat it as a broad similarity measure; the "
            "top-20 Jaccard view is more interpretable for specific indicator overlap."
        )


# --------------------------------------------------------------------------- #
# Tab 4 — ragged edge
# --------------------------------------------------------------------------- #
def _ragged_edge() -> None:
    st.markdown("### Availability is not the same as informativeness")
    df = D.load_ragged_edge()
    if df.empty:
        st.warning("Ragged-edge diagnostics unavailable.")
        return

    n_total = len(df)
    n_timely = int((df["pub_lag"] == 0).sum())
    n_delayed = int((df["pub_lag"] >= 1).sum())
    n_filled = int((df["status"] == "ar_filled").sum())
    T.stat_cards([
        (f"{n_total}", "series at origin"),
        (f"{n_timely}", "available same month (lag 0)"),
        (f"{n_delayed}", "delayed (lag ≥ 1)"),
        (f"{n_filled}", "AR-backfilled at the edge"),
    ])

    mat = D.publag_category_matrix()
    if not mat.empty:
        st.plotly_chart(charts.publag_composition(mat), width='stretch')

    T.callout(
        "At any forecast origin the panel has a <b>ragged edge</b>: timely (lag-0) "
        "series — overwhelmingly <b>surveys</b> — are already published, while hard "
        "real-activity series (Production, Orders, Turnover) arrive one to two months "
        "late and must be model-filled. A soft-data tilt in raw shares can therefore "
        "reflect <i>what is available in real time</i> rather than a judgement that "
        "surveys are more informative (Bańbura &amp; Rünstler 2011; Giannone, "
        "Reichlin &amp; Small 2008)."
    )
