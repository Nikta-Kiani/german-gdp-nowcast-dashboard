"""Plotly figure builders.

Each function returns a fully styled ``go.Figure``. Properties (bars vs lines vs
heatmaps), hover templates and axis labels are tailored per chart, while colours
are pulled from :mod:`config` so model/category hues stay consistent everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C

# COVID window shading on the nowcast path chart.
_COVID_TS_START, _COVID_TS_END = pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")


def _symlog(y: np.ndarray | pd.Series) -> np.ndarray:
    """Signed log1p transform — compresses spikes while keeping sign for negatives."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    finite = np.isfinite(y)
    pos = finite & (y >= 0)
    neg = finite & (y < 0)
    out[pos] = np.log1p(y[pos])
    out[neg] = -np.log1p(-y[neg])
    return out


def _symlog_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Tick positions (transformed) and labels (original pp) for a symlog axis."""
    candidates = sorted({
        -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10,
    })
    visible = [v for v in candidates if lo <= v <= hi]
    if not visible:
        visible = [lo, 0.0, hi] if lo < 0 < hi else [lo, hi]
    tickvals = _symlog(np.array(visible)).tolist()
    ticktext = [f"{v:g}" for v in visible]
    return tickvals, ticktext


def _cat_display(cat: str) -> str:
    return C.CATEGORY_DISPLAY.get(cat, cat)


# =========================================================================== #
# Part II — Nowcasting
# =========================================================================== #
def rmsfe_regime_bars(rmsfe_long: pd.DataFrame, models: list[str],
                      regimes: list[str]) -> go.Figure:
    """Grouped horizontal bars: RMSFE per model, one panel per regime."""
    fig = make_subplots(
        rows=1, cols=len(regimes), shared_yaxes=True, horizontal_spacing=0.04,
        subplot_titles=[
            f"<b>{r}</b><br><span style='font-size:11px;color:{C.SUBTLE}'>"
            f"{C.REGIMES[r][0]}–{C.REGIMES[r][1]}</span>" for r in regimes
        ],
    )
    ordered = [m for m in C.MODEL_ORDER if m in models]
    y_labels = [C.model_label(m) for m in ordered]
    for col, regime in enumerate(regimes, start=1):
        sub = rmsfe_long[rmsfe_long["regime"] == regime].set_index("model")
        vals = [sub["rmsfe"].get(m, np.nan) for m in ordered]
        colors = [C.model_color(m) for m in ordered]
        fig.add_trace(
            go.Bar(
                x=vals, y=y_labels, orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.2f}" if np.isfinite(v) else "" for v in vals],
                textposition="outside", textfont=dict(size=10, color=C.SUBTLE),
                cliponaxis=False, showlegend=False,
                hovertemplate=("<b>%{y}</b><br>" + regime +
                               "<br>RMSFE: %{x:.3f} pp<extra></extra>"),
            ),
            row=1, col=col,
        )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="RMSFE (pp)", ticksuffix="", showgrid=True,
                     title_standoff=12)
    fig.update_layout(
        height=480, bargap=0.25,
        margin=dict(t=132, b=55, l=140, r=40),
        title=dict(
            text="Predictive accuracy by economic regime<br>"
                 "<span style='font-size:13px;color:%s'>RMSFE at the final "
                 "(M3) information set, by regime</span>" % C.SUBTLE,
            y=0.95, yanchor="top",
        ),
    )
    # Larger top margin opens a clear gap between the figure title and the
    # per-regime subplot titles (which sit at the top of the plotting area).
    for ann in fig.layout.annotations[:len(regimes)]:
        ann.update(font=dict(size=13, color=C.INK))
    return fig


def nowcast_timeseries(ts_long: pd.DataFrame, models: list[str],
                       gdp: pd.DataFrame, log_y: bool = False) -> go.Figure:
    """Quarterly nowcast paths vs realised GDP growth, one line per model."""
    fig = go.Figure()
    fig.add_vrect(
        x0=_COVID_TS_START, x1=_COVID_TS_END,
        fillcolor=C.REGIME_COLORS["COVID"], opacity=0.10, line_width=0,
        layer="below",
    )

    def _plot_y(raw: pd.Series) -> np.ndarray:
        y = raw.astype(float).to_numpy()
        return _symlog(y) if log_y else y

    if not gdp.empty:
        g = gdp[(gdp["date"] >= "2011-01-01")].copy()
        y_raw = g["gdp"].astype(float)
        fig.add_trace(go.Scatter(
            x=g["date"], y=_plot_y(y_raw),
            customdata=y_raw.to_numpy(),
            name="Realised GDP",
            mode="lines", line=dict(color=C.ACTUAL_COLOR, width=2.6),
            hovertemplate=(
                "<b>Realised GDP</b><br>%{x|%Yq%q}: "
                "%{customdata:.2f} pp<extra></extra>"
            ),
        ))
    for m in [x for x in C.MODEL_ORDER if x in models]:
        sub = ts_long[ts_long["model"] == m].sort_values("date")
        if sub.empty:
            continue
        y_raw = sub["nowcast"].astype(float)
        fig.add_trace(go.Scatter(
            x=sub["date"], y=_plot_y(y_raw),
            customdata=y_raw.to_numpy(),
            name=C.model_label(m),
            mode="lines+markers",
            line=dict(color=C.model_color(m), width=1.9),
            marker=dict(size=4),
            hovertemplate=(
                "<b>" + C.model_label(m) + "</b><br>"
                "%{x|%Yq%q}: %{customdata:.2f} pp<extra></extra>"
            ),
        ))

    y_title = "GDP growth (pp, symlog)" if log_y else "GDP growth (pp)"
    fig.update_layout(
        height=480, hovermode="x unified",
        title="Nowcast vs realised GDP growth (quarter-on-quarter, %)",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=70, r=30),
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text=y_title)

    if log_y:
        raw_vals: list[float] = []
        if not gdp.empty:
            g = gdp[gdp["date"] >= "2011-01-01"]
            raw_vals.extend(g["gdp"].dropna().tolist())
        for m in models:
            sub = ts_long[ts_long["model"] == m]
            raw_vals.extend(sub["nowcast"].dropna().tolist())
        if raw_vals:
            lo, hi = float(min(raw_vals)), float(max(raw_vals))
            tickvals, ticktext = _symlog_ticks(lo, hi)
            fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return fig


def horizon_profile(df: pd.DataFrame, models: list[str], regime: str) -> go.Figure:
    """RMSFE across the within-quarter information sets M1 -> M2 -> M3."""
    fig = go.Figure()
    sub = df[df["regime"] == regime]
    miq_label = {1: "M1", 2: "M2", 3: "M3"}
    for m in models:
        s = sub[sub["model"] == m].sort_values("month_in_quarter")
        if s.empty:
            continue
        color = _dfm_table_color(m)
        fig.add_trace(go.Scatter(
            x=[miq_label[i] for i in s["month_in_quarter"]],
            y=s["RMSFE"], name=m, mode="lines+markers",
            line=dict(color=color, width=2.4), marker=dict(size=8),
            hovertemplate="<b>" + m + "</b><br>%{x}: %{y:.3f} pp<extra></extra>",
        ))
    fig.update_layout(
        height=440,
        title=f"Within-quarter information accrual — {regime}",
        legend=dict(orientation="v", y=1.0, yanchor="top", x=1.015,
                    xanchor="left", font=dict(size=11),
                    title=dict(text="Model", font=dict(size=11.5,
                                                       color=C.SUBTLE))),
        margin=dict(t=70, b=60, l=70, r=170),
    )
    fig.update_xaxes(title_text="Information set (month in quarter)")
    fig.update_yaxes(title_text="RMSFE (pp)")
    return fig


_BIAS_COLOR = "#C9617F"
_VARIANCE_COLOR = "#3D5FAE"


def bias_variance_decomposition(df: pd.DataFrame, models: list[str],
                                regime: str) -> go.Figure:
    """Stacked bias^2 / variance decomposition of RMSFE^2 at M1, M2, M3.

    One subplot per model; each bar is RMSFE^2 = bias^2 + variance for that
    month-in-quarter, so the total bar height is comparable to the squared
    RMSFE line in the accrual chart above, but split into how much of it is a
    systematic offset (bias^2) versus dispersion around that offset (variance).
    """
    sub = df[(df["regime"] == regime) & (df["model"].isin(models))]
    shown = [m for m in models if m in set(sub["model"])]
    if not shown:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=len(shown), shared_yaxes=True, horizontal_spacing=0.045,
        subplot_titles=[f"<b>{m}</b>" for m in shown],
    )
    month_order = ["M1", "M2", "M3"]
    for col, m in enumerate(shown, start=1):
        s = sub[sub["model"] == m].set_index("month_in_quarter").reindex(month_order)
        show_leg = col == 1
        fig.add_trace(go.Bar(
            x=month_order, y=s["bias_sq"], name="Bias²",
            marker_color=_BIAS_COLOR, showlegend=show_leg, legendgroup="bias",
            customdata=np.column_stack([s["bias"], s["RMSFE"]]),
            hovertemplate="<b>%{x}</b><br>Bias: %{customdata[0]:.3f} pp<br>"
                          "Bias²: %{y:.4f} pp²<br>"
                          "RMSFE: %{customdata[1]:.3f} pp<extra></extra>",
        ), row=1, col=col)
        fig.add_trace(go.Bar(
            x=month_order, y=s["variance"], name="Variance",
            marker_color=_VARIANCE_COLOR, showlegend=show_leg, legendgroup="var",
            customdata=np.column_stack([s["bias"], s["RMSFE"]]),
            hovertemplate="<b>%{x}</b><br>Variance: %{y:.4f} pp²<br>"
                          "RMSFE: %{customdata[1]:.3f} pp<extra></extra>",
        ), row=1, col=col)
    fig.update_layout(
        barmode="stack", height=440,
        title=f"Bias–variance decomposition of RMSFE² — {regime}",
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                    font=dict(size=11.5)),
        margin=dict(t=70, b=70, l=70, r=30),
    )
    fig.update_yaxes(title_text="Mean squared error (pp²)", row=1, col=1)
    for ann in fig.layout.annotations[:len(shown)]:
        ann.update(font=dict(size=12, color=C.INK))
    return fig


def revision_band(df: pd.DataFrame, log_y: bool = False) -> go.Figure:
    """DFM-EN within-quarter revision band: M1 vs M3 nowcasts and realised GDP.

    The shaded band spans the first (M1) and final (M3) nowcast of each
    quarter; its width is the information accrued inside the quarter. The
    solid line is the final M3 nowcast, markers are realised first-release GDP.
    """
    sub = df.sort_values("date")

    def _y(vals: pd.Series) -> np.ndarray:
        arr = vals.astype(float).to_numpy()
        return _symlog(arr) if log_y else arr

    fig = go.Figure()
    fig.add_vrect(x0=_COVID_TS_START, x1=_COVID_TS_END,
                  fillcolor=C.REGIME_COLORS["COVID"], opacity=0.10,
                  line_width=0, layer="below")
    band_color = "rgba(236,111,142,0.26)"
    fig.add_trace(go.Scatter(
        x=sub["date"], y=_y(sub["nowcast_M1"]), name="M1 nowcast",
        mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
        customdata=sub["nowcast_M1"].to_numpy(),
        hovertemplate="M1: %{customdata:.2f} pp<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=sub["date"], y=_y(sub["nowcast_M3"]),
        name="M1 → M3 revision band", mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        fill="tonexty", fillcolor=band_color,
        customdata=sub["nowcast_M3"].to_numpy(),
        hovertemplate="M3: %{customdata:.2f} pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub["date"], y=_y(sub["nowcast_M3"]), name="Final (M3) nowcast",
        mode="lines+markers", line=dict(color=C.model_color("DFM-EN"), width=2.2),
        marker=dict(size=4),
        customdata=sub["nowcast_M3"].to_numpy(),
        hovertemplate="<b>M3 nowcast</b><br>%{x|%Yq%q}: "
                      "%{customdata:.2f} pp<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sub["date"], y=_y(sub["actual"]), name="Realised GDP",
        mode="lines", line=dict(color=C.ACTUAL_COLOR, width=2.4),
        customdata=sub["actual"].to_numpy(),
        hovertemplate="<b>Realised</b><br>%{x|%Yq%q}: "
                      "%{customdata:.2f} pp<extra></extra>",
    ))
    fig.update_layout(
        height=460, hovermode="x unified",
        title="How the DFM-EN nowcast firms up within the quarter",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=70, r=30),
    )
    fig.update_yaxes(title_text="GDP growth (pp, symlog)" if log_y
                     else "GDP growth (pp)")
    if log_y:
        vals = pd.concat([sub["nowcast_M1"], sub["nowcast_M3"],
                          sub["actual"]]).dropna()
        tickvals, ticktext = _symlog_ticks(float(vals.min()), float(vals.max()))
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return fig


def mz_forest(mz: pd.DataFrame) -> go.Figure:
    """Mincer–Zarnowitz coefficient plot: α and β with 95% CIs per model.

    Two aligned panels (intercept and slope) with reference lines at the
    efficient values α = 0 and β = 1. Models whose joint efficiency test
    rejects at 5% are drawn with a dark ring. A forest layout avoids the
    label pile-up of a scatter when several models cluster near the ideal.
    """
    sub = mz.sort_values("beta", ascending=True).reset_index(drop=True)
    models = sub["model"].tolist()

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=["<b>Intercept α</b> (bias)",
                        "<b>Slope β</b> (calibration of magnitude)"],
    )
    for col, (val, se, ref) in enumerate(
            [("alpha", "se_alpha", 0.0), ("beta", "se_beta", 1.0)], start=1):
        fig.add_vline(x=ref, line=dict(color=C.SUBTLE, width=1.3, dash="dash"),
                      row=1, col=col)
        for i, r in sub.iterrows():
            rejected = float(r["p_joint_H0_a0_b1"]) < 0.05
            color = _dfm_table_color(r["model"])
            fig.add_trace(go.Scatter(
                x=[r[val] - 1.96 * r[se], r[val] + 1.96 * r[se]],
                y=[r["model"]] * 2, mode="lines",
                line=dict(color=color, width=2.5), opacity=0.45,
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[r[val]], y=[r["model"]], mode="markers",
                marker=dict(size=11, color=color,
                            line=dict(color=C.INK if rejected
                                      else "rgba(0,0,0,0)", width=2)),
                hovertemplate=(f"<b>{r['model']}</b><br>"
                               f"α = {r['alpha']:.2f} · β = {r['beta']:.2f}"
                               f"<br>joint p(α=0, β=1) = "
                               f"{r['p_joint_H0_a0_b1']:.3f}<extra></extra>"),
                showlegend=False,
            ), row=1, col=col)
    fig.update_layout(
        height=420,
        title="Mincer–Zarnowitz forecast efficiency — actual = α + β · nowcast"
              "<br><span style='font-size:12px;color:" + C.SUBTLE + "'>"
              "Whiskers are 95% CIs (HAC). Dashed lines mark the efficient "
              "values α = 0, β = 1;<br>a dark ring marks models where the "
              "joint test rejects efficiency at 5%.</span>",
        margin=dict(t=125, b=55, l=150, r=30),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=models, row=1, col=1)
    for ann in fig.layout.annotations[:2]:
        ann.update(font=dict(size=12, color=C.INK))
    return fig


_DFM_TABLE_EXTRA = {
    "AR(1) expanding": "AR1",
}
_DFM_TABLE_FALLBACK = {
    "Rolling-AR(1) 40q": "#5A78A0",
    "AR(1) + IC": "#4A6488",
}


def _dfm_table_color(name: str) -> str:
    alias = _DFM_TABLE_EXTRA.get(name)
    if alias:
        return C.model_color(alias)
    if name in C.MODELS:
        return C.model_color(name)
    return _DFM_TABLE_FALLBACK.get(name, C.SUBTLE)


def post_covid_bars(df: pd.DataFrame, regime_col: str) -> go.Figure:
    """Horizontal RMSFE bars for the post-COVID benchmark set (one regime)."""
    label = regime_col.replace("_rmsfe", "")
    sub = df[["model", regime_col]].dropna().sort_values(regime_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=sub[regime_col], y=sub["model"], orientation="h",
        marker=dict(color=[_dfm_table_color(m) for m in sub["model"]]),
        text=[f"{v:.2f}" for v in sub[regime_col]], textposition="outside",
        textfont=dict(size=11, color=C.SUBTLE), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>RMSFE: %{x:.3f} pp<extra></extra>",
    ))
    fig.update_layout(
        height=420, title=f"Benchmark horse-race — {label} (RMSFE, M3)",
        margin=dict(t=70, b=50, l=170, r=50),
    )
    fig.update_xaxes(title_text="RMSFE (pp)")
    return fig


_XGB_SENS_LABELS = {
    "seed_0": "seed = 0",
    "seed_1": "seed = 1",
    "seed_7": "seed = 7",
    "seed_42": "seed = 42 (headline)",
    "seed_123": "seed = 123",
    "hp_max_depth-1": "max_depth −1 (5)",
    "hp_max_depth+1": "max_depth +1 (7)",
    "hp_lr_half": "learning_rate ×0.5",
    "hp_lr_double": "learning_rate ×2",
    "hp_n_estimators-100": "n_estimators −100 (400)",
    "hp_n_estimators+100": "n_estimators +100 (600)",
}


def xgb_sensitivity_bars(
    df: pd.DataFrame,
    dfm_range: tuple[float, float],
    best_dfm: tuple[str, float],
    rolling_ar1: float,
) -> go.Figure:
    """Post-COVID RMSFE across XGB-Full seed and hyperparameter re-runs.

    Bars are sorted by RMSFE and coloured by perturbation type; the headline
    ``seed=42`` run gets a dark outline. A shaded band marks the range spanned
    by the DFM family's own post-COVID RMSFE and a dotted line marks the best
    single DFM variant, so it is immediately visible how many re-runs would
    flip the "XGB beats every DFM post-COVID" headline claim. A second dotted
    line marks Rolling-AR(1) 40q, the naive benchmark used in the DM test.
    """
    sub = df.copy()
    sub["label"] = sub["run"].map(lambda r: _XGB_SENS_LABELS.get(r, r))
    # Ascending RMSFE in the data; autorange="reversed" puts the best at the top.
    sub = sub.sort_values("rmsfe_post", ascending=True).reset_index(drop=True)

    xgb_color = C.model_color("XGB-Full")
    hp_color = "#9ED9C8"
    ar_color = "#5A78A0"
    dfm_band = "#C5D0DE"
    bar_colors = [xgb_color if c == "Seed" else hp_color for c in sub["category"]]
    line_colors = ["#0D6B54" if bl else "rgba(0,0,0,0)" for bl in sub["is_baseline"]]
    line_widths = [2.8 if bl else 0 for bl in sub["is_baseline"]]
    text_vals = [
        f"<b>{v:.3f}</b>" if bl else f"{v:.3f}"
        for v, bl in zip(sub["rmsfe_post"], sub["is_baseline"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["rmsfe_post"], y=sub["label"], orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=line_colors, width=line_widths),
            cornerradius=3,
        ),
        text=text_vals, textposition="outside",
        textfont=dict(size=11, color=C.INK), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Post-COVID RMSFE: %{x:.3f} pp<extra></extra>",
        showlegend=False,
    ))

    # Legend swatches — kept out of the data domain so they don't crowd the title.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=11, color=xgb_color, symbol="square"),
        name="Seed", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=11, color=hp_color, symbol="square"),
        name="Hyperparameter", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="lines",
        line=dict(color=ar_color, width=2, dash="dot"),
        name=f"Rolling-AR(1) ({rolling_ar1:.2f})", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=11, color=dfm_band, symbol="square",
                    line=dict(width=1, color=C.SUBTLE)),
        name=f"DFM band ({dfm_range[0]:.2f}–{dfm_range[1]:.2f})", showlegend=True,
    ))

    lo, hi = dfm_range
    fig.add_vrect(
        x0=lo, x1=hi, fillcolor=dfm_band, opacity=0.28, line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=best_dfm[1],
        line=dict(color=C.SUBTLE, width=1.2, dash="dash"),
        layer="below",
    )
    fig.add_vline(
        x=rolling_ar1,
        line=dict(color=ar_color, width=1.6, dash="dot"),
        layer="below",
    )

    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=12, color=C.INK),
        ticksuffix="  ",
    )
    fig.update_xaxes(
        title_text="Post-COVID RMSFE (pp)",
        title_font=dict(size=12, color=C.SUBTLE),
        tickfont=dict(size=11, color=C.SUBTLE),
        range=[0, float(sub["rmsfe_post"].max()) * 1.18],
        showgrid=True, gridcolor=C.GRID, gridwidth=1,
        zeroline=False,
    )
    fig.update_layout(
        height=480, bargap=0.32,
        title=dict(text=""),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.08,   # ↑ raise this for more gap above the bars
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11.5, color=C.INK),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(t=58, b=52, l=178, r=64),  # ↑ raise t if the legend gets clipped
        plot_bgcolor=C.PAPER,
        paper_bgcolor=C.PAPER,
    )
    return fig


def dm_heatmap(
    df: pd.DataFrame,
    *,
    title: str = "Diebold–Mariano equal-accuracy test (p-values)",
    subtitle: str = "",
) -> go.Figure:
    """Diebold–Mariano p-value matrix (lower = more significant difference)."""
    z = df.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(df.columns), y=list(df.index),
        colorscale=[[0.0, "#1F6F54"], [0.05, "#5BAE8A"], [0.05, "#D8E6F0"],
                    [0.1, "#AFC6DE"], [1.0, "#E8EDF4"]],
        zmin=0, zmax=1, colorbar=dict(title="DM p-value", thickness=14),
        hovertemplate="%{y} vs %{x}<br>p = %{z:.3f}<extra></extra>",
        hoverongaps=False,
    ))
    full_title = title
    if subtitle:
        full_title += (
            f"<br><span style='font-size:12px;color:{C.SUBTLE}'>{subtitle}</span>"
        )
    fig.update_layout(
        height=520, title=full_title,
        margin=dict(t=90 if subtitle else 70, b=120, l=160, r=30),
    )
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(autorange="reversed")
    return fig


def contributions_stacked(df: pd.DataFrame, start: str, end: str,
                          mode: str = "pp",
                          origin_hovers: dict[tuple[pd.Timestamp, str], str]
                          | None = None,
                          model_label: str = "DFM-EN") -> go.Figure:
    """Stacked category contributions to the nowcast over time.

    ``model_label`` selects the model named in the chart titles (e.g.
    ``"DFM-EN"`` or ``"DFM-TVP"``); the plot design is otherwise identical.

    ``mode="pp"`` shows signed percentage-point contributions with the nowcast
    and realised-GDP lines overlaid. ``mode="pct"`` rescales each origin so the
    positive and negative stacks each sum to ±100 %, exposing the *proportional*
    mix of categories even when one block dominates in absolute terms.
    """
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    fig = go.Figure()
    piv = sub.pivot_table(index="date", columns="category", values="contrib_pp",
                          aggfunc="mean").reindex(columns=C.CATEGORY_ORDER)
    active = [c for c in C.CATEGORY_ORDER
              if c in piv.columns and piv[c].abs().sum() != 0]
    hovers = origin_hovers or {}

    def _hover_extra(dt, cat: str) -> str:
        text = hovers.get((pd.Timestamp(dt), cat), "")
        return f"<br>{text}" if text else ""

    if mode == "pct":
        pos = piv[active].clip(lower=0).sum(axis=1).replace(0, np.nan)
        neg = piv[active].clip(upper=0).sum(axis=1).abs().replace(0, np.nan)
        for cat in active:
            col = piv[cat]
            scaled = np.where(col >= 0, col.div(pos) * 100.0,
                              col.div(neg) * 100.0)
            scaled = np.round(scaled, 2)
            custom = []
            for dt, val in zip(piv.index, scaled):
                extra = _hover_extra(dt, cat)
                pct = f"{val:+.2f}" if np.isfinite(val) else ""
                custom.append([extra, pct])
            fig.add_trace(go.Bar(
                x=piv.index, y=scaled, name=_cat_display(cat),
                marker_color=C.CATEGORY_COLORS[cat],
                customdata=custom,
                hovertemplate=("<b>" + _cat_display(cat) + "</b><br>"
                               "%{x|%b %Y}: %{customdata[1]}%<br>"
                               "%{customdata[0]}<extra></extra>"),
            ))
        title = (f"{model_label} nowcast decomposition — relative category "
                 "share (% of each origin)")
        ytitle = "Share of nowcast magnitude (%)"
    else:
        for cat in active:
            extras = [_hover_extra(dt, cat) for dt in piv.index]
            fig.add_trace(go.Bar(
                x=piv.index, y=piv[cat], name=_cat_display(cat),
                marker_color=C.CATEGORY_COLORS[cat],
                customdata=extras,
                hovertemplate=("<b>" + _cat_display(cat) + "</b><br>"
                               "%{x|%b %Y}: %{y:+.3f} pp"
                               "%{customdata}<extra></extra>"),
            ))
        nc = sub.groupby("date")["nowcast"].first()
        ac = sub.groupby("date")["actual"].first()
        fig.add_trace(go.Scatter(x=nc.index, y=nc.values, name="Nowcast",
                                 mode="lines+markers",
                                 line=dict(color=C.INK, width=2)))
        fig.add_trace(go.Scatter(x=ac.index, y=ac.values, name="Realised GDP",
                                 mode="lines", line=dict(color=C.ACCENT, width=2,
                                                         dash="dot")))
        title = f"{model_label} nowcast decomposition — category contributions (pp)"
        ytitle = "Contribution to nowcast (pp)"

    fig.update_layout(
        barmode="relative", height=480, title=title,
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(t=70, b=100, l=70, r=30),
    )
    fig.update_yaxes(title_text=ytitle)
    return fig


_COVID_START = pd.Timestamp("2020-01-01")
_COVID_END = pd.Timestamp("2021-12-31")
_STAGNATION_ONSET = pd.Timestamp("2022-01-01")


def _stacked_share_areas(
    fig: go.Figure,
    wide: pd.DataFrame,
    *,
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    """Add one stacked-area panel (share of |loading|, sums to 1)."""
    cats = [c for c in C.FACTOR_LOADING_CATEGORIES if c in wide.columns]
    y_base = pd.Series(0.0, index=wide.index)
    for i, cat in enumerate(cats):
        y_vals = wide[cat].fillna(0.0)
        y_top = y_base + y_vals
        color = C.CATEGORY_COLORS.get(cat, "#9CA3AF")
        label = _cat_display(cat)
        fig.add_trace(
            go.Scatter(
                x=wide.index,
                y=y_top,
                customdata=y_vals,
                name=label,
                mode="lines",
                line=dict(width=0, color=color),
                fill="tozeroy" if i == 0 else "tonexty",
                fillcolor=color,
                hoveron="fills",
                showlegend=show_legend,
                legendgroup=cat,
                hovertemplate=(
                    f"<b>{label}</b><br>%{{x|%Y Q%q}}: %{{customdata:.1%}}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
        y_base = y_top


def factor_loading_category_stacks(cat_df: pd.DataFrame) -> go.Figure:
    """Stage 1: category shares of |loading| per factor (panels a left/right)."""
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=[
            (f"<span style='color:{C.FACTOR_COLORS[0]}'>"
             f"{C.FACTOR_SHORT[0]}</span><br>"
             "<span style='font-size:11px;color:#64748B'>"
             "Which data categories drive this factor?</span>"),
            (f"<span style='color:{C.FACTOR_COLORS[1]}'>"
             f"{C.FACTOR_SHORT[1]}</span><br>"
             "<span style='font-size:11px;color:#64748B'>"
             "Which data categories drive this factor?</span>"),
        ],
    )
    for f in (1, 2):
        sub = cat_df[cat_df["factor"] == f]
        wide = (
            sub.pivot_table(index="date", columns="category", values="share",
                            aggfunc="first")
            .reindex(columns=C.FACTOR_LOADING_CATEGORIES, fill_value=0.0)
            .sort_index()
        )
        _stacked_share_areas(fig, wide, row=1, col=f, show_legend=(f == 2))
        fig.add_vrect(
            x0=_COVID_START, x1=_COVID_END, row=1, col=f,
            fillcolor="rgba(226,137,155,0.12)", line_width=0, layer="below",
        )
        fig.add_vline(
            x=_STAGNATION_ONSET, row=1, col=f,
            line=dict(color="#6B7280", width=1, dash="dash"),
        )
    fig.update_yaxes(title_text="Share of |loading|", range=[0, 1], row=1, col=1)
    fig.update_layout(
        height=420,
        title=dict(
            text=("Stage 1 — What do the two factors load on? "
                  "(category shares, M3 origins)"),
            font=dict(size=14),
        ),
        hovermode="closest",
        legend=dict(
            orientation="v", y=0.5, yanchor="middle",
            x=1.02, xanchor="left", font=dict(size=11),
            title=dict(text="Category", font=dict(size=11.5, color=C.SUBTLE)),
        ),
        margin=dict(t=90, b=55, l=70, r=130),
    )
    fig.update_xaxes(title_text="Quarter (M3 forecast origin)", row=1, col=1)
    return fig


def tvp_bridge_loadings(tvp_df: pd.DataFrame) -> go.Figure:
    """Stage 2: time-varying factor → GDP bridge coefficients (panel b)."""
    fig = go.Figure()
    fig.add_vrect(
        x0=_COVID_START, x1=_COVID_END,
        fillcolor="rgba(226,137,155,0.15)", line_width=0, layer="below",
        annotation_text="COVID (down-weighted)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color=C.SUBTLE,
    )
    fig.add_vline(
        x=_STAGNATION_ONSET, line=dict(color="#6B7280", width=1, dash="dash"),
    )
    fig.add_hline(y=0.0, line=dict(color="#C5CED8", width=0.8))
    for j, col in enumerate(["tvp_loading_1", "tvp_loading_2"]):
        if col not in tvp_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=tvp_df["date"], y=tvp_df[col],
            name=f"λ{j + 1}  {C.FACTOR_SHORT[j]}",
            mode="lines+markers",
            line=dict(color=C.FACTOR_COLORS[j], width=2.1),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>λ{j + 1} {C.FACTOR_SHORT[j]}</b><br>"
                "%{x|%Y Q%q}: %{y:.3f} pp per unit of factor<extra></extra>"
            ),
        ))
    fig.add_annotation(
        x=_STAGNATION_ONSET, y=1.0, yref="paper",
        text="2022 stagnation onset",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=10, color="#6B7280"),
    )
    fig.update_layout(
        height=360,
        title=dict(
            text=("Stage 2 — How strongly does each factor transmit to GDP? "
                  "(DFM-TVP bridge)"),
            font=dict(size=14),
        ),
        yaxis_title="Loading on GDP (pp per unit of factor)",
        xaxis_title="Quarter (M3 forecast origin)",
        hovermode="x unified",
        legend=dict(
            orientation="h", y=1.14, yanchor="top",
            x=0.5, xanchor="center", bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(t=115, b=55, l=70, r=30),
    )
    return fig


# =========================================================================== #
# Part I — Indicator selection
# =========================================================================== #
def structural_shift_area(
    share: pd.DataFrame,
    title: str = "Structural shift in selected indicators — category mass share",
    height: int = 470,
) -> go.Figure:
    """Stacked area of category mass share across forecast origins."""
    fig = go.Figure()
    cats = [c for c in C.CATEGORY_ORDER if c in share.columns]
    y_base = pd.Series(0.0, index=share.index)
    for i, cat in enumerate(cats):
        y_vals = share[cat]
        y_top = y_base + y_vals
        color = C.CATEGORY_COLORS[cat]
        label = _cat_display(cat)
        fig.add_trace(go.Scatter(
            x=share.index,
            y=y_top,
            customdata=y_vals,
            name=label,
            mode="lines",
            line=dict(width=0, color=color),
            fill="tozeroy" if i == 0 else "tonexty",
            fillcolor=color,
            hoveron="fills",
            hovertemplate=(
                f"<b>{label}</b><br>%{{x|%b %Y}}: %{{customdata:.1%}}<extra></extra>"
            ),
        ))
        y_base = y_top
    fig.add_vrect(x0=pd.Timestamp("2020-01-01"), x1=pd.Timestamp("2021-12-31"),
                  line_width=0, fillcolor="rgba(0,0,0,0.05)", layer="above",
                  annotation_text="COVID", annotation_position="top left",
                  annotation_font_size=11)
    fig.update_layout(
        height=height,
        title=title,
        hovermode="closest",
        legend=dict(orientation="v", y=1.0, yanchor="top", x=1.015,
                    xanchor="left", font=dict(size=11),
                    title=dict(text="Category", font=dict(size=11.5,
                                                          color=C.SUBTLE))),
        margin=dict(t=70, b=55, l=70, r=150),
    )
    fig.update_yaxes(title_text="Share of selected mass", tickformat=".0%",
                     range=[0, 1])
    fig.update_xaxes(title_text="Forecast origin")
    return fig


_OVER_COLOR = "#A8516E"   # over-weighted vs availability (warm rose)
_UNDER_COLOR = "#2D6CB3"  # under-weighted vs availability (brand blue)

_SEGMENT_COLORS = {
    "Soft (surveys)": "#E2899B",
    "Hard (real activity)": "#2D6CB3",
    "Other": "#C9C2B4",
}


def deviation_bars(share: pd.DataFrame, uni: pd.Series, title: str,
                   height: int = 470) -> go.Figure:
    """Small-multiple diverging bars: selected share − universe share (pp).

    One panel per method. Bars to the right (rose) mark categories a method
    over-weights relative to how many such series exist in the panel; bars to
    the left (blue) mark under-weighted categories. This carries the same
    message as a lift ratio but in directly readable percentage points.
    """
    _SHORT = {
        "EN (raw)": "EN", "EN (smoothed)": "EN sm.",
        "Block-balanced (k=20)": "Block-bal.", "PLS": "PLS",
        "XGBoost (SHAP)": "XGBoost",
        "ifoCAST (fixed)": "ifoCAST", "EN-only": "EN-only",
        "ifoCAST": "ifoCAST",
    }
    methods = list(share.columns)
    cats = [c for c in C.CATEGORY_ORDER if c in share.index]
    dev = share.loc[cats].sub(uni.reindex(cats), axis=0) * 100.0

    fig = make_subplots(
        rows=1, cols=len(methods), shared_yaxes=True, horizontal_spacing=0.025,
        subplot_titles=[f"<b>{_SHORT.get(m, m)}</b>" for m in methods],
    )
    span = float(np.nanmax(np.abs(dev.to_numpy()))) if dev.size else 10.0
    span = max(span * 1.15, 5.0)
    for col, m in enumerate(methods, start=1):
        vals = dev[m].to_numpy(dtype=float)
        fig.add_trace(go.Bar(
            x=vals, y=[_cat_display(c) for c in cats], orientation="h",
            marker_color=[_OVER_COLOR if v >= 0 else _UNDER_COLOR for v in vals],
            showlegend=False,
            customdata=np.column_stack([
                share[m].reindex(cats).to_numpy() * 100,
                uni.reindex(cats).to_numpy() * 100,
            ]),
            hovertemplate=("<b>%{y}</b> · " + m +
                           "<br>selected: %{customdata[0]:.1f}% · "
                           "available: %{customdata[1]:.1f}%"
                           "<br>deviation: %{x:+.1f} pp<extra></extra>"),
        ), row=1, col=col)
        fig.add_vline(x=0, line=dict(color=C.SUBTLE, width=1), row=1, col=col)
        fig.update_xaxes(range=[-span, span], row=1, col=col,
                         tickfont=dict(size=10), ticksuffix="")
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_layout(
        height=height, bargap=0.3,
        title=dict(text=title + "<br><span style='font-size:12px;color:" +
                   C.SUBTLE + "'>Selected share − availability share, "
                   "percentage points. Rose = over-weighted, "
                   "blue = under-weighted.</span>"),
        margin=dict(t=110, b=50, l=110, r=20),
    )
    for ann in fig.layout.annotations[:len(methods)]:
        ann.update(font=dict(size=11, color=C.INK))
    return fig


def soft_hard_lines(df: pd.DataFrame, uni: pd.Series, title: str,
                    height: int = 440) -> go.Figure:
    """Soft vs hard share of selected mass through time, with base-rate dashes.

    Collapsing eleven categories to the economically meaningful triad makes the
    rotation story legible: when the solid line sits above its dashed
    availability line, the method genuinely over-weights that block.
    """
    fig = go.Figure()
    fig.add_vrect(x0=pd.Timestamp("2020-01-01"), x1=pd.Timestamp("2021-12-31"),
                  line_width=0, fillcolor="rgba(0,0,0,0.05)", layer="below",
                  annotation_text="COVID", annotation_position="top left",
                  annotation_font_size=11)
    for seg in ["Soft (surveys)", "Hard (real activity)"]:
        if seg not in df.columns:
            continue
        color = _SEGMENT_COLORS[seg]
        fig.add_trace(go.Scatter(
            x=df.index, y=df[seg] * 100, name=seg, mode="lines",
            line=dict(color=color, width=2.4),
            hovertemplate="<b>" + seg + "</b><br>%{x|%b %Y}: "
                          "%{y:.1f}% of selected mass<extra></extra>",
        ))
        base = float(uni.get(seg, np.nan)) * 100
        if np.isfinite(base):
            fig.add_hline(y=base, line=dict(color=color, width=1.3, dash="dot"),
                          opacity=0.75)
            fig.add_annotation(
                x=1.0, xref="paper", y=base, yanchor="middle", xanchor="left",
                text=f"{base:.0f}% available", showarrow=False,
                font=dict(size=10.5, color=color),
            )
    fig.update_layout(
        height=height, title=title,
        legend=dict(orientation="h", y=1.06, yanchor="bottom", x=0.0),
        margin=dict(t=92, b=55, l=70, r=110),
    )
    fig.update_yaxes(title_text="Share of selected mass (%)",
                     rangemode="tozero", ticksuffix="%")
    fig.update_xaxes(title_text="Forecast origin")
    return fig


def regime_rotation_bars(long: pd.DataFrame, uni: pd.Series,
                         height: int = 430) -> go.Figure:
    """Hard-data share of selected mass by regime, grouped per method.

    The single most decision-relevant regime statistic: does a method rotate
    toward hard real-activity data when the cycle turns? The dashed line marks
    the hard-data availability share, so bars above it indicate genuine
    over-weighting.
    """
    seg = "Hard (real activity)"
    sub = long[long["segment"] == seg]
    methods = [m for m in sub["method"].unique()]
    regimes = [r for r in C.REGIMES if r in set(sub["regime"])]

    fig = go.Figure()
    for r in regimes:
        vals = [float(sub[(sub["method"] == m) & (sub["regime"] == r)]["share"]
                      .sum()) * 100 for m in methods]
        fig.add_trace(go.Bar(
            x=methods, y=vals, name=r, marker_color=C.REGIME_COLORS[r],
            text=[f"{v:.0f}%" for v in vals], textposition="outside",
            textfont=dict(size=10, color=C.SUBTLE), cliponaxis=False,
            hovertemplate=("<b>%{x}</b> · " + r +
                           "<br>hard-data share: %{y:.1f}%<extra></extra>"),
        ))
    base = float(uni.get(seg, np.nan)) * 100
    if np.isfinite(base):
        fig.add_hline(y=base, line=dict(color=C.INK, width=1.3, dash="dot"),
                      annotation_text=f"{base:.0f}% of panel is hard data",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color=C.SUBTLE))
    fig.update_layout(
        barmode="group", height=height + 40,
        title="Reliance on hard real-activity data by regime<br>"
              "<span style='font-size:12px;color:" + C.SUBTLE + "'>"
              "Share of each method's selected mass in Orders, Turnover, "
              "Production, Construction and Trade</span>",
        legend=dict(orientation="h", y=-0.16, yanchor="top", x=0.5,
                    xanchor="center"),
        margin=dict(t=90, b=95, l=70, r=30),
    )
    fig.update_yaxes(title_text="Hard-data share of selected mass (%)",
                     ticksuffix="%", rangemode="tozero")
    return fig


def agreement_heatmap(df: pd.DataFrame, title: str, label: str,
                      height: int = 520) -> go.Figure:
    """Generic square method-by-method agreement heatmap (Spearman ρ / Jaccard)."""
    fig = go.Figure(go.Heatmap(
        z=df.values, x=list(df.columns), y=list(df.index),
        colorscale=[[0, "#F4F8FC"], [0.5, "#8FB8DE"], [1, "#1F4E79"]],
        zmin=0, zmax=1, colorbar=dict(title=label, thickness=14),
        text=[[f"{v:.2f}" for v in row] for row in df.values],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="%{y} vs %{x}<br>" + label + " = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=height, title=title,
                      margin=dict(t=70, b=130, l=160, r=30))
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(autorange="reversed")
    return fig


def universe_bar(uni: pd.Series) -> go.Figure:
    """Horizontal bar of the predictor universe's category composition."""
    s = uni.reindex(C.CATEGORY_ORDER).fillna(0.0)
    fig = go.Figure(go.Bar(
        x=(s.values * 100), y=[_cat_display(c) for c in s.index],
        orientation="h", marker_color=[C.CATEGORY_COLORS[c] for c in s.index],
        text=[f"{v*100:.1f}%" for v in s.values], textposition="outside",
        textfont=dict(size=10, color=C.SUBTLE), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of the 585-series universe"
                      "<extra></extra>",
    ))
    fig.update_layout(
        height=300, bargap=0.3,
        title="Predictor universe — how many series exist per category",
        margin=dict(t=60, b=40, l=120, r=50),
    )
    fig.update_xaxes(title_text="Share of universe (%)")
    fig.update_yaxes(autorange="reversed")
    return fig


def regime_share_bars(long: pd.DataFrame, methods: list[str]) -> go.Figure:
    """Horizontal grouped bars: category mass share (%) by method, one panel per regime.

    Each panel is one economic regime; bars within a panel compare methods
    side-by-side so the supervisor reads shift across regimes horizontally.
    """
    regimes = list(C.REGIMES)
    method_colors = {
        "EN":             C.ACCENT,
        "Block-balanced": "#8F3D58",
        "PLS":            "#7FB7A6",
        "XGBoost (SHAP)": "#3E9B73",
        "ifoCAST (fixed)": "#C9617F",
    }
    cats     = [c for c in C.CATEGORY_ORDER]
    cat_lbl  = [_cat_display(c) for c in cats]

    fig = make_subplots(
        rows=1, cols=len(regimes), shared_yaxes=True,
        horizontal_spacing=0.055,
        subplot_titles=[
            f"<b>{r}</b><br><span style='font-size:11px;color:{C.SUBTLE}'>"
            f"{C.REGIMES[r][0]}–{C.REGIMES[r][1]}</span>" for r in regimes
        ],
    )
    for col, regime in enumerate(regimes, start=1):
        for method in methods:
            sub = long[(long["regime"] == regime) & (long["method"] == method)]
            sub = sub.set_index("category").reindex(cats)["share"].fillna(0)
            fig.add_trace(go.Bar(
                x=sub.values * 100,   # convert to percent
                y=cat_lbl,
                orientation="h",
                name=method,
                marker=dict(
                    color=method_colors.get(method, C.SUBTLE),
                    line=dict(color=C.PAPER, width=0.6),
                ),
                showlegend=(col == 1),
                legendgroup=method,
                hovertemplate=(
                    "<b>" + method + "</b> · " + regime +
                    "<br>%{y}: %{x:.1f}% of selected set<extra></extra>"
                ),
            ), row=1, col=col)

    # Uniform x-range across the three regime panels so they are directly
    # comparable, with a little headroom above the tallest bar.
    vals = long[long["method"].isin(methods)]["share"].to_numpy(dtype=float)
    xmax = float(np.nanmax(vals)) * 100 if vals.size else 50.0
    xmax = max(xmax * 1.08, 10.0)

    fig.update_layout(
        barmode="group", bargap=0.34, bargroupgap=0.06, height=560,
        title=dict(
            text="How does selection composition shift across regimes?<br>"
                 "<span style='font-size:13px;color:" + C.SUBTLE + "'>"
                 "Share of selected indicators in each category (% of total "
                 "selected, before base-rate adjustment)</span>",
            y=0.97, yanchor="top",
        ),
        legend=dict(
            orientation="h", y=-0.16, x=0.5, xanchor="center",
            font=dict(size=11.5), itemsizing="constant",
            title=dict(text=""),
        ),
        margin=dict(t=132, b=88, l=140, r=44),
    )
    fig.update_xaxes(
        title_text="% of selected set", ticksuffix="%", range=[0, xmax],
        dtick=20, showgrid=True, gridcolor=C.GRID, gridwidth=1,
        zeroline=True, zerolinecolor=C.GRID, zerolinewidth=1,
        title_standoff=12, ticks="outside", tickcolor=C.GRID,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    for ann in fig.layout.annotations[:len(regimes)]:
        ann.update(font=dict(size=12.5, color=C.INK))
    return fig


def composition_bars(comp: pd.DataFrame) -> go.Figure:
    """Grouped horizontal bars: category share (%) per DFM candidate input set.

    Each bar group is one category; bars within the group compare input sets.
    Values are the share of that category inside the set (columns sum to 100%).
    """
    input_set_colors = {
        "EN only":               C.ACCENT,
        "Block-balanced (k=20)": "#8F3D58",
        "PLS":                   "#7FB7A6",
        "ifoCAST (fixed)":       "#C9617F",
    }
    cats    = [c for c in C.CATEGORY_ORDER if c in comp.index]
    cat_lbl = [_cat_display(c) for c in cats]

    fig = go.Figure()
    for col in comp.columns:
        vals = [float(comp.loc[c, col]) if c in comp.index else 0.0 for c in cats]
        fig.add_trace(go.Bar(
            x=vals, y=cat_lbl, orientation="h",
            name=col,
            marker_color=input_set_colors.get(col, C.SUBTLE),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside", textfont=dict(size=10, color=C.SUBTLE),
            cliponaxis=False,
            hovertemplate=(
                "<b>" + col + "</b><br>"
                "%{y}: %{x:.1f}% of selected set<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group", height=500,
        title=(
            "Economic composition of each DFM candidate input set<br>"
            "<span style='font-size:13px;color:" + C.SUBTLE + "'>"
            "% of indicators in the set belonging to each category</span>"
        ),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"),
        margin=dict(t=100, b=80, l=130, r=50),
    )
    fig.update_xaxes(title_text="Share of input set (%)", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def publag_composition(mat: pd.DataFrame) -> go.Figure:
    """Stacked bars: category composition of each publication-lag bucket.

    Legend sits to the right of the plotting area so it can never collide
    with the x-axis title, and the lag buckets carry their own annotation of
    the total series count.
    """
    fig = go.Figure()
    lag_labels = [C.PUBLAG_LABELS.get(i, f"lag {i}") for i in mat.index]
    for cat in C.CATEGORY_ORDER:
        if cat not in mat.columns or mat[cat].sum() == 0:
            continue
        fig.add_trace(go.Bar(
            y=lag_labels, x=mat[cat], orientation="h", name=_cat_display(cat),
            marker_color=C.CATEGORY_COLORS[cat],
            hovertemplate="<b>" + _cat_display(cat) + "</b><br>"
                          "%{y}<br>%{x} series<extra></extra>",
        ))
    totals = mat.sum(axis=1)
    for lbl, tot in zip(lag_labels, totals):
        fig.add_annotation(
            x=float(tot), y=lbl, xanchor="left", xshift=6,
            text=f"<b>{int(tot)}</b>", showarrow=False,
            font=dict(size=11, color=C.SUBTLE),
        )
    fig.update_layout(
        barmode="stack", height=380,
        title="The ragged edge — what is actually on the desk at the "
              "forecast origin",
        legend=dict(orientation="v", y=1.0, yanchor="top", x=1.01,
                    xanchor="left", font=dict(size=11),
                    title=dict(text="Category", font=dict(size=11.5,
                                                          color=C.SUBTLE))),
        margin=dict(t=70, b=60, l=185, r=160),
    )
    fig.update_xaxes(title_text="Number of indicators available",
                     title_standoff=10)
    fig.update_yaxes(autorange="reversed")
    return fig
