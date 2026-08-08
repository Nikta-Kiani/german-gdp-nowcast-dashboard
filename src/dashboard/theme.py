"""Visual theme: a shared Plotly template and injected Streamlit CSS.

Centralising the look here keeps every figure and page on the same tonal,
publication-grade language (ifo / Bundesbank flavour).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import config as C

THEME_NAME = "thesis"


def register_plotly_template() -> None:
    """Register and activate the shared Plotly template (idempotent)."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=C.FONT_FAMILY, size=13, color=C.INK),
        paper_bgcolor=C.PAPER,
        plot_bgcolor=C.PAPER,
        title=dict(font=dict(size=18, color=C.INK), x=0.0, xanchor="left",
                   pad=dict(b=12)),
        margin=dict(t=70, b=60, l=70, r=30),
        colorway=[C.CATEGORY_COLORS[c] for c in C.CATEGORY_ORDER],
        xaxis=dict(showgrid=False, zeroline=False, linecolor=C.GRID,
                   ticks="outside", tickcolor=C.GRID, tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor=C.GRID, gridwidth=1, zeroline=False,
                   tickfont=dict(size=12)),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=C.GRID,
                    borderwidth=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", bordercolor=C.GRID,
                        font=dict(family=C.FONT_FAMILY, size=12, color=C.INK)),
        hovermode="closest",
    )
    pio.templates[THEME_NAME] = tpl
    pio.templates.default = f"plotly_white+{THEME_NAME}"


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --shadow-sm: 0 1px 2px rgba(20,35,59,0.05), 0 1px 3px rgba(20,35,59,0.06);
    --shadow-md: 0 4px 14px rgba(20,35,59,0.08), 0 2px 5px rgba(20,35,59,0.05);
    --shadow-lg: 0 14px 32px rgba(20,35,59,0.13), 0 3px 8px rgba(20,35,59,0.06);
    --ease: 0.18s cubic-bezier(0.4, 0, 0.2, 1);
}}

html, body, [class*="css"] {{
    font-family: {C.FONT_FAMILY};
    color: {C.INK};
}}

/* Refined, low-contrast scrollbars */
* {{ scrollbar-width: thin; scrollbar-color: #C7D2E0 transparent; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{
    background: #C7D2E0; border-radius: 999px; border: 2px solid {C.PAPER};
}}
::-webkit-scrollbar-thumb:hover {{ background: #AEBCCF; }}
.main .block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}}
h1, h2, h3, h4 {{ color: {C.INK}; font-weight: 700; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.0rem; }}
h2 {{ font-size: 1.45rem; margin-top: 0.4rem; }}
h3 {{ font-size: 1.15rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {C.PANEL};
    border-right: 1px solid {C.GRID};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

/* Sidebar navigation rendered as a pill list */
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 0.2rem; }}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    display: flex; align-items: center; width: 100%;
    padding: 0.52rem 0.7rem; border-radius: 10px; margin: 0;
    cursor: pointer;
    transition: background var(--ease), box-shadow var(--ease);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
    display: none;  /* hide the default radio dot for a cleaner nav */
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: rgba(45,108,179,0.07);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 0.9rem; font-weight: 600; color: {C.SUBTLE};
    transition: color var(--ease);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
    background: rgba(45,108,179,0.11);
    box-shadow: inset 3px 0 0 {C.ACCENT};
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{
    color: {C.ACCENT};
}}

/* Hero band */
.hero {{
    position: relative; overflow: hidden;
    background:
        radial-gradient(115% 130% at 90% -25%,
                        rgba(126,182,236,0.40) 0%, rgba(126,182,236,0) 48%),
        linear-gradient(135deg, #14233B 0%, #1F3A5F 55%, #2D6CB3 130%);
    color: #FFFFFF;
    padding: 1.95rem 2.1rem;
    border-radius: 18px;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255,255,255,0.08);
}}
.hero h1 {{ color: #FFFFFF; margin: 0 0 0.4rem 0; font-size: 1.85rem; }}
.hero p {{ color: #D7E3F4; margin: 0; font-size: 1.02rem; line-height: 1.5; }}
.hero .eyebrow {{
    text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.72rem;
    color: #9FC0E8; font-weight: 600; margin-bottom: 0.5rem;
}}

/* Metric / stat cards */
.stat-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.85rem; margin: 0.4rem 0 1.1rem 0;
}}
.stat-card {{
    position: relative; overflow: hidden;
    background: {C.PAPER}; border: 1px solid {C.GRID};
    border-radius: 14px; padding: 1rem 1.1rem 0.95rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--ease), box-shadow var(--ease),
                border-color var(--ease);
}}
.stat-card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {C.ACCENT} 0%, #5B93D1 100%);
    opacity: 0.9;
}}
.stat-card:hover {{
    transform: translateY(-3px); box-shadow: var(--shadow-md);
    border-color: #D4DEEC;
}}
.stat-card .v {{
    font-size: 1.55rem; font-weight: 700; color: {C.ACCENT};
    font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}}
.stat-card .l {{ font-size: 0.78rem; color: {C.SUBTLE}; text-transform: uppercase;
    letter-spacing: 0.06em; margin-top: 0.2rem; line-height: 1.3; }}

/* Note / insight callout */
.callout {{
    background: {C.PANEL}; border-left: 4px solid {C.ACCENT};
    border-radius: 0 10px 10px 0; padding: 0.85rem 1.1rem; margin: 0.6rem 0 1.1rem 0;
    font-size: 0.94rem; color: #33415C; line-height: 1.55;
}}
.callout b {{ color: {C.INK}; }}

/* Section eyebrow */
.eyebrow-dark {{
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.74rem;
    color: {C.ACCENT}; font-weight: 700; margin-bottom: 0.1rem;
}}
hr {{ border: none; border-top: 1px solid {C.GRID}; margin: 1.6rem 0; }}

div[data-testid="stMetricValue"] {{ color: {C.ACCENT}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.3rem; }}
.stTabs [data-baseweb="tab"] {{
    font-weight: 600; color: {C.SUBTLE}; padding: 0.5rem 0.9rem;
    border-radius: 8px 8px 0 0;
    transition: color var(--ease), background var(--ease);
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {C.INK}; background: rgba(45,108,179,0.05);
}}
.stTabs [aria-selected="true"] {{ color: {C.ACCENT}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {C.ACCENT}; }}
footer {{ visibility: hidden; }}

/* Selected pills in multiselect / selectbox -> brand blue (not red) */
span[data-baseweb="tag"] {{
    background-color: {C.ACCENT} !important;
    border-color: {C.ACCENT} !important;
}}
span[data-baseweb="tag"] span {{ color: #FFFFFF !important; }}
span[data-baseweb="tag"] svg {{ fill: #FFFFFF !important; }}

/* Radio / checkbox accents */
[data-baseweb="radio"] svg {{ color: {C.ACCENT} !important; }}

/* ---------- Pipeline flow diagrams ---------- */
.flow {{ display: flex; flex-direction: column; align-items: stretch;
         gap: 0; margin: 0.4rem 0 0.6rem 0; }}
.flow-box {{
    border: 1px solid {C.GRID}; border-radius: 12px; padding: 0.7rem 0.95rem;
    background: {C.PAPER}; box-shadow: var(--shadow-sm);
    text-align: center; line-height: 1.4;
    transition: transform var(--ease), box-shadow var(--ease);
}}
.flow-box:hover {{
    transform: translateY(-3px); box-shadow: var(--shadow-md); z-index: 1;
}}
.flow-box .ttl {{ font-weight: 700; color: {C.INK}; font-size: 0.95rem; }}
.flow-box .sub {{ color: {C.SUBTLE}; font-size: 0.8rem; margin-top: 0.15rem; }}
.flow-box.accent {{ border-left: 4px solid {C.ACCENT}; }}
.flow-arrow {{ text-align: center; color: {C.SUBTLE}; font-size: 1.1rem;
               line-height: 1; margin: 0.28rem 0; }}
.flow-row {{ display: flex; gap: 0.7rem; }}
.flow-row > .flow-box {{ flex: 1 1 0; }}
.flow-pipe {{
    display: flex; flex-wrap: wrap; align-items: stretch; justify-content: center;
    gap: 0.35rem 0.25rem; margin: 0.55rem 0 0.35rem 0;
}}
.flow-pipe .flow-box {{
    flex: 1 1 150px; min-width: 132px; max-width: 210px; text-align: left;
    padding: 0.65rem 0.75rem;
}}
.flow-pipe .flow-box .tag {{
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; color: {C.SUBTLE};
    margin-bottom: 0.22rem;
}}
.flow-pipe-arrow {{
    display: flex; align-items: center; justify-content: center;
    color: {C.SUBTLE}; font-size: 1.15rem; font-weight: 700;
    flex: 0 0 1.1rem; padding-top: 0.35rem;
}}
@media (max-width: 900px) {{
    .flow-pipe-arrow {{ flex: 0 0 100%; transform: rotate(90deg); padding: 0; }}
}}
.flow-tone-part1 {{ border-color: #C3AEE0; background: #F7F3FC; }}
.flow-tone-stage1 {{ border-color: #B39AD8; background: #FAF7FE; }}
.flow-tone-stage2 {{ border-color: #E9C08C; background: #FDF7EE; }}
.flow-tone-decomp {{ border-color: #F0A0B8; background: #FDE8EE; }}
.flow-tone-input {{ border-color: #9DBCE6; background: #F2F7FC; }}
.flow-tone-prep  {{ border-color: #9AC9BD; background: #F1F8F5; }}
.flow-tone-meth  {{ border-color: #C3AEE0; background: #F7F3FC; }}
.flow-tone-vec   {{ border-color: #C3AEE0; background: #FAF7FE; }}
.flow-tone-dfm   {{ border-color: #F0A0B8; background: #FDE8EE; }}
.flow-tone-out   {{ border-color: #93C7A8; background: #F1F9F4; }}
.flow-tone-sv    {{ border-color: #E9C08C; background: #FDF7EE; }}
.flow-tone-bench {{ border-color: #B6C0CF; background: #F6F8FB; }}
.flow-tone-mlp   {{ border-color: #B39AD8; background: #F7F3FC; }}
.flow-tone-mlp-in {{ border-color: #8B7EF0; background: #EDEAFE; }}
.flow-tone-xgb   {{ border-color: #5DD4B4; background: #E0F7F2; }}
.flow-tone-xgb-in {{ border-color: #2DB896; background: #E0F7F2; }}
.flow-section {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #33415C; margin: 0.85rem 0 0.45rem 0;
}}
.flow-ref {{
    color: {C.SUBTLE}; font-size: 0.82rem; line-height: 1.55;
    margin-top: 0.65rem; padding: 0.75rem 0.9rem;
    background: {C.PANEL}; border: 1px dashed {C.GRID}; border-radius: 10px;
}}
/* ML model workflow panels (shared by XGB + MLP tabs) */
.ml-wrap {{
    border: 1px solid {C.GRID}; border-radius: 16px; padding: 1rem 1.1rem 1.05rem;
    background: #FFFFFF; margin: 0.55rem 0 0.85rem 0;
}}
.ml-wrap .hdr {{
    font-weight: 700; font-size: 0.95rem; margin-bottom: 0.55rem;
}}
.ml-wrap .lead {{
    color: #33415C; font-size: 0.86rem; line-height: 1.55; margin-bottom: 0.75rem;
}}
.ml-wrap .model-note {{
    font-size: 0.84rem; line-height: 1.52; color: #33415C;
    padding: 0.55rem 0.7rem; border-radius: 10px; margin-bottom: 0.65rem;
}}
.ml-timeline {{
    display: grid; gap: 0.55rem; margin: 0.65rem 0 0.75rem 0;
}}
.ml-tl-row {{
    display: grid; grid-template-columns: 118px 1fr; gap: 0.55rem;
    align-items: stretch;
}}
.ml-tl-label {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; color: {C.SUBTLE}; padding-top: 0.42rem;
}}
.ml-tl-bar {{
    border-radius: 10px; padding: 0.55rem 0.7rem; border: 1px solid {C.GRID};
    font-size: 0.8rem; line-height: 1.45; color: {C.INK};
}}
.ml-tl-bar.train {{ border-color: #9AC9BD; background: #F1F8F5; }}
.ml-tl-bar.mem   {{ border-color: #B39AD8; background: #F7F3FC; }}
.ml-tl-bar.tgt   {{ border-color: #93C7A8; background: #F1F9F4; }}
.ml-tl-bar .rng {{ font-weight: 700; color: {C.INK}; }}
.ml-tl-bar .note {{ color: {C.SUBTLE}; font-size: 0.76rem; margin-top: 0.12rem; }}
.ml-compare {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.65rem; margin: 0.55rem 0;
}}
@media (max-width: 760px) {{
    .ml-compare {{ grid-template-columns: 1fr; }}
    .ml-tl-row {{ grid-template-columns: 1fr; }}
}}
.ml-cmp {{
    border-radius: 12px; padding: 0.75rem 0.85rem; border: 2px solid;
}}
.ml-cmp .ttl {{ font-weight: 700; font-size: 0.88rem; margin-bottom: 0.28rem; }}
.ml-cmp .body {{ font-size: 0.8rem; line-height: 1.5; color: #33415C; }}
.ml-cmp.train {{ border-color: #6DB89E; background: #F1F8F5; color: #1F5C45; }}
.ml-cmp.mem   {{ border-color: #8B7EF0; background: #EDEAFE; color: #4C3DB8; }}
.ml-glossary {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.55rem; margin: 0.65rem 0 0.25rem 0;
}}
.ml-gloss {{
    background: {C.PANEL}; border: 1px solid {C.GRID}; border-radius: 10px;
    padding: 0.62rem 0.72rem;
}}
.ml-gloss .term {{ font-weight: 700; font-size: 0.82rem; margin-bottom: 0.15rem; }}
.ml-gloss .def {{
    font-size: 0.78rem; line-height: 1.48; color: #33415C;
}}
.flow-box .step {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.35rem; height: 1.35rem; border-radius: 999px;
    color: #FFFFFF; font-size: 0.72rem; font-weight: 700;
    margin-right: 0.35rem; vertical-align: middle;
}}
.flow-box.left {{ text-align: left; }}
.flow-box .ex {{
    margin-top: 0.35rem; padding: 0.42rem 0.55rem; border-radius: 8px;
    background: rgba(255,255,255,0.72); border: 1px dashed {C.GRID};
    font-size: 0.76rem; color: #4A5568; line-height: 1.45; text-align: left;
}}
.flow-arrow.lbl {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: {C.SUBTLE};
}}
.flow-tag {{ display:inline-block; font-size:0.7rem; font-weight:700;
    letter-spacing:0.04em; text-transform:uppercase; color:{C.ACCENT}; }}

/* ---------- Model specification cards (Part II) ---------- */
.spec-intro {{
    color: {C.SUBTLE}; font-size: 0.94rem; line-height: 1.6;
    margin: 0.2rem 0 1.2rem 0;
}}
.spec-protocol {{
    background: linear-gradient(135deg, #F2F7FC 0%, #FFFFFF 70%);
    border: 1px solid #C8DAEF; border-left: 4px solid {C.ACCENT};
    border-radius: 14px; padding: 1.1rem 1.25rem 1rem 1.25rem;
    margin-bottom: 1.4rem;
}}
.spec-protocol .ttl {{
    font-weight: 700; color: {C.INK}; font-size: 0.98rem; margin-bottom: 0.75rem;
}}
.spec-protocol-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.65rem;
}}
.spec-protocol-item {{
    background: rgba(255,255,255,0.85); border: 1px solid {C.GRID};
    border-radius: 10px; padding: 0.65rem 0.75rem;
}}
.spec-protocol-item .k {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: {C.ACCENT}; margin-bottom: 0.2rem;
}}
.spec-protocol-item .v {{
    font-size: 0.84rem; color: {C.INK}; line-height: 1.45;
}}
.spec-family-tabs .stTabs [data-baseweb="tab-list"] {{
    gap: 0.35rem; flex-wrap: wrap;
}}
.spec-card {{
    background: {C.PAPER}; border: 1px solid {C.GRID}; border-radius: 14px;
    overflow: hidden; margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--ease), box-shadow var(--ease);
}}
.spec-card:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
.spec-card-hdr {{
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 0.75rem; padding: 0.85rem 1.05rem;
    border-bottom: 1px solid {C.GRID};
}}
.spec-card-hdr .left {{ flex: 1 1 auto; min-width: 0; }}
.spec-card-hdr .title {{
    font-weight: 700; font-size: 1rem; color: {C.INK}; line-height: 1.3;
}}
.spec-card-hdr .family {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.18rem;
}}
.spec-card-hdr .badge {{
    flex: 0 0 auto; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
    padding: 0.22rem 0.55rem; border-radius: 999px; white-space: nowrap;
}}
.spec-card-body {{ padding: 0.95rem 1.05rem 1.05rem 1.05rem; }}
.spec-card-desc {{
    color: #33415C; font-size: 0.9rem; line-height: 1.58; margin-bottom: 0.85rem;
}}
.spec-kv {{
    display: grid; grid-template-columns: minmax(120px, 34%) 1fr;
    gap: 0.35rem 0.9rem; font-size: 0.86rem; line-height: 1.45;
}}
.spec-kv .k {{ color: {C.SUBTLE}; padding-top: 0.12rem; }}
.spec-kv .v {{ color: {C.INK}; }}
.spec-kv .v code {{
    background: {C.PANEL}; padding: 0.08rem 0.35rem; border-radius: 4px;
    font-size: 0.82rem;
}}
.spec-variants {{
    margin-top: 0.95rem; padding-top: 0.85rem; border-top: 1px dashed {C.GRID};
}}
.spec-variants .lbl {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: {C.SUBTLE}; margin-bottom: 0.55rem;
}}
.spec-variant {{
    background: {C.PANEL}; border: 1px solid {C.GRID}; border-radius: 10px;
    padding: 0.65rem 0.8rem; margin-bottom: 0.45rem;
}}
.spec-variant:last-child {{ margin-bottom: 0; }}
.spec-variant .name {{
    font-weight: 700; font-size: 0.86rem; color: {C.INK}; margin-bottom: 0.18rem;
}}
.spec-variant .note {{ font-size: 0.84rem; color: #33415C; line-height: 1.5; }}
.spec-baseline-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.65rem;
}}
.spec-baseline {{
    background: {C.PANEL}; border: 1px solid {C.GRID}; border-radius: 10px;
    padding: 0.75rem 0.85rem;
}}
.spec-baseline .name {{
    font-weight: 700; font-size: 0.88rem; color: {C.INK}; margin-bottom: 0.22rem;
}}
.spec-baseline .note {{ font-size: 0.82rem; color: #33415C; line-height: 1.48; }}
.spec-formula-panel {{
    margin-top: 0.95rem; padding: 0.95rem 1.05rem 1.0rem 1.05rem;
    background: {C.PANEL}; border: 1px solid {C.GRID}; border-radius: 12px;
}}
.spec-formula-panel .ttl {{
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: {C.SUBTLE}; margin-bottom: 0.75rem;
}}
.spec-formula-block {{ margin-bottom: 0.72rem; }}
.spec-formula-block:last-child {{ margin-bottom: 0; }}
.spec-formula-block .lbl {{
    font-size: 0.78rem; font-weight: 600; color: {C.INK}; margin-bottom: 0.28rem;
}}
.spec-formula-block .eq {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.84rem; line-height: 1.65; color: #243044;
    background: {C.PAPER}; border: 1px solid {C.GRID}; border-radius: 8px;
    padding: 0.55rem 0.75rem; overflow-x: auto;
}}
.spec-formula-block .note {{
    font-size: 0.8rem; color: {C.SUBTLE}; line-height: 1.48; margin-top: 0.28rem;
}}

/* ---------- Full-window accuracy ranking table ---------- */
.rank-table {{
    width: 100%; border-collapse: collapse;
    font-size: 0.875rem; margin: 0.2rem 0 0.5rem 0;
    border: 1px solid {C.GRID}; border-radius: 10px; overflow: hidden;
}}
.rank-table thead th {{
    background: {C.PANEL}; color: {C.SUBTLE};
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; text-align: left;
    padding: 0.55rem 0.9rem; border-bottom: 1px solid {C.GRID};
}}
.rank-table thead th.num {{ text-align: right; }}
.rank-table tbody td {{
    padding: 0.48rem 0.9rem; border-bottom: 1px solid {C.GRID};
    vertical-align: middle; color: {C.INK};
}}
.rank-table tbody tr:last-child td {{ border-bottom: none; }}
.rank-table tbody tr.best {{ background: {C.PANEL}; }}
.rank-table tbody tr:hover {{ background: rgba(247,249,252,0.85); }}
.rank-rk {{
    font-variant-numeric: tabular-nums; color: {C.SUBTLE}; font-weight: 500;
}}
.rank-model {{ font-weight: 600; color: {C.INK}; }}
.rank-family {{
    font-size: 0.78rem; color: {C.SUBTLE}; white-space: nowrap;
}}
.rank-num {{
    font-variant-numeric: tabular-nums; text-align: right;
}}
.rank-num.best {{ font-weight: 600; color: {C.INK}; }}
.rank-bias-pos {{ color: #8B5A52; font-variant-numeric: tabular-nums; }}
.rank-bias-neg {{ color: #4A6678; font-variant-numeric: tabular-nums; }}
.rank-n {{ color: {C.SUBTLE}; font-variant-numeric: tabular-nums; }}
.rank-caption {{
    font-size: 0.78rem; color: {C.SUBTLE}; margin: 0.1rem 0 0.6rem 0;
    line-height: 1.5;
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="hero">{eb}<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def callout(text: str) -> None:
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


def stat_cards(cards: list[tuple[str, str]]) -> None:
    html = '<div class="stat-row">'
    for value, label in cards:
        html += f'<div class="stat-card"><div class="v">{value}</div>' \
                f'<div class="l">{label}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow-dark">{text}</div>', unsafe_allow_html=True)


def _spec_kv_rows(rows: list[tuple[str, str]]) -> str:
    """Render key/value rows for model specification cards."""
    parts = ['<div class="spec-kv">']
    for key, val in rows:
        parts.append(f'<div class="k">{key}</div><div class="v">{val}</div>')
    parts.append("</div>")
    return "".join(parts)


def spec_protocol_panel(title: str, items: list[tuple[str, str]]) -> None:
    """Shared real-time protocol panel for the model-specifications tab."""
    cells = "".join(
        f'<div class="spec-protocol-item"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>'
        for k, v in items
    )
    st.markdown(
        f'<div class="spec-protocol"><div class="ttl">{title}</div>'
        f'<div class="spec-protocol-grid">{cells}</div></div>',
        unsafe_allow_html=True,
    )


def spec_card(
    title: str,
    family: str,
    accent: str,
    description: str,
    rows: list[tuple[str, str]],
    badge: str = "",
    badge_bg: str = C.PANEL,
    badge_fg: str = C.SUBTLE,
    variants: list[tuple[str, str]] | None = None,
    variant_label: str = "Reported variants",
) -> None:
    """Single model-family card with optional variant sub-cards."""
    badge_html = (
        f'<span class="badge" style="background:{badge_bg};color:{badge_fg}">'
        f"{badge}</span>"
        if badge
        else ""
    )
    variants_html = ""
    if variants:
        vcards = "".join(
            f'<div class="spec-variant"><div class="name">{name}</div>'
            f'<div class="note">{note}</div></div>'
            for name, note in variants
        )
        variants_html = (
            f'<div class="spec-variants"><div class="lbl">{variant_label}</div>'
            f"{vcards}</div>"
        )
    st.markdown(
        f'<div class="spec-card" style="border-top:4px solid {accent}">'
        f'<div class="spec-card-hdr" style="background:linear-gradient('
        f"90deg, {accent}12 0%, transparent 55%)\">"
        f'<div class="left"><div class="family" style="color:{accent}">'
        f"{family}</div><div class=\"title\">{title}</div></div>"
        f"{badge_html}</div>"
        f'<div class="spec-card-body"><div class="spec-card-desc">{description}'
        f"</div>{_spec_kv_rows(rows)}{variants_html}</div></div>",
        unsafe_allow_html=True,
    )


def accuracy_ranking_table(rows: list[dict]) -> None:
    """Render the full-window accuracy ranking as a compact HTML table.

    Each row dict must provide: ``rank``, ``model``, ``family``, ``rmse``,
    ``mae``, ``bias``, ``n``.
    """
    if not rows:
        st.info("No model accuracy data available.")
        return

    best_rmse = min(r["rmse"] for r in rows)
    best_mae = min(r["mae"] for r in rows)

    def _num(value: float, is_best: bool) -> str:
        cls = "rank-num best" if is_best else "rank-num"
        return f'<td class="{cls}">{value:.3f}</td>'

    body = []
    for r in rows:
        bias = r["bias"]
        bias_cls = f"rank-num rank-bias-{'pos' if bias >= 0 else 'neg'}"
        tr_cls = " class=\"best\"" if r["rmse"] == best_rmse else ""
        body.append(
            f"<tr{tr_cls}>"
            f'<td><span class="rank-rk">{r["rank"]}</span></td>'
            f'<td><span class="rank-model">{r["model"]}</span></td>'
            f'<td><span class="rank-family">{r["family"]}</span></td>'
            f'{_num(r["rmse"], r["rmse"] == best_rmse)}'
            f'{_num(r["mae"], r["mae"] == best_mae)}'
            f'<td class="{bias_cls}">{bias:+.3f}</td>'
            f'<td class="rank-num rank-n">{r["n"]}</td>'
            "</tr>"
        )

    html = (
        '<table class="rank-table"><thead><tr>'
        "<th>#</th><th>Model</th><th>Family</th>"
        '<th class="num">RMSE (pp)</th><th class="num">MAE (pp)</th>'
        '<th class="num">Bias</th><th class="num">N</th>'
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def spec_baseline_grid(items: list[tuple[str, str]]) -> None:
    """Compact grid of baseline / ensemble definitions."""
    cells = "".join(
        f'<div class="spec-baseline"><div class="name">{name}</div>'
        f'<div class="note">{note}</div></div>'
        for name, note in items
    )
    st.markdown(f'<div class="spec-baseline-grid">{cells}</div>', unsafe_allow_html=True)


def spec_formula_panel(
    title: str,
    blocks: list[tuple[str, str, str | None]],
) -> None:
    """Render labelled model equations for the specifications tab.

    Each block is ``(label, equation_html, optional_note)``.
    """
    inner = []
    for label, eq, note in blocks:
        note_html = (
            f'<div class="note">{note}</div>' if note else ""
        )
        inner.append(
            f'<div class="spec-formula-block"><div class="lbl">{label}</div>'
            f'<div class="eq">{eq}</div>{note_html}</div>'
        )
    st.markdown(
        f'<div class="spec-formula-panel"><div class="ttl">{title}</div>'
        f'{"".join(inner)}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Pipeline flow diagrams (HTML/CSS analogues of the thesis TikZ figures)
# --------------------------------------------------------------------------- #
def _box(title: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="flow-box {tone}"><div class="ttl">{title}</div>'
            f'{sub_html}</div>')


_ARROW = '<div class="flow-arrow">&#9660;</div>'


def selection_flow() -> None:
    """Part I indicator-selection pipeline (HTML analogue of Fig. selection)."""
    rows = [
        _box("Predictor panel &amp; target",
             "585 monthly series · Germany · 1991M1–2025M12 · "
             "first-release vintages · target: real GDP QoQ growth",
             "flow-tone-input"),
        _ARROW,
        _box("Stationarity transformations",
             "series-specific, validated by joint ADF + KPSS",
             "flow-tone-prep"),
        _ARROW,
        _box("Preprocessing",
             "coverage mask ≥ 30% · MICE imputation · standardisation · "
             "back-transform to raw levels → quarterly mean → re-transform",
             "flow-tone-prep"),
        _ARROW,
        '<div class="flow-row">'
        + _box("Elastic Net",
               "ℓ₁+ℓ₂, 5-fold CV · t-stat pre-filter · COVID weights",
               "flow-tone-meth")
        + _box("Block-balanced (k=20)",
               "EN + ≥1 per category · cap 20", "flow-tone-meth")
        + _box("PLS + VIP",
               "H=5 latent components · Part I comparison", "flow-tone-meth")
        + "</div>",
        _ARROW,
        _box("DFM input set",
             "EN-only selection matrix  (180 origins, 2011M1–2025M12)",
             "flow-tone-dfm"),
    ]
    st.markdown('<div class="flow">' + "".join(rows) + "</div>",
                unsafe_allow_html=True)


def nowcast_flow() -> None:
    """Part II nowcasting pipeline (HTML analogue of Fig. nowcast-pipeline)."""
    rows = [
        '<div class="flow-row">'
        + _box("Part I output",
               "EN-only + block-balanced + ifoCAST fixed set", "flow-tone-input")
        + _box("Target",
               "German GDP, QoQ log-growth, first release", "flow-tone-input")
        + "</div>",
        _ARROW,
        _box("Real-time panel preparation",
             "pub-lag mask → AR(<i>p</i>) BIC ragged-edge fill "
             "→ Mariano–Murasawa mixed-frequency encoding "
             "(monthly indicators + GDP at quarter-end)",
             "flow-tone-prep"),
        _ARROW,
        '<div class="flow-row">'
        + _box("Stage 1 — EM-DFM",
               "DynamicFactorMQ (EM) · r=2 factors, AR(2) · Kalman smoother",
               "flow-tone-input")
        + _box("Stage 2 — Bayesian SV",
               "VAR(p) on factors · AR(1) log-vol · NUTS · multiplier √r̄",
               "flow-tone-sv")
        + "</div>",
        _ARROW,
        '<div class="flow-row">'
        + _box("Point nowcast",
               "ŷ_q = E[y | I_t] · Kalman predictive SD", "flow-tone-input")
        + _box("Predictive distribution",
               "σ_pred = σ_em·√r̄ · ŷ_q ± z·σ_pred", "flow-tone-out")
        + "</div>",
        _ARROW,
        _box("Benchmarks (same real-time grid)",
             "RW · AR(1) · XGB-Full · MLP-Factor · equal-weight combo",
             "flow-tone-bench"),
        _ARROW,
        _box("Evaluation — 2011Q1–2025Q4, M1/M2/M3 origins",
             "RMSFE · Mincer–Zarnowitz · Diebold–Mariano · "
             "coverage / CRPS (headline: M3)", "flow-tone-out"),
    ]
    st.markdown('<div class="flow">' + "".join(rows) + "</div>",
                unsafe_allow_html=True)


def _pipe_box(tag: str, title: str, sub: str, tone: str) -> str:
    return (
        f'<div class="flow-box {tone}">'
        f'<div class="tag">{tag}</div>'
        f'<div class="ttl">{title}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def _pipe_arrow() -> str:
    return '<div class="flow-pipe-arrow">&#8594;</div>'


def dfm_interpretation_flow() -> None:
    """DFM-EN interpretation chain: selection → factors → nowcast → decomposition."""
    accent = C.model_color("DFM-EN")
    dfm_bg, _ = C.model_badge("DFM-EN")
    st.markdown(
        f'<div class="ml-wrap" style="border-color:{accent}">'
        f'<div class="hdr" style="color:{accent}">'
        "From indicator selection to nowcast attribution</div>"
        '<div class="lead">'
        "The DFM-EN pipeline is read in <b>three linked steps</b>. Part I decides "
        "<i>which</i> indicators enter the panel; Stage&nbsp;1 describes <i>what "
        "the latent factors represent</i>; Stage&nbsp;2 tracks <i>how strongly each "
        "factor transmits to GDP</i>; the decomposition below shows <i>which "
        "categories moved the actual nowcast</i> at each monthly origin. "
        "Same category colours throughout — but each step measures a different object."
        "</div>"
        f'<div class="model-note" style="background:{dfm_bg}">'
        "<b>Do not merge the charts numerically.</b> Factor loading shares describe "
        "internal factor structure (unsigned, sum to 1 <i>within</i> each factor). "
        "The decomposition attributes the forecast in signed percentage points "
        "(sums to the nowcast). Qualitative consistency is informative; "
        "rank-for-rank comparison is not."
        "</div></div>",
        unsafe_allow_html=True,
    )
    pipe = [
        _pipe_box(
            "Part I",
            "Elastic Net selection",
            "Which indicators enter the panel at each origin?",
            "flow-tone-part1",
        ),
        _pipe_arrow(),
        _pipe_box(
            "Prep",
            "DFM fit",
            "EM-Kalman on the EN-selected set · r = 2 factors",
            "flow-tone-dfm",
        ),
        _pipe_arrow(),
        _pipe_box(
            "Stage 1",
            "Factor loadings",
            "Indicator → factor structure (share of |λ| by category)",
            "flow-tone-stage1",
        ),
        _pipe_arrow(),
        _pipe_box(
            "Output",
            "Nowcast",
            "ŷ<sub>q</sub> = E[GDP growth | information set]",
            "flow-tone-out",
        ),
        _pipe_arrow(),
        _pipe_box(
            "Stage 2 + attribution",
            "TVP bridge &amp; decomposition",
            "Factor → GDP transmission · category contributions (pp)",
            "flow-tone-decomp",
        ),
    ]
    st.markdown('<div class="flow-pipe">' + "".join(pipe) + "</div>",
                unsafe_allow_html=True)
    st.markdown(
        '<div class="flow-ref">'
        "<b>Stage 2 bridge</b> uses the DFM-TVP random-walk loadings (COVID "
        "quarters down-weighted) — the same coefficients plotted below. "
        "<b>Decomposition</b> is a proportional split of the nowcast across "
        "categories at M1/M2/M3 monthly origins (2017–2025)."
        "</div>",
        unsafe_allow_html=True,
    )


def _step_box(num: str, title: str, sub: str = "", tone: str = "",
              example: str = "", left: bool = True,
              step_color: str = "#33415C") -> str:
    align = "left" if left else ""
    ex_html = f'<div class="ex">{example}</div>' if example else ""
    return (
        f'<div class="flow-box {tone} {align}">'
        f'<div class="ttl"><span class="step" style="background:{step_color}">'
        f"{num}</span>{title}</div>"
        f'<div class="sub">{sub}</div>{ex_html}</div>'
    )


def _ml_flow_section(title: str) -> None:
    st.markdown(f'<div class="flow-section">{title}</div>', unsafe_allow_html=True)


def _ml_glossary(terms: list[tuple[str, str]], term_color: str) -> None:
    cells = "".join(
        f'<div class="ml-gloss"><div class="term" style="color:{term_color}">'
        f"{name}</div><div class=\"def\">{body}</div></div>"
        for name, body in terms
    )
    st.markdown(f'<div class="ml-glossary">{cells}</div>', unsafe_allow_html=True)


def _render_html(html: str) -> None:
    """Render HTML via Streamlit without markdown code-block artefacts.
    """
    st.markdown(html, unsafe_allow_html=True)


def _ml_tl_row(label: str, bar_class: str, rng: str, note: str) -> str:
    return (
        '<div class="ml-tl-row">'
        f'<div class="ml-tl-label">{label}</div>'
        f'<div class="ml-tl-bar {bar_class}">'
        f'<div class="rng">{rng}</div>'
        f'<div class="note">{note}</div>'
        '</div></div>'
    )


def _ml_nowcast_intro(
    *,
    border: str,
    hdr_color: str,
    note_bg: str,
    title: str,
    model_note: str,
    extra_timeline: str = "",
    extra_compare: str = "",
) -> None:
    """Shared opening panel: nowcast question + training/target timeline."""
    timeline = "".join([
        _ml_tl_row(
            "Training data", "train",
            "1991Q2 → <i>q</i>−1 &nbsp;(expanding window)",
            "All past quarters used to <b>fit model parameters</b>. "
            "≈79 rows at 2011Q1, growing to ≈135 by 2025Q4. "
            "A fresh model is estimated at every origin.",
        ),
        extra_timeline,
        _ml_tl_row(
            "Nowcast target", "tgt",
            "Quarter <i>q</i> GDP growth (first release)",
            "Predicted before the official GDP release. "
            "Contemporaneous indicator readings for quarter <i>q</i> enter after "
            "the real-time AR fill — not only lags.",
        ),
    ])
    _render_html(
        f'<div class="ml-wrap" style="border-color:{border}">'
        f'<div class="hdr" style="color:{hdr_color}">{title}</div>'
        '<div class="lead">'
        "At each evaluation date we ask: <b>What will German GDP growth be in "
        "the <i>current</i> quarter <i>q</i>?</b> We answer at <b>M3</b> "
        "(month 3 of quarter <i>q</i>) using only the real-time information set "
        "defined in the <b>shared protocol panel above</b> — identical cut-off "
        "to the DFM models."
        "</div>"
        f'<div class="model-note" style="background:{note_bg}">{model_note}</div>'
        f'<div class="ml-timeline">{timeline}</div>'
        f"{extra_compare}"
        "</div>"
    )


def _ml_realtime_prep_step(step_color: str) -> str:
    """Step shared by DFM, XGB, and MLP: publication-lag mask + AR fill."""
    return _step_box(
        "2", "Real-time data preparation",
        "Publication-lag mask → AR(<i>p</i>) BIC ragged-edge fill on the "
        "monthly transformed panel",
        "flow-tone-prep", step_color=step_color,
        example="At 2011M03, unreleased March readings are AR-completed from "
        "past data only; no future information enters.",
    )


def _xgb_quarterly_agg_step(step_color: str) -> str:
    """XGB-only: raw-level bridge after real-time monthly prep."""
    return _step_box(
        "2b", "Aggregate to quarterly (XGB only)",
        "Raw-level bridge: back-transform → quarterly mean → re-transform "
        "to stationary growth rates",
        "flow-tone-prep", step_color=step_color,
        example="The headline DFM skips this step and stays in mixed frequency.",
    )


def xgb_flow() -> None:
    """XGBoost nowcasting workflow for the model-specifications tab."""
    accent = C.model_color("XGB-Full")
    xgb_bg, _ = C.model_badge("XGB-Full")
    _ml_nowcast_intro(
        border="#5DD4B4",
        hdr_color=accent,
        note_bg=xgb_bg,
        title="XGB-Full workflow",
        model_note=(
            "<b>Tabular model.</b> Each quarter is one row of lagged features; "
            "there is no sequence encoder. The full monthly panel (coverage ≥ 30%, "
            "≈580 series) enters with lags L0–L2; SHAP pruning reduces dimension "
            "inside the backtest. Hyperparameters are tuned <b>once</b> on "
            "1991Q1–2010Q4 and held fixed."
        ),
    )

    _ml_flow_section("Terms specific to XGBoost")
    _ml_glossary([
        ("Gradient boosting",
         "An ensemble of small regression trees fitted sequentially; each new "
         "tree corrects the errors of the previous ones (Chen &amp; Guestrin, 2016)."),
        ("SHAP pruning",
         "Every 4 origins, refit a probe model and rank features by mean |SHAP| "
         "contribution. Keep the smallest set reaching 90% cumulative importance "
         "(floor 20 base series). GDP lags L1–L2 are never dropped."),
        ("Empirical intervals",
         "90% bands from the distribution of <b>past forecast errors</b> "
         "(symmetric quantiles), not from the model's loss function. "
         "Available once ≥ 8 prior errors exist."),
    ], term_color=accent)

    _ml_flow_section("Step A — Build the feature matrix")
    prep = [
        _step_box(
            "1", "Select the feature universe",
            "All monthly series with coverage ≥ 30% at the M3 origin "
            "(≈580 base series; <i>no</i> Part I selection mask)",
            "flow-tone-input", step_color=accent,
        ),
        '<div class="flow-arrow lbl">&#9660; shared real-time protocol</div>',
        _ml_realtime_prep_step(accent),
        '<div class="flow-arrow lbl">&#9660; XGB-only aggregation</div>',
        _xgb_quarterly_agg_step(accent),
        '<div class="flow-arrow lbl">&#9660; expand into columns</div>',
        _step_box(
            "3", "Create lagged features",
            "For each series: values at lags L0, L1, L2 (current quarter and "
            "two lags) → ≈1,740 indicator columns",
            "flow-tone-xgb-in", step_color=accent,
            example="L0 = partial-quarter aggregate available at M3; "
            "L1/L2 = previous quarters.",
        ),
        _step_box(
            "4", "Add GDP autoregressive terms",
            "Append GDP growth at lags L1 and L2 — always retained",
            "flow-tone-xgb", step_color=accent,
            example="Closes the gap with the DFM, which embeds past GDP "
            "in its state space (Bańbura et al., 2013).",
        ),
        _step_box(
            "5", "Training matrix",
            "Rows = quarters 1991Q2 → <i>q</i>−1 · one prediction row for "
            "quarter <i>q</i> · remaining NaN filled with column training means",
            "flow-tone-prep", step_color=accent,
        ),
    ]
    st.markdown('<div class="flow">' + "".join(prep) + "</div>",
                unsafe_allow_html=True)

    _ml_flow_section("Step B — Fit trees and produce the nowcast")
    arch = [
        _step_box(
            "6", "SHAP feature screen (every 4 origins)",
            "Drop low-importance indicator columns; keep ≈170–200 base series "
            "(≈510–600 columns incl. lags)",
            "flow-tone-xgb", step_color=accent,
            example="Between refits, the previous feature list is reused "
            "to limit compute.",
        ),
        '<div class="flow-arrow lbl">&#9660; expanding-window fit</div>',
        _step_box(
            "7", "Gradient boosted trees",
            "XGBRegressor on all training rows with fixed tuned hyperparameters "
            "(squared-error loss, histogram method)",
            "flow-tone-xgb-in", step_color=accent,
        ),
        '<div class="flow-arrow lbl">&#9660; single-row prediction</div>',
        '<div class="flow-row">'
        + _step_box("ŷ", "Point nowcast", "GDP growth in quarter <i>q</i> (pp)",
                    "flow-tone-out", left=False, step_color=accent)
        + _step_box("PI", "90% interval", "From past error quantiles",
                    "flow-tone-out", left=False, step_color=accent)
        + "</div>",
        _step_box(
            "⚙", "Fixed hyperparameters (tuned once, 1991Q1–2010Q4)",
            "RandomizedSearchCV · TimeSeriesSplit(5) · n_iter = 40 · "
            "tuned on core set + GDP lags (≈95 features) as a regularisation "
            "choice — then applied to the full SHAP-pruned design",
            "flow-tone-bench", step_color=accent,
        ),
    ]
    st.markdown('<div class="flow">' + "".join(arch) + "</div>",
                unsafe_allow_html=True)

    st.markdown(
        '<div class="flow-ref"><b>References.</b> '
        "Chen &amp; Guestrin (2016); Bańbura, Giannone &amp; Reichlin (2013); "
        "Chatfield (1993) for empirical intervals; "
        "Medeiros et al. (2021); Goulet Coulombe et al. (2022) for fixed-design "
        "tuning. Compare with the <b>MLP-Factor</b> tab: same nowcast target and "
        "real-time protocol, but the MLP receives only the two DFM factors "
        "(plus lags) instead of the wide lagged panel."
        "</div>",
        unsafe_allow_html=True,
    )


def mlp_flow() -> None:
    """Factor-augmented MLP nowcasting workflow for the model-specifications tab."""
    accent = C.model_color("MLP-Factor")
    mlp_bg, _ = C.model_badge("MLP-Factor")

    _ml_nowcast_intro(
        border="#8B7EF0",
        hdr_color=accent,
        note_bg=mlp_bg,
        title="MLP · factor-augmented workflow (non-linearity test)",
        model_note=(
            "<b>Non-linearity test.</b> The panel is first compressed to the "
            "<b>two estimated factors of the headline DFM-EN</b> (read-only refit "
            "at each origin). A tiny, heavily regularised neural net then maps "
            "those factors (plus lags) to GDP. The question is sharp: with the "
            "data already summarised by DFM factors, can a <i>non-linear</i> "
            "factor→GDP map beat the DFM's <i>linear</i> measurement equation? "
            "A tie or loss is an honest, publishable null."
        ),
    )

    _ml_flow_section("Terms specific to the factor-augmented MLP")
    _ml_glossary([
        ("Factor-augmented",
         "Inputs are not the ≈580 raw series but the DFM's two estimated factors "
         "F1, F2 (each at lags L0–L2 → 6 features). The same factors drive the "
         "linear DFM nowcast, isolating the effect of non-linearity alone."),
        ("Heavy L2 (alpha)",
         "Strong weight decay (alpha = 10) keeps a one-hidden-layer net stable on "
         "only ≈80 quarterly observations; tuned once by time-series CV and frozen."),
        ("Seed averaging",
         "The nowcast is the mean of 5 random initialisations, which tames the "
         "init variance of a small network without adding any free parameters."),
    ], term_color=accent)

    _ml_flow_section("Step A — Build the factor features")
    prep = [
        _step_box(
            "1", "Extract DFM-EN factors (read-only)",
            "At each M3 origin, refit the headline DFM-EN (en_only set, "
            "<i>k</i>=2, factor_order=2) and read the smoothed factor states — "
            "the DFM code is never modified",
            "flow-tone-input", step_color=accent,
        ),
        '<div class="flow-arrow lbl">&#9660; shared real-time protocol (inside the DFM fit)</div>',
        _ml_realtime_prep_step(accent),
        '<div class="flow-arrow lbl">&#9660; monthly → quarterly</div>',
        _step_box(
            "3", "Aggregate factors to quarterly",
            "Average the monthly smoothed factors within each quarter → "
            "quarterly F1, F2 (training rows and the prediction row share the "
            "<b>same</b> DFM fit, so sign/rotation flips are irrelevant)",
            "flow-tone-prep", step_color=accent,
        ),
        _step_box(
            "4", "Lag features (factors only)",
            "F1, F2 at lags L0, L1, L2 → 6 features · <b>no</b> GDP "
            "autoregressive terms (keeps the test 'factors → GDP' pure)",
            "flow-tone-mlp-in", step_color=accent,
            example="Unlike XGB, which adds GDP lags and the wide panel.",
        ),
        _step_box(
            "5", "Training matrix + scaling",
            "Rows = quarters 1991Q1 → <i>q</i>−1 · one prediction row for quarter "
            "<i>q</i> · StandardScaler fitted on the training window only",
            "flow-tone-prep", step_color=accent,
        ),
    ]
    st.markdown('<div class="flow">' + "".join(prep) + "</div>",
                unsafe_allow_html=True)

    _ml_flow_section("Step B — Fit the tiny MLP and produce the nowcast")
    arch = [
        _step_box(
            "6", "One-hidden-layer MLP",
            "MLPRegressor · 16 tanh units · L-BFGS solver · heavy L2 "
            "(alpha = 10) · 6 inputs → 1 output",
            "flow-tone-mlp", step_color=accent,
        ),
        '<div class="flow-arrow lbl">&#9660; average 5 random inits</div>',
        _step_box(
            "7", "Seed-averaged prediction",
            "Fit with seeds 0–4 on the expanding window; the nowcast is the mean "
            "across seeds",
            "flow-tone-mlp", step_color=accent,
        ),
        '<div class="flow-arrow lbl">&#9660; single-row prediction</div>',
        _step_box("ŷ", "Point nowcast", "GDP growth in quarter <i>q</i> (pp)",
                  "flow-tone-out", left=False, step_color=accent),
        _step_box(
            "⚙", "Fixed hyperparameters (tuned once, ≤2010Q4)",
            "TimeSeriesSplit(5) grid over hidden units {8, 16} × "
            "alpha {0.01, 0.1, 1, 10} → frozen for all 60 origins",
            "flow-tone-bench", step_color=accent,
        ),
    ]
    st.markdown('<div class="flow">' + "".join(arch) + "</div>",
                unsafe_allow_html=True)

    st.markdown(
        '<div class="flow-ref"><b>References.</b> '
        "Bańbura, Giannone &amp; Reichlin (2013); Stock &amp; Watson (2002) for "
        "factor models; Medeiros et al. (2021); Goulet Coulombe et al. (2022) for "
        "fixed-design machine-learning tuning. Compare with the <b>XGBoost</b> "
        "tab: same nowcast target and real-time protocol, but XGB uses the wide "
        "lagged panel and gradient-boosted trees rather than the two DFM factors."
        "</div>",
        unsafe_allow_html=True,
    )
