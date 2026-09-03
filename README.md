# German GDP nowcast dashboard

Interactive companion to the master's thesis
*[Nowcasting and Indicator Selection in a Data-Rich Environment: An Application to German GDP Growth](#citation)*.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://german-gdp-nowcast-dashboard.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pipeline](https://img.shields.io/badge/pipeline-german--gdp--nowcasting-2D6CB3)](https://github.com/Nikta-Kiani/german-gdp-nowcasting)

**[Live demo](https://german-gdp-nowcast-dashboard.streamlit.app)** (real thesis results) · **[Estimation pipeline](https://github.com/Nikta-Kiani/german-gdp-nowcasting)**

The first official German GDP estimate arrives about a month after the quarter ends. Until then, a reading on the current quarter has to be inferred from monthly indicators. This dashboard is the front end for that exercise: which of 585 monthly series recursive selectors recover, and how competing nowcasts behave in the pre-COVID, COVID and post-COVID windows.

A clean clone runs on a **synthetic demo sample**, so every number you see locally is fabricated. The live demo above uses the real results. Licensed source data are not in this repository.

<p align="center">
  <img src="assets/screenshots/overview.png" width="90%" alt="Dashboard overview">
</p>

<p align="center">
  <img src="assets/screenshots/selection.png" width="44%" alt="Part I — indicator selection">
  &nbsp;
  <img src="assets/screenshots/nowcasting.png" width="44%" alt="Part II — nowcasting results">
</p>

<p align="center"><sub>Screenshots of the three pages with the real thesis results, matching the live demo. A clean clone without staged data runs in demo mode.</sub></p>

## What the thesis finds

**Part I.** Elastic net, a block-balanced variant, partial least squares and gradient-boosting importance all put 65–100% of selected mass on delayed hard activity (production, turnover, orders, trade, construction), against 29% of the panel. They under-weight timely series relative to the panel's 70% lag-0 share. Rank correlations among the four methods are 0.28–0.46; only two series are selected by the elastic net at every origin; mean Jaccard overlap with the frozen ifoCAST set is 0.11.

**Part II.** Over 60 quarters the equal-weight combination of DFM-EN, DFM-block-balanced and DFM-ifoCAST has the lowest RMSFE (0.677), against 0.784 for DFM-EN and 2.406 for an expanding AR(1). Almost all of that gain is the eight pandemic quarters, and no test against the AR(1) rejects. After 2022 a rolling AR(1) leads (0.207), and every reported DFM has a higher average RMSFE at M3 than at M1. The 90% model confidence set retains all eleven headline models.

The monthly panel reduces error when a large disturbance is under way. It does not dominate a short-memory autoregression once growth has settled at a new, low-variance mean.

## Pages

- **Overview** — design, three findings, and the three evaluation regimes
- **Part I · Indicator selection** — category composition over time, cross-method agreement, publication lags
- **Part II · Nowcasting results** — accuracy by regime, within-quarter updates, tests, decompositions, and model cards

All comparisons are pseudo-real-time: first-release GDP and publication lags are respected; historical predictor revisions are not reconstructed.

## Run it locally

```bash
git clone https://github.com/Nikta-Kiani/german-gdp-nowcast-dashboard.git
cd german-gdp-nowcast-dashboard

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>. No extra data setup is required. A "Demo mode" banner appears in the sidebar whenever the synthetic sample is in use.

To point the app at real outputs, drop them in `data/real/` or set `DASHBOARD_DATA_DIR`. The expected layout is in [`data/README.md`](data/README.md).

## Why the real results are not in this repo

The source workbook combines ifo and Macrobond series and cannot be redistributed, so neither can artefacts derived from it. This repository ships code and a synthetic sample only. The live demo loads the real cut from a private data repo at startup; the token never enters git. Details are in [`docs/DATA.md`](docs/DATA.md).

The models themselves are estimated in [german-gdp-nowcasting](https://github.com/Nikta-Kiani/german-gdp-nowcasting).

## Layout

```text
german-gdp-nowcast-dashboard/
├── app.py                      # streamlit run app.py
├── requirements.txt
├── src/dashboard/              # config, data layer, charts, pages
├── data/demo/                  # synthetic sample (tracked)
├── data/README.md              # how to stage real outputs
├── scripts/
│   ├── generate_demo_data.py   # rebuilds data/demo/
│   └── stage_real_data.py      # copies pipeline outputs into data/real/
├── assets/screenshots/
└── docs/DATA.md
```


