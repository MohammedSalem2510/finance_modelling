# Portfolio Risk Analytics Dashboard

> An interactive, multi-tab financial risk modelling web application built with Python and Streamlit. Pulls live market data via Yahoo Finance and applies industry-standard risk methodologies including Value at Risk (VaR), drawdown analysis, VaR backtesting, and hedge effectiveness analysis.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Live Demo

The app is deployed and accessible here: [Launch Dashboard](YOUR_STREAMLIT_URL)

---

## Features

The dashboard is split into five analytical tabs:

### Tab 1 — Risk Overview
- Input any stock/ETF ticker(s) and a custom date range
- Computes **Value at Risk (VaR)** using three methodologies:
  - **Historical VaR** — empirical percentile of past return distribution
  - **Parametric VaR** — Gaussian (normal distribution) assumption
  - **Student's t VaR** — fat-tailed distribution fit, more conservative for financial returns
- Displays annualised **Sharpe Ratio** to assess risk-adjusted performance
- Supports single assets and multi-asset portfolios with custom weighting by investment amount

### Tab 2 — Portfolio Positions
- Build a multi-asset portfolio by entering tickers and position sizes (£)
- Calculates **portfolio-level VaR** using weighted returns
- Computes **Marginal VaR** — the contribution of each individual asset to total portfolio risk
- Shows **diversification benefit** — the reduction in VaR from holding a basket vs individual positions

### Tab 3 — Drawdown Analysis
- Plots **cumulative portfolio value** over the selected period
- Computes and visualises the **historical drawdown curve** (peak-to-trough decline over time)
- Runs a **Monte Carlo simulation** (1,000 paths, 252 trading days) using a fitted Student's t distribution to model the distribution of expected maximum drawdowns going forward

### Tab 4 — VaR Backtesting
- Implements a **rolling 252-day window backtest** of Historical and t-distribution VaR
- Counts **VaR violations** (days where actual return exceeded the VaR threshold)
- Runs the **Kupiec Proportion of Failures (POF) test** — a formal statistical test for VaR model accuracy
- Classifies model performance using the **Basel III traffic light framework**:
  - Green Zone: ≤4 violations (model acceptable)
  - Yellow Zone: 5–9 violations (model under review)
  - Red Zone: ≥10 violations (model rejected)

### Tab 5 — Hedge Analysis
- Analyses portfolio correlation and beta against user-selected hedge instruments (default: FTSE 100, Gold)
- Computes optimal **hedge ratio** using beta-adjusted correlation
- Simulates a **hedged portfolio** and computes VaR before and after hedging
- Quantifies **VaR reduction (%)** from the hedge to evaluate its effectiveness

---

## Project Structure

```
finance_modelling/
├── app.py                    # Streamlit entry point, tab routing
├── risk_engine.py            # Core quantitative risk library
├── charts.py                 # Reusable chart components (Plotly)
├── tabs/
│   ├── tab1_risk.py          # Risk Overview tab
│   ├── tab2_positions.py     # Portfolio Positions tab
│   ├── tab3_drawdown.py      # Drawdown Analysis tab
│   ├── tab4_backtest.py      # VaR Backtesting tab
│   └── tab5_hedge.py         # Hedge Analysis tab
├── 01_data_exploration.ipynb # Exploratory analysis & methodology development
├── requirements.txt
└── .streamlit/               # Streamlit theme/config
```

---

## Risk Engine — Methodology

All quantitative logic lives in `risk_engine.py`. Key functions:

| Function | Description |
|---|---|
| `historical_var()` | Empirical percentile VaR at configurable confidence level |
| `parametric_var()` | Gaussian VaR using mean and standard deviation |
| `t_var()` | VaR using MLE-fitted Student's t distribution |
| `sharpe_ratio()` | Annualised Sharpe ratio (252 trading days, configurable risk-free rate) |
| `marginal_var()` | Per-asset marginal contribution to portfolio VaR via correlation decomposition |
| `diversification_benefit()` | Difference between sum-of-individual VaRs and portfolio VaR |
| `drawdown_analysis()` | Historical cumulative value and drawdown time series |
| `monte_carlo_drawdown()` | Forward-looking drawdown simulation using fitted t-distribution |
| `backtest_var()` | Rolling window VaR backtest with violation tracking |
| `kupiec_test()` | Kupiec POF likelihood ratio test for VaR model validation |
| `basel_zone()` | Basel III traffic light classification based on violation count |
| `stress_test()` | Applies historical stress scenarios (COVID crash, SVB collapse, Ukraine war, rate hike shock) |
| `hedge_analysis()` | Beta-adjusted hedge ratio and VaR comparison for hedged vs unhedged portfolio |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web application framework
- **yfinance** — live market data
- **pandas / numpy** — data manipulation and numerical computation
- **scipy.stats** — statistical distributions (Normal, Student's t, Chi-squared)
- **Plotly** — interactive charts

---

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/MohammedSalem2510/finance_modelling.git
cd finance_modelling

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Usage

1. Navigate to any tab using the tab selector at the top of the dashboard
2. Enter one or more stock tickers (e.g. `AAPL`, `TSLA`, `MSFT`)
3. Select a date range and confidence level (default: 99%)
4. For portfolio tabs, enter position sizes in £ to compute weighted risk metrics
5. All charts are interactive — hover, zoom, and pan using Plotly controls

---

## Concepts Covered

- Market risk quantification (VaR, CVaR framework)
- Fat-tailed return modelling (Student's t distribution)
- Portfolio theory (diversification, marginal risk contribution)
- Model validation (Kupiec test, Basel regulatory framework)
- Monte Carlo simulation for forward-looking risk
- Hedge effectiveness and beta-neutral portfolio construction

---

## Author

**Mohammed Salem**  
MEng Electronic & Electrical Engineering, University of Manchester  
[GitHub](https://github.com/MohammedSalem2510) · [LinkedIn](https://www.linkedin.com/in/mohammed-a-m-salem/)

---

## License

This project is licensed under the MIT License — see [LICENSE.md](LICENSE.md) for details.
