import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import theme
from risk_engine import (
    InsufficientDataError,
    fetch_data, compute_returns, compute_portfolio_returns,
    historical_var, parametric_var, t_var, sharpe_ratio,
    marginal_var, diversification_benefit, drawdown_analysis,
    monte_carlo_drawdown, backtest_var, kupiec_test, basel_zone,
    stress_test, hedge_analysis
)

st.set_page_config(
    page_title="Portfolio Risk Analyser",
    layout="wide"
)

theme.inject_css()
theme.apply_chart_theme()

theme.header(
    "Portfolio Risk Analyser",
    "Value at Risk / Drawdown / Stress / Hedge",
)

with st.sidebar:
    st.markdown("## Configuration")

    theme.sidebar_section("Portfolio Builder")

    n_assets = st.number_input("Number of assets", min_value=1, max_value=5, value=1, step=1, key="n_assets")

    tickers = []
    amounts = {}

    for i in range(int(n_assets)):
        st.markdown(f'<div class="asset-row">Asset {i+1}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            ticker = st.text_input("Ticker", value=["HSBA.L", "LLOY.L", "BARC.L", "BP.L", "VOD.L"][i], key=f"ticker_{i}")
        with col2:
            amount = st.number_input("£ Amount", min_value=100, max_value=1000000, value=10000, step=500, key=f"amount_{i}")
        tickers.append(ticker)
        amounts[ticker] = amount

    theme.sidebar_section("Date Range")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2020-01-01"), max_value=pd.Timestamp.today())
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp.today(), max_value=pd.Timestamp.today())

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    with st.expander("Advanced Settings"):
        var_limit = st.number_input("VaR limit (£)", min_value=100, max_value=100000, value=300, step=100, key="var_limit")
        confidence = st.selectbox("Confidence level", options=[0.99, 0.95, 0.90], format_func=lambda x: f"{x:.0%}")
        risk_free_rate = st.number_input("Risk free rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25, key="rfr") / 100

if run_button:
    st.session_state.analysis_run = True

if not st.session_state.get("analysis_run", False):
    st.info("Configure the portfolio in the sidebar, then run the analysis.")
    st.stop()

with st.spinner("Downloading price data..."):
    price_data = fetch_data(tickers, start_date, end_date)

if price_data.empty:
    st.error(
        f"No price data returned for {', '.join(dict.fromkeys(tickers))}. "
        "Check the ticker symbols, or retry shortly — the data provider "
        "throttles frequent requests."
    )
    st.stop()

missing = [t for t in dict.fromkeys(tickers) if t not in price_data.columns]
if missing:
    st.warning(
        f"No price data for {', '.join(missing)} — "
        "these holdings were excluded and the remaining weights re-normalised."
    )

tickers = [t for t in dict.fromkeys(tickers) if t in price_data.columns]
amounts = {t: amounts[t] for t in tickers}

if not tickers:
    st.error(
        "None of the requested tickers returned usable price data. "
        "Check the symbols and the selected date range."
    )
    st.stop()

returns_df = compute_returns(price_data[tickers])
if len(returns_df) < 2:
    st.error(
        f"Only {len(returns_df)} day(s) of overlapping return data for "
        f"{', '.join(tickers)}. Widen the date range, or drop tickers whose "
        "trading history does not overlap."
    )
    st.stop()

with st.spinner("Running analysis..."):
    try:
        portfolio_ret, weights = compute_portfolio_returns(returns_df, amounts)
        portfolio_value = sum(amounts.values())

        h_var = historical_var(portfolio_ret, confidence)
        p_var = parametric_var(portfolio_ret, confidence)
        t_v = t_var(portfolio_ret, confidence)
        sharpe = sharpe_ratio(portfolio_ret, risk_free_rate)
        div_benefit = diversification_benefit(returns_df, weights, confidence)
        m_vars = marginal_var(returns_df, weights, confidence)
        cumulative, drawdown = drawdown_analysis(portfolio_ret, portfolio_value)
        max_dd = float(drawdown.min())
    except InsufficientDataError as exc:
        st.error(f"{exc} Try widening the date range or checking the tickers.")
        st.stop()

with st.sidebar:
    theme.status_line("Trading days loaded", f"{len(portfolio_ret):,}")

from tabs import tab1_risk, tab2_positions, tab3_drawdown, tab4_backtest, tab5_hedge

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Overview",
    "Position Management",
    "Drawdown & Monte Carlo",
    "Backtest & Stress Test",
    "Hedge Analysis"
])

with tab1:
    tab1_risk.render(
        portfolio_ret, portfolio_value, weights, amounts,
        tickers, h_var, p_var, t_v, sharpe, max_dd,
        div_benefit, m_vars, risk_free_rate, confidence
    )

with tab2:
    tab2_positions.render(
        portfolio_ret, portfolio_value, weights, amounts,
        tickers, t_v, price_data, returns_df,
        var_limit, confidence, start_date, end_date
    )

with tab3:
    tab3_drawdown.render(
        portfolio_ret, portfolio_value, drawdown, cumulative,
        max_dd, confidence
    )

with tab4:
    tab4_backtest.render(portfolio_ret, portfolio_value, confidence)

with tab5:
    tab5_hedge.render(portfolio_ret, portfolio_value, confidence)
