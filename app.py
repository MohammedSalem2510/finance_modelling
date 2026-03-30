import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from risk_engine import (
    fetch_data, compute_returns, compute_portfolio_returns,
    historical_var, parametric_var, t_var, sharpe_ratio,
    marginal_var, diversification_benefit, drawdown_analysis,
    monte_carlo_drawdown, backtest_var, kupiec_test, basel_zone,
    stress_test, hedge_analysis
)
from charts import (
    plot_return_distribution, plot_pnl_drawdown, plot_monte_carlo,
    plot_backtest, plot_stress_test, plot_hedge_analysis,
    plot_marginal_var
)

st.set_page_config(
    page_title="Portfolio Risk Analyser",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Portfolio Risk Analyser")
st.markdown("*Quantitative risk analysis using VaR, Monte Carlo simulation, stress testing and hedge analysis*")
st.divider()

with st.sidebar:
    st.header("Portfolio Builder")

    n_assets = st.number_input("Number of assets", min_value=1, max_value=5, value=1, step=1)

    tickers = []
    amounts = {}

    for i in range(int(n_assets)):
        st.markdown(f"**Asset {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input(f"Ticker", value=["HSBA.L", "LLOY.L", "BARC.L", "BP.L", "VOD.L"][i], key=f"ticker_{i}")
        with col2:
            amount = st.number_input(f"Amount (£)", min_value=100, max_value=1000000, value=10000, step=500, key=f"amount_{i}")
        tickers.append(ticker)
        amounts[ticker] = amount

    st.divider()
    st.header("Settings")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"),
                               max_value=pd.Timestamp.today())
    with col2:
        end_date = st.date_input("End date", value=pd.to_datetime("2024-01-01"),
                             max_value=pd.Timestamp.today())
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    var_limit = st.number_input("VaR limit (£)", min_value=100, max_value=100000, value=300, step=100)

    confidence = st.selectbox(
        "Confidence level",
        options=[0.99, 0.95, 0.90],
        format_func=lambda x: f"{x:.0%}"
    )

    risk_free_rate = st.number_input(
        "Risk free rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.25
    ) / 100

    st.divider()
    st.header("Proposed Trade")
    proposed_ticker = st.text_input("Ticker to add", value="")
    proposed_shares = st.number_input("Number of shares", min_value=0, value=0, step=10)

    run_button = st.button("▶ Run Analysis", type="primary", use_container_width=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Overview",
    "📐 Position Management",
    "📉 Drawdown & Monte Carlo",
    "🔍 Backtest & Stress Test",
    "🛡️ Hedge Analysis"
])

if not run_button:
    st.info("👈 Configure your portfolio in the sidebar and click Run Analysis to begin.")
    st.stop()

with st.spinner("Downloading price data..."):
    price_data = fetch_data(tickers, start_date, end_date)

if price_data.empty:
    st.error("No data found. Please check your ticker symbols.")
    st.stop()

with st.spinner("Running analysis..."):
    returns_df = compute_returns(price_data)
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

st.sidebar.success(f"✅ Loaded {len(portfolio_ret)} trading days")
