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
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 2rem; 
        max-width: 1400px; 
    }
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 24px 0 24px;
        margin-bottom: 8px;
    }
    .app-header-left { display: flex; align-items: center; gap: 12px; }
    .app-logo {
        width: 36px; height: 36px;
        background: #1C2B4A;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px;
    }
    .app-title {
        font-size: 20px;
        font-weight: 700;
        color: #1C2B4A;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-badge {
        font-size: 11px;
        font-weight: 600;
        color: #0F6E56;
        background: #E1F5EE;
        padding: 3px 10px;
        border-radius: 99px;
        letter-spacing: 0.02em;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #1C2B4A;
        line-height: 1.1;
    }
    .kpi-value.negative { color: #DC2626; }
    .kpi-value.positive { color: #0F6E56; }
    .kpi-sub {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 4px;
    }
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 13px;
        font-weight: 600;
        color: #1C2B4A;
        letter-spacing: -0.1px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #64748B;
        font-weight: 500;
        font-size: 13px;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1C2B4A !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1C2B4A !important;
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 7px !important;
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 7px !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #FFFFFF !important;
        border: none !important;
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > button {
        background-color: #FFFFFF !important;
        border: none !important;
        border-left: 1px solid #E2E8F0 !important;
        color: #1C2B4A !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > button:hover {
        background-color: #F1F5F9 !important;
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > button p,
    [data-testid="stSidebar"] .stNumberInput > div > div > button svg {
        color: #1C2B4A !important;
        fill: #1C2B4A !important;
        stroke: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 7px !important;
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div > div {
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
        color: #1C2B4A !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stDateInput input {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 7px !important;
        color: #1C2B4A !important;
    }
    [data-testid="stSidebar"] .stExpander {
        background-color: #243352 !important;
        border: 1px solid #2D3F5E !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stExpander:hover {
        background-color: #243352 !important;
        border: 1px solid #4A90A4 !important;
    }
    [data-testid="stSidebar"] .stExpander summary {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }
    [data-testid="stSidebar"] .stExpander summary:hover {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stExpander summary p {
        color: #E2E8F0 !important;
    }
    .sidebar-section {
        font-size: 10px;
        font-weight: 700;
        color: #4A90A4 !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2D3F5E;
    }
    .asset-number {
        font-size: 10px;
        font-weight: 700;
        color: #4A90A4 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .stButton button {
        background-color: #0F6E56 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px !important;
        letter-spacing: 0.02em;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background-color: #085041 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15,110,86,0.3) !important;
    }
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="metric-container"] label {
        color: #64748B !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1C2B4A !important;
        font-weight: 700 !important;
        font-size: 22px !important;
    }
    hr { border-color: #E2E8F0; margin: 12px 0; }
    .stAlert { border-radius: 10px; }
    [data-testid="stDataFrame"] { 
        border-radius: 10px; 
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stNumberInput > div > div > button p {
    font-size: 12px !important;
    font-weight: 400 !important;
    color: #1C2B4A !important;
    font-family: sans-serif !important;
    }
    [data-testid="stSidebar"] .stExpander summary:hover p,
    [data-testid="stSidebar"] details summary {
        background-color: #243352 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] details[open] summary {
        background-color: #243352 !important;
        color: #E2E8F0 !important;
        border-radius: 8px 8px 0 0 !important;
    }
    [data-testid="stSidebar"] details summary * {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] details[open] summary * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <div class="app-logo">📊</div>
        <div class="app-title">Portfolio Risk Analyser</div>
        <div class="app-badge">Live Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Portfolio Risk Analyser")
    st.markdown('<div class="sidebar-section">Portfolio Builder</div>', unsafe_allow_html=True)

    n_assets = st.number_input("Number of assets", min_value=1, max_value=5, value=1, step=1, key="n_assets")

    tickers = []
    amounts = {}

    for i in range(int(n_assets)):
        st.markdown(f'<div class="asset-number">Asset {i+1}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            ticker = st.text_input("Ticker", value=["HSBA.L", "LLOY.L", "BARC.L", "BP.L", "VOD.L"][i], key=f"ticker_{i}")
        with col2:
            amount = st.number_input("£ Amount", min_value=100, max_value=1000000, value=10000, step=500, key=f"amount_{i}")
        tickers.append(ticker)
        amounts[ticker] = amount

    st.markdown('<div class="sidebar-section">Date Range</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2020-01-01"), max_value=pd.Timestamp.today())
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp.today(), max_value=pd.Timestamp.today())

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("▶  Run Analysis", type="primary", use_container_width=True)

    with st.expander("Advanced Settings"):
        var_limit = st.number_input("VaR limit (£)", min_value=100, max_value=100000, value=300, step=100, key="var_limit")
        confidence = st.selectbox("Confidence level", options=[0.99, 0.95, 0.90], format_func=lambda x: f"{x:.0%}")
        risk_free_rate = st.number_input("Risk free rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.25, key="rfr") / 100

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Overview",
    "📐 Position Management",
    "📉 Drawdown & Monte Carlo",
    "🔍 Backtest & Stress Test",
    "🛡️ Hedge Analysis"
])

if not run_button:
    st.info("Configure your portfolio settings in the sidebar and click Run Analysis to begin.")
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