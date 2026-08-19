import streamlit as st
import pandas as pd
import yfinance as yf

import theme
import tickers as tickers_mod
from charts import plot_correlation_matrix, plot_daily_returns
from risk_engine import historical_var, t_var


def render(portfolio_ret, portfolio_value, weights, amounts,
           tickers, t_v, price_data, returns_df,
           var_limit, confidence, start_date, end_date):

    col1, col2 = st.columns([1.5, 1])

    with col1:
        theme.panel_title("Current Portfolio Positions")

        position_rows = []
        for ticker in tickers:
            try:
                current_price = float(price_data[ticker].iloc[-1])
                ticker_returns = returns_df[ticker].dropna()
                ticker_var = historical_var(ticker_returns, confidence)
                position_value = amounts[ticker]
                position_shares = position_value / current_price
                position_pnl = position_value - (position_shares * float(price_data[ticker].iloc[0]))
                position_rows.append({
                    "Ticker": ticker,
                    "Price": f"£{current_price:,.2f}",
                    "Position (£)": f"£{position_value:,.0f}",
                    "Weight": f"{weights[ticker]:.1%}",
                    "Est. Shares": f"{position_shares:,.0f}",
                    "Asset VaR 99%": f"{ticker_var:.2%}",
                    "P&L": f"£{position_pnl:,.0f}"
                })
            except Exception:
                continue

        positions_df = pd.DataFrame(position_rows)
        st.dataframe(positions_df, hide_index=True, use_container_width=True)

        if len(tickers) > 1:
            theme.panel_title("Asset Correlation Matrix")
            st.pyplot(plot_correlation_matrix(returns_df.corr()))
        else:
            theme.panel_title(f"{tickers[0]} Daily Returns")
            st.pyplot(plot_daily_returns(returns_df[tickers[0]], tickers[0]))

    with col2:
        theme.panel_title("Proposed Trade Checker")

        if "trade_result" not in st.session_state:
            st.session_state.trade_result = None
        if "trade_error" not in st.session_state:
            st.session_state.trade_error = None

        tc_col1, tc_col2 = st.columns(2)
        with tc_col1:
            prop_ticker = tickers_mod.picker("Ticker", key="tab2_ticker")
        with tc_col2:
            prop_shares = st.number_input("Shares", min_value=0, value=100, step=10, key="tab2_shares")

        check_button = st.button("Check Trade", type="primary", use_container_width=True)

        if check_button and prop_ticker and prop_shares > 0:
            st.session_state.trade_result = None
            st.session_state.trade_error = None
            try:
                with st.spinner("Checking trade..."):
                    prop_data = yf.download(prop_ticker, start=start_date, end=end_date, progress=False)
                    prop_price = float(prop_data["Close"].squeeze().iloc[-1])
                    prop_returns = prop_data["Close"].squeeze().pct_change(fill_method=None).dropna()
                    prop_var = t_var(prop_returns, confidence)
                    prop_value = prop_shares * prop_price
                    prop_var_gbp = abs(prop_var) * prop_value
                    max_shares = int((var_limit / abs(prop_var)) / prop_price)
                st.session_state.trade_result = {
                    "prop_var_gbp": prop_var_gbp,
                    "prop_price": prop_price,
                    "prop_value": prop_value,
                    "prop_shares": prop_shares,
                    "prop_ticker": prop_ticker,
                    "max_shares": max_shares,
                    "var_limit": var_limit,
                }
            except Exception as e:
                st.session_state.trade_error = str(e) or f"Could not fetch data for {prop_ticker}."

        if st.session_state.trade_result:
            r = st.session_state.trade_result
            within_limit = r["prop_var_gbp"] <= r["var_limit"]
            if within_limit:
                title = "Trade approved"
                body = (
                    f"Adding {r['prop_shares']:,} shares of {r['prop_ticker']} "
                    f"keeps portfolio VaR within the "
                    f"£{r['var_limit']:,.0f} limit."
                )
            else:
                title = "Trade exceeds VaR limit"
                body = (
                    f"This trade generates £{r['prop_var_gbp']:,.0f} of VaR "
                    f"against a £{r['var_limit']:,.0f} limit. Reduce to "
                    f"{r['max_shares']:,} shares or fewer."
                )

            theme.callout(
                title,
                body=body,
                rows=[
                    ("Trade VaR", f"£{r['prop_var_gbp']:,.0f}"),
                    ("Notional", f"£{r['prop_value']:,.0f}"),
                    ("Price", f"£{r['prop_price']:,.2f}"),
                    ("Max shares", f"{r['max_shares']:,}"),
                ],
                tone="positive" if within_limit else "negative",
            )

        elif st.session_state.trade_error:
            st.error(st.session_state.trade_error)

        theme.panel_title("VaR Limit Status")

        current_var_gbp = abs(t_v) * portfolio_value
        var_utilisation = current_var_gbp / var_limit * 100

        if var_utilisation <= 80:
            tone, status_text = "positive", "Within limit"
        elif var_utilisation <= 100:
            tone, status_text = "warning", "Approaching limit"
        else:
            tone, status_text = "negative", "Limit exceeded"

        theme.callout(
            status_text,
            rows=[
                ("Current T-dist VaR", f"£{current_var_gbp:,.0f}"),
                ("VaR limit", f"£{var_limit:,.0f}"),
                ("Utilisation", f"{var_utilisation:.1f}%"),
            ],
            tone=tone,
        )

        theme.panel_title(f"Max Position at £{var_limit:,.0f} VaR Limit")

        for ticker in tickers:
            try:
                current_price = float(price_data[ticker].iloc[-1])
                ticker_returns = returns_df[ticker].dropna()
                ticker_var = t_var(ticker_returns, confidence)
                max_position = var_limit / abs(ticker_var)
                max_shares = int(max_position / current_price)
                theme.callout(
                    ticker,
                    rows=[
                        ("Max position", f"£{max_position:,.0f}"),
                        ("Max shares", f"{max_shares:,}"),
                        ("Current price", f"£{current_price:,.2f}"),
                        ("Asset T-dist VaR", f"{ticker_var:.2%}"),
                    ],
                    tone="muted",
                )
            except Exception:
                continue
