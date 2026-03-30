import streamlit as st
import pandas as pd
import yfinance as yf
from risk_engine import historical_var, t_var


def render(portfolio_ret, portfolio_value, weights, amounts,
           tickers, t_v, price_data, returns_df,
           var_limit, confidence, start_date, end_date):

    st.markdown('<div class="card-title">Current Portfolio Positions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
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

        st.divider()

        if len(tickers) > 1:
            st.markdown('<div class="card-title">Asset Correlation</div>', unsafe_allow_html=True)
            corr_matrix = returns_df.corr()

            import matplotlib.pyplot as plt
            import numpy as np

            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            instruments = list(corr_matrix.columns)
            im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(instruments)))
            ax.set_yticks(range(len(instruments)))
            ax.set_xticklabels(instruments, fontsize=9, color="#555555")
            ax.set_yticklabels(instruments, fontsize=9, color="#555555")
            for i in range(len(instruments)):
                for j in range(len(instruments)):
                    ax.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}",
                           ha="center", va="center",
                           fontsize=9, fontweight="bold", color="#1A1A1A")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("Asset Correlation Matrix",
                        fontsize=11, fontweight="bold", loc="left")
            ax.tick_params(colors="#555555", labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.markdown('<div class="card-title">Daily Returns</div>', unsafe_allow_html=True)

            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            ticker_ret = returns_df[tickers[0]]
            positive = ticker_ret.copy()
            negative = ticker_ret.copy()
            positive[positive < 0] = 0
            negative[negative > 0] = 0

            ax.bar(ticker_ret.index, positive, color="#0F6E56", alpha=0.8, width=1)
            ax.bar(ticker_ret.index, negative, color="#DC2626", alpha=0.8, width=1)
            ax.axhline(0, color="#555555", linewidth=0.8)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")
            ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
            ax.xaxis.grid(False)
            ax.set_title(f"{tickers[0]} Daily Returns",
                        fontsize=11, fontweight="bold", loc="left")
            ax.set_ylabel("Daily Return", fontsize=9, color="#555555")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.tick_params(colors="#555555", labelsize=8)
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

    with col2:
        st.markdown('<div class="card-title">Proposed Trade Checker</div>', unsafe_allow_html=True)

        if "trade_result" not in st.session_state:
            st.session_state.trade_result = None
        if "trade_error" not in st.session_state:
            st.session_state.trade_error = None

        tc_col1, tc_col2, tc_col3 = st.columns([1, 1, 2])

        with tc_col1:
            prop_ticker = st.text_input("Ticker", value="HSBA.L", key="tab2_ticker")
        with tc_col2:
            prop_shares = st.number_input("Number of shares", min_value=0, value=100, step=10, key="tab2_shares")
        with tc_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            check_button = st.button("Check Trade", type="primary")

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
                st.session_state.trade_error = str(e) or f"Could not fetch data for {prop_ticker}. Please check the ticker symbol."

        if st.session_state.trade_result:
            r = st.session_state.trade_result
            if r["prop_var_gbp"] <= r["var_limit"]:
                result_color = "#0F6E56"
                result_bg = "#F0FDF4"
                result_border = "#86EFAC"
                result_title = "Trade approved"
                result_body = f"Adding {r['prop_shares']:,} shares of {r['prop_ticker']} at £{r['prop_price']:,.2f} each (total £{r['prop_value']:,.0f}) keeps your VaR within the £{r['var_limit']:,.0f} limit."
            else:
                result_color = "#DC2626"
                result_bg = "#FEF2F2"
                result_border = "#FCA5A5"
                result_title = "Trade exceeds VaR limit"
                result_body = f"This trade generates £{r['prop_var_gbp']:,.0f} of VaR against your £{r['var_limit']:,.0f} limit. Reduce to {r['max_shares']:,} shares or less."

            st.markdown(f"""
            <div style="
                background: {result_bg};
                border: 1px solid {result_border};
                border-left: 4px solid {result_color};
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
            ">
                <div style="font-size:13px; font-weight:600; color:{result_color}; margin-bottom:8px;">{result_title}</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:4px;">{result_body}</div>
                <div style="font-size:12px; color:#64748B; margin-top:8px;">
                    Trade VaR: <b style="color:#1C2B4A;">£{r['prop_var_gbp']:,.0f}</b> &nbsp;|&nbsp;
                    Share price: <b style="color:#1C2B4A;">£{r['prop_price']:,.2f}</b> &nbsp;|&nbsp;
                    Max safe shares: <b style="color:#1C2B4A;">{r['max_shares']:,}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif st.session_state.trade_error:
            st.error(st.session_state.trade_error)

        st.divider()

        st.markdown('<div class="card-title">VaR Limit Status</div>', unsafe_allow_html=True)

        current_var_gbp = abs(t_v) * portfolio_value
        var_utilisation = current_var_gbp / var_limit * 100

        if var_utilisation <= 80:
            status_color = "#0F6E56"
            status_bg = "#F0FDF4"
            status_border = "#86EFAC"
            status_text = "Within limit"
        elif var_utilisation <= 100:
            status_color = "#92400E"
            status_bg = "#FFFBEB"
            status_border = "#FCD34D"
            status_text = "Approaching limit"
        else:
            status_color = "#DC2626"
            status_bg = "#FEF2F2"
            status_border = "#FCA5A5"
            status_text = "Limit exceeded"

        st.markdown(f"""
        <div style="
            background: {status_bg};
            border: 1px solid {status_border};
            border-left: 4px solid {status_color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        ">
            <div style="font-size:13px; font-weight:600; color:{status_color}; margin-bottom:8px;">{status_text}</div>
            <div style="font-size:12px; color:#64748B; margin-bottom:4px;">Current T-dist VaR: <b style="color:#1C2B4A;">£{current_var_gbp:,.0f}</b></div>
            <div style="font-size:12px; color:#64748B; margin-bottom:4px;">VaR limit: <b style="color:#1C2B4A;">£{var_limit:,.0f}</b></div>
            <div style="font-size:12px; color:#64748B;">Utilisation: <b style="color:{status_color};">{var_utilisation:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        for ticker in tickers:
            try:
                current_price = float(price_data[ticker].iloc[-1])
                ticker_returns = returns_df[ticker].dropna()
                ticker_var = t_var(ticker_returns, confidence)
                max_position = var_limit / abs(ticker_var)
                max_shares = int(max_position / current_price)

                st.markdown(f"""
                <div style="
                    background: #FFFFFF;
                    border: 1px solid #E2E8F0;
                    border-radius: 8px;
                    padding: 14px 16px;
                    margin-bottom: 8px;
                ">
                    <div style="font-size:12px; font-weight:600; color:#1C2B4A; margin-bottom:6px;">{ticker} — Max position at £{var_limit:,.0f} VaR limit</div>
                    <div style="font-size:12px; color:#64748B;">Max position: <b style="color:#1C2B4A;">£{max_position:,.0f}</b></div>
                    <div style="font-size:12px; color:#64748B;">Max shares: <b style="color:#1C2B4A;">{max_shares:,}</b></div>
                    <div style="font-size:12px; color:#64748B;">Current price: <b style="color:#1C2B4A;">£{current_price:,.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                continue

