import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t as t_dist

import theme
from charts import plot_return_distribution, plot_marginal_var
from risk_engine import historical_var, t_var


def render(portfolio_ret, portfolio_value, weights, amounts,
           tickers, h_var, p_var, t_v, sharpe, max_dd,
           div_benefit, m_vars, risk_free_rate, confidence):

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Portfolio Value", f"£{portfolio_value:,.0f}")
    with col2:
        st.metric("Historical VaR 99%", f"£{abs(h_var) * portfolio_value:,.0f}",
                  delta=f"{h_var:.2%}", delta_color="inverse")
    with col3:
        st.metric("T-dist VaR 99%", f"£{abs(t_v) * portfolio_value:,.0f}",
                  delta=f"{t_v:.2%}", delta_color="inverse")
    with col4:
        st.metric("Max Drawdown", f"{max_dd:.2%}",
                  delta=f"{max_dd:.2%}", delta_color="inverse")
    with col5:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                  delta="Good" if sharpe > 1 else "Weak",
                  delta_color="normal" if sharpe > 1 else "inverse")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        theme.panel_title("VaR Comparison")

        worst_var = min(h_var, p_var, t_v)
        best_var = max(h_var, p_var, t_v)

        var_df = pd.DataFrame({
            "Method": ["Historical", "Parametric (Normal)", "T-Distribution"],
            "VaR (%)": [f"{h_var:.3%}", f"{p_var:.3%}", f"{t_v:.3%}"],
            "VaR (£)": [
                f"£{abs(h_var) * portfolio_value:,.2f}",
                f"£{abs(p_var) * portfolio_value:,.2f}",
                f"£{abs(t_v) * portfolio_value:,.2f}"
            ],
            "Risk Level": [
                "Highest" if h_var == worst_var else "Lowest" if h_var == best_var else "Medium",
                "Highest" if p_var == worst_var else "Lowest" if p_var == best_var else "Medium",
                "Highest" if t_v == worst_var else "Lowest" if t_v == best_var else "Medium",
            ]
        })
        st.dataframe(var_df, hide_index=True, use_container_width=True)

        theme.panel_title("Portfolio Statistics")

        nu, mu, sigma = t_dist.fit(portfolio_ret)

        stats_df = pd.DataFrame({
            "Metric": [
                "Mean daily return",
                "Daily volatility",
                "Annualised return",
                "Annualised volatility",
                "T-dist degrees of freedom",
                "Diversification benefit"
            ],
            "Value": [
                f"{float(portfolio_ret.mean()):.4%}",
                f"{float(portfolio_ret.std()):.4%}",
                f"{float(portfolio_ret.mean()) * 252:.2%}",
                f"{float(portfolio_ret.std()) * np.sqrt(252):.2%}",
                f"{nu:.2f}",
                f"£{div_benefit * portfolio_value:,.2f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

        if len(tickers) > 1:
            theme.panel_title("Portfolio Weights")
            weights_data = pd.DataFrame({
                "Ticker": list(weights.keys()),
                "Weight": [f"{w:.1%}" for w in weights.values()],
                "Value (£)": [f"£{amounts[t]:,.0f}" for t in weights.keys()]
            })
            st.dataframe(weights_data, hide_index=True, use_container_width=True)

            theme.panel_title("Marginal VaR by Asset")
            st.pyplot(plot_marginal_var(m_vars, portfolio_value))

    with col2:
        theme.panel_title("Return Distribution vs Fitted Models")
        st.pyplot(plot_return_distribution(portfolio_ret, h_var, p_var, t_v))

        theme.panel_title("Key Insights")

        fat_tail = nu < 5
        var_gap = abs(h_var - p_var) / abs(p_var) * 100
        sharpe_comment = "strong" if sharpe > 1 else "weak" if sharpe < 0.5 else "moderate"
        ann_return = float(portfolio_ret.mean()) * 252
        ann_vol = float(portfolio_ret.std()) * np.sqrt(252)

        theme.callout(
            f"Fat tails {'detected' if fat_tail else 'not detected'}",
            body=(
                "Extreme moves occur more frequently than the normal "
                "distribution predicts."
                if fat_tail else
                "Return distribution is close to normal."
            ),
            rows=[("T-dist degrees of freedom", f"{nu:.2f}")],
            tone="negative" if fat_tail else "positive",
        )

        theme.callout(
            f"Normal model underestimates risk by {var_gap:.1f}%",
            body=(
                "Parametric VaR sits below historical VaR. Use t-distribution "
                "VaR for more reliable tail estimates."
            ),
            rows=[
                ("Understatement",
                 f"£{abs(abs(h_var) - abs(p_var)) * portfolio_value:,.0f}"),
                ("Historical VaR", f"{h_var:.3%}"),
                ("Parametric VaR", f"{p_var:.3%}"),
            ],
            tone="warning",
        )

        theme.callout(
            f"Sharpe ratio {sharpe:.2f} — {sharpe_comment} risk-adjusted returns",
            rows=[
                ("Annualised return", f"{ann_return:.2%}"),
                ("Annualised volatility", f"{ann_vol:.2%}"),
                ("Risk-free rate", f"{risk_free_rate:.2%}"),
            ],
            tone="positive" if sharpe > 1 else "negative" if sharpe < 0 else "warning",
        )
