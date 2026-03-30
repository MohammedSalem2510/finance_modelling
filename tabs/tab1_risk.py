import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t as t_dist
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

    st.divider()

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="card-title">VaR Comparison</div>', unsafe_allow_html=True)

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

        st.divider()

        st.markdown('<div class="card-title">Portfolio Statistics</div>', unsafe_allow_html=True)

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
            st.divider()
            st.markdown('<div class="card-title">Portfolio Weights</div>', unsafe_allow_html=True)
            weights_data = pd.DataFrame({
                "Ticker": list(weights.keys()),
                "Weight": [f"{w:.1%}" for w in weights.values()],
                "Value (£)": [f"£{amounts[t]:,.0f}" for t in weights.keys()]
            })
            st.dataframe(weights_data, hide_index=True, use_container_width=True)

            st.divider()
            st.markdown('<div class="card-title">Marginal VaR by Asset</div>', unsafe_allow_html=True)
            st.pyplot(plot_marginal_var(m_vars, portfolio_value))

    with col2:
        st.markdown('<div class="card-title">Return Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_return_distribution(portfolio_ret, h_var, p_var, t_v))

        st.divider()

        st.markdown('<div class="card-title">Key Insights</div>', unsafe_allow_html=True)

        fat_tail = nu < 5
        var_gap = abs(h_var - p_var) / abs(p_var) * 100
        sharpe_comment = "strong" if sharpe > 1 else "weak" if sharpe < 0.5 else "moderate"

        insights = [
            {
                "color": "#DC2626" if fat_tail else "#0F6E56",
                "bg": "#FEF2F2" if fat_tail else "#F0FDF4",
                "border": "#FCA5A5" if fat_tail else "#86EFAC",
                "title": f"Fat tails {'detected' if fat_tail else 'not detected'}",
                "body": f"T-distribution degrees of freedom: {nu:.2f}. {'Extreme moves occur more frequently than the normal distribution predicts.' if fat_tail else 'Return distribution is close to normal.'}"
            },
            {
                "color": "#92400E",
                "bg": "#FFFBEB",
                "border": "#FCD34D",
                "title": f"Normal model underestimates risk by {var_gap:.1f}%",
                "body": f"Parametric VaR is £{abs(abs(h_var) - abs(p_var)) * portfolio_value:,.0f} lower than historical VaR. Use t-distribution VaR for more reliable estimates."
            },
            {
                "color": "#0F6E56" if sharpe > 1 else "#DC2626" if sharpe < 0 else "#92400E",
                "bg": "#F0FDF4" if sharpe > 1 else "#FEF2F2" if sharpe < 0 else "#FFFBEB",
                "border": "#86EFAC" if sharpe > 1 else "#FCA5A5" if sharpe < 0 else "#FCD34D",
                "title": f"Sharpe ratio: {sharpe:.2f} — {sharpe_comment} risk-adjusted returns",
                "body": f"Annualised return of {float(portfolio_ret.mean()) * 252:.2%} against volatility of {float(portfolio_ret.std()) * np.sqrt(252):.2%} at a {risk_free_rate:.1%} risk-free rate."
            }
        ]

        for insight in insights:
            st.markdown(f"""
            <div style="
                background: {insight['bg']};
                border: 1px solid {insight['border']};
                border-left: 4px solid {insight['color']};
                border-radius: 8px;
                padding: 14px 16px;
                margin-bottom: 10px;
            ">
                <div style="margin-bottom:4px;">
                    <span style="font-size:13px; font-weight:600; color:{insight['color']};">{insight['title']}</span>
                </div>
                <p style="font-size:12px; color:#64748B; margin:0; line-height:1.5;">{insight['body']}</p>
            </div>
            """, unsafe_allow_html=True)