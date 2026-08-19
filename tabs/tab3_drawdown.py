import streamlit as st
import numpy as np
import pandas as pd

import theme
from risk_engine import monte_carlo_drawdown
from charts import plot_pnl_drawdown, plot_monte_carlo


def render(portfolio_ret, portfolio_value, drawdown, cumulative,
           max_dd, confidence):

    final_value = float(cumulative.iloc[-1])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Drawdown", f"{max_dd:.2%}",
                  delta=f"{max_dd:.2%}", delta_color="inverse")
    with col2:
        st.metric("Final Portfolio Value", f"£{final_value:,.0f}",
                  delta=f"£{final_value - portfolio_value:,.0f}")
    with col3:
        recovery_days = int((drawdown < -0.10).sum())
        st.metric("Days in 10%+ Drawdown", f"{recovery_days:,}")
    with col4:
        worst_day = float(portfolio_ret.min())
        st.metric("Worst Single Day", f"{worst_day:.2%}",
                  delta=f"{worst_day:.2%}", delta_color="inverse")

    theme.panel_title("Cumulative P&L and Drawdown")
    st.pyplot(plot_pnl_drawdown(portfolio_ret, portfolio_value))

    col1, col2 = st.columns([1.6, 1])

    with col1:
        theme.panel_title("Monte Carlo — Max Drawdown Distribution")

        with st.spinner("Running Monte Carlo simulation..."):
            max_drawdowns = monte_carlo_drawdown(
                portfolio_ret, portfolio_value, n_simulations=1000
            )

        st.pyplot(plot_monte_carlo(max_drawdowns, max_dd))

    with col2:
        theme.panel_title("Simulation Summary")

        median_dd = float(np.percentile(max_drawdowns, 50))
        pct_95 = float(np.percentile(max_drawdowns, 5))
        worst_sim = float(max_drawdowns.min())
        prob_20 = float((max_drawdowns < -0.20).mean())
        prob_30 = float((max_drawdowns < -0.30).mean())

        sim_df = pd.DataFrame({
            "Metric": [
                "Median max drawdown",
                "95th percentile",
                "Worst simulation",
                "P(drawdown > 20%)",
                "P(drawdown > 30%)"
            ],
            "Value": [
                f"{median_dd:.2%}",
                f"{pct_95:.2%}",
                f"{worst_sim:.2%}",
                f"{prob_20:.2%}",
                f"{prob_30:.2%}"
            ]
        })
        st.dataframe(sim_df, hide_index=True, use_container_width=True)

        theme.panel_title("Drawdown Insights")

        theme.callout(
            f"Actual max drawdown {max_dd:.2%}",
            body=(
                "Severe drawdown exceeding 20% — significant peak-to-trough loss."
                if max_dd < -0.20 else
                "Moderate drawdown between 10% and 20%."
                if max_dd < -0.10 else
                "Mild drawdown below 10%."
            ),
            tone="negative" if max_dd < -0.20 else "warning" if max_dd < -0.10 else "positive",
        )

        theme.callout(
            f"{prob_20:.1%} chance of a 20%+ drawdown in any year",
            body=(
                "Based on 1,000 simulated paths from the fitted "
                "t-distribution with fat tails."
            ),
            rows=[
                ("P(drawdown > 20%)", f"{prob_20:.2%}"),
                ("P(drawdown > 30%)", f"{prob_30:.2%}"),
            ],
            tone="negative" if prob_20 > 0.30 else "warning" if prob_20 > 0.15 else "positive",
        )

        theme.callout(
            f"Actual vs median simulation: {abs(max_dd - median_dd):.2%} gap",
            body=(
                f"The actual drawdown "
                f"{'exceeded' if max_dd < median_dd else 'was better than'} "
                f"the simulated median."
            ),
            rows=[
                ("Actual", f"{max_dd:.2%}"),
                ("Simulated median", f"{median_dd:.2%}"),
            ],
            tone="muted",
        )
