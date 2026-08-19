import streamlit as st
import pandas as pd

import theme
from charts import plot_hedge_analysis
from risk_engine import hedge_analysis


def render(portfolio_ret, portfolio_value, confidence):

    with st.spinner("Running hedge analysis..."):
        hedge_results, all_returns = hedge_analysis(portfolio_ret)

    best_hedge = max(hedge_results.items(), key=lambda x: x[1]["t_var_reduction"])
    worst_hedge = min(hedge_results.items(), key=lambda x: x[1]["t_var_reduction"])

    unhedged_var = abs(best_hedge[1]["t_var_unhedged"]) * portfolio_value
    hedged_var = abs(best_hedge[1]["t_var_hedged"]) * portfolio_value

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Hedge", best_hedge[0])
    with col2:
        st.metric("T-VaR Reduction", f"{best_hedge[1]['t_var_reduction']:.2%}")
    with col3:
        st.metric("Unhedged T-VaR", f"£{unhedged_var:,.0f}")
    with col4:
        st.metric("Hedged T-VaR", f"£{hedged_var:,.0f}",
                  delta=f"-£{abs(hedged_var - unhedged_var):,.0f}",
                  delta_color="inverse")

    theme.panel_title("Hedge Diagnostics")
    st.pyplot(plot_hedge_analysis(portfolio_ret, hedge_results, all_returns, portfolio_value))

    col1, col2 = st.columns(2)

    with col1:
        theme.panel_title("Hedge Summary")

        summary_rows = []
        for name, result in hedge_results.items():
            summary_rows.append({
                "Instrument": name,
                "Correlation": f"{result['correlation']:.3f}",
                "Beta": f"{result['beta']:.3f}",
                "Hedge ratio": f"{result['hedge_ratio']:.3f}",
                "Hist VaR reduction": f"{result['var_reduction']:.2%}",
                "T-dist VaR reduction": f"{result['t_var_reduction']:.2%}"
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        theme.panel_title("Recommendation")

        best_name = best_hedge[0]
        best_result = best_hedge[1]
        strong_candidate = best_result["t_var_reduction"] > 0.10

        if strong_candidate:
            rec_text = (
                f"Consider hedging with {best_name}. A hedge ratio of "
                f"{abs(best_result['hedge_ratio']):.2f} reduces 99% VaR by "
                f"{best_result['t_var_reduction']:.2%}."
            )
            rec_rows = [
                ("Tail risk removed",
                 f"£{abs(abs(best_result['t_var_hedged']) - abs(best_result['t_var_unhedged'])) * portfolio_value:,.0f}"),
                ("Hedge ratio", f"{abs(best_result['hedge_ratio']):.2f}"),
            ]
        else:
            rec_text = (
                f"No strong hedge candidate found. The best available hedge "
                f"({best_name}) reduces VaR by only "
                f"{best_result['t_var_reduction']:.2%}. Consider instruments "
                f"more closely correlated with the portfolio."
            )
            rec_rows = [
                ("Best candidate", best_name),
                ("VaR reduction", f"{best_result['t_var_reduction']:.2%}"),
                ("Weakest candidate", worst_hedge[0]),
            ]

        theme.callout(
            "Hedging recommendation",
            body=rec_text,
            rows=rec_rows,
            tone="positive" if strong_candidate else "warning",
        )

    with col2:
        theme.panel_title("Hedge Insights")

        for name, result in hedge_results.items():
            reduction = result["t_var_reduction"]
            corr = result["correlation"]

            if reduction > 0.20:
                tone, effectiveness = "positive", "Effective hedge"
            elif reduction > 0.05:
                tone, effectiveness = "warning", "Partial hedge"
            else:
                tone, effectiveness = "negative", "Ineffective hedge"

            theme.callout(
                f"{name} — {effectiveness}",
                body=(
                    "Strong correlation makes this an effective hedge."
                    if abs(corr) > 0.5 else
                    "Low correlation limits hedging effectiveness."
                    if abs(corr) < 0.2 else
                    "Moderate correlation provides partial protection."
                ),
                rows=[
                    ("Correlation with portfolio", f"{corr:.3f}"),
                    ("T-dist VaR reduction", f"{reduction:.2%}"),
                    ("Unhedged VaR",
                     f"£{abs(result['t_var_unhedged']) * portfolio_value:,.0f}"),
                    ("Hedged VaR",
                     f"£{abs(result['t_var_hedged']) * portfolio_value:,.0f}"),
                ],
                tone=tone,
            )
