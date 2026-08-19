import streamlit as st
import pandas as pd
import numpy as np

import theme
from charts import plot_backtest, plot_stress_test
from risk_engine import (
    InsufficientDataError, backtest_var, kupiec_test, basel_zone, stress_test
)

ZONE_TONES = {"Green": "positive", "Yellow": "warning", "Red": "negative"}


def render(portfolio_ret, portfolio_value, confidence):

    theme.panel_title("VaR Backtest")

    with st.spinner("Running backtest — this may take a moment..."):
        try:
            h_var_series, t_var_series, h_violations, t_violations = backtest_var(
                portfolio_ret, confidence=confidence
            )
        except InsufficientDataError:
            st.info(
                f"The backtest needs more than 252 trading days of history; "
                f"this portfolio has {len(portfolio_ret)}. Extend the start "
                "date to run it."
            )
            return

    n = len(h_violations)
    h_count = int(h_violations.sum())
    t_count = int(t_violations.sum())
    expected = n * (1 - confidence)
    h_kupiec = kupiec_test(n, h_count, confidence)
    t_kupiec = kupiec_test(n, t_count, confidence)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days Tested", f"{n:,}")
    with col2:
        st.metric("Expected Violations", f"{expected:.1f}")
    with col3:
        st.metric("Historical Violations", f"{h_count}",
                  delta=f"{h_count/n:.2%} rate",
                  delta_color="inverse" if h_count > expected * 1.5 else "off")
    with col4:
        st.metric("T-dist Violations", f"{t_count}",
                  delta=f"{t_count/n:.2%} rate",
                  delta_color="inverse" if t_count > expected * 1.5 else "off")

    col1, col2 = st.columns([1.6, 1])

    with col1:
        theme.panel_title("Actual Returns vs Rolling VaR")
        st.pyplot(plot_backtest(portfolio_ret, h_var_series, t_var_series,
                                h_violations, t_violations))

    with col2:
        theme.panel_title("Model Validation")

        for name, count, kupiec in [
            ("Historical VaR", h_count, h_kupiec),
            ("T-distribution VaR", t_count, t_kupiec)
        ]:
            zone = basel_zone(count)
            kupiec_pass = kupiec > 0.05
            theme.callout(
                name,
                body=(
                    "Model passes the Kupiec test at 5% significance."
                    if kupiec_pass else
                    "Model fails the Kupiec test — consider recalibration."
                ),
                rows=[
                    ("Basel zone", zone),
                    ("Violations", f"{count} of {n:,} ({count/n:.2%})"),
                    ("Kupiec p-value", f"{kupiec:.4f}"),
                ],
                tone=ZONE_TONES.get(zone, "muted"),
            )

        theme.panel_title("Interpretation")
        st.markdown(
            f"""
            <div class="legend-note">
                <p><b>Basel zones</b> count violations over 250 days:</p>
                <p style="color:{theme.GREEN};">0–4 violations — Green, model accurate</p>
                <p style="color:{theme.AMBER};">5–9 violations — Yellow, needs review</p>
                <p style="color:{theme.RED};">10+ violations — Red, model inaccurate</p>
                <p style="margin-top:8px;"><b>Kupiec test</b> — a p-value above 0.05 means
                the violation rate is statistically consistent with a correctly
                calibrated model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    theme.panel_title("Stress Test — Crisis Scenarios")

    with st.spinner("Running stress tests..."):
        stress_results = stress_test(portfolio_ret, portfolio_value)

    st.pyplot(plot_stress_test(portfolio_ret, portfolio_value))

    theme.panel_title("Stress Test Summary")

    summary_rows = []
    for name, data in stress_results.items():
        summary_rows.append({
            "Scenario": name,
            "Period": f"{data['start']} to {data['end']}",
            "Trading Days": data["n_days"],
            "Total Return": f"{data['total_return']:+.2%}",
            "Worst Day": f"{data['worst_day']:.2%}",
            "Max Drawdown": f"{data['max_drawdown']:.2%}",
            "P&L": f"£{data['total_return'] * portfolio_value:,.0f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    worst_scenario = min(stress_results.items(), key=lambda x: x[1]["total_return"])
    best_scenario = max(stress_results.items(), key=lambda x: x[1]["total_return"])

    col1, col2 = st.columns(2)
    with col1:
        theme.callout(
            f"Worst scenario: {worst_scenario[0]}",
            rows=[
                ("Total return", f"{worst_scenario[1]['total_return']:+.2%}"),
                ("P&L", f"£{worst_scenario[1]['total_return'] * portfolio_value:,.0f}"),
                ("Max drawdown", f"{worst_scenario[1]['max_drawdown']:.2%}"),
            ],
            tone="negative",
        )
    with col2:
        theme.callout(
            f"Best scenario: {best_scenario[0]}",
            rows=[
                ("Total return", f"{best_scenario[1]['total_return']:+.2%}"),
                ("P&L", f"£{best_scenario[1]['total_return'] * portfolio_value:,.0f}"),
                ("Max drawdown", f"{best_scenario[1]['max_drawdown']:.2%}"),
            ],
            tone=theme.pnl_tone(best_scenario[1]["total_return"]),
        )
