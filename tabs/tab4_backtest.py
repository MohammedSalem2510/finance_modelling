import streamlit as st
import pandas as pd
import numpy as np
from charts import plot_backtest, plot_stress_test
from risk_engine import backtest_var, kupiec_test, basel_zone, stress_test


def render(portfolio_ret, portfolio_value, confidence):

    st.markdown('<div class="card-title">VaR Backtest</div>', unsafe_allow_html=True)

    with st.spinner("Running backtest — this may take a moment..."):
        h_var_series, t_var_series, h_violations, t_violations = backtest_var(
            portfolio_ret, confidence=confidence
        )

    n = len(h_violations)
    h_count = int(h_violations.sum())
    t_count = int(t_violations.sum())
    expected = n * (1 - confidence)
    h_kupiec = kupiec_test(n, h_count, confidence)
    t_kupiec = kupiec_test(n, t_count, confidence)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days tested", f"{n:,}")
    with col2:
        st.metric("Expected violations", f"{expected:.1f}")
    with col3:
        st.metric("Historical violations", f"{h_count}",
                 delta=f"{h_count/n:.2%} rate",
                 delta_color="inverse" if h_count > expected * 1.5 else "off")
    with col4:
        st.metric("T-dist violations", f"{t_count}",
                 delta=f"{t_count/n:.2%} rate",
                 delta_color="inverse" if t_count > expected * 1.5 else "off")

    st.divider()

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="card-title">Backtest Chart</div>', unsafe_allow_html=True)
        st.pyplot(plot_backtest(portfolio_ret, h_var_series, t_var_series,
                               h_violations, t_violations))

    with col2:
        st.markdown('<div class="card-title">Model Validation</div>', unsafe_allow_html=True)

        for name, count, kupiec in [
            ("Historical VaR", h_count, h_kupiec),
            ("T-distribution VaR", t_count, t_kupiec)
        ]:
            zone = basel_zone(count)
            if "Green" in zone:
                color = "#0F6E56"
                bg = "#F0FDF4"
                border = "#86EFAC"
            elif "Yellow" in zone:
                color = "#92400E"
                bg = "#FFFBEB"
                border = "#FCD34D"
            else:
                color = "#DC2626"
                bg = "#FEF2F2"
                border = "#FCA5A5"

            kupiec_pass = kupiec > 0.05
            st.markdown(f"""
            <div style="
                background: {bg};
                border: 1px solid {border};
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 14px 16px;
                margin-bottom: 10px;
            ">
                <div style="font-size:13px; font-weight:600; color:{color}; margin-bottom:8px;">{name}</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:4px;">Basel zone: <b style="color:#1C2B4A;">{zone}</b></div>
                <div style="font-size:12px; color:#64748B; margin-bottom:4px;">Violations: <b style="color:#1C2B4A;">{count} of {n:,} days ({count/n:.2%})</b></div>
                <div style="font-size:12px; color:#64748B; margin-bottom:4px;">Kupiec p-value: <b style="color:#1C2B4A;">{kupiec:.4f}</b></div>
                <div style="font-size:12px; color:{'#0F6E56' if kupiec_pass else '#DC2626'};">
                    {'Model passes Kupiec test at 5% significance' if kupiec_pass else 'Model fails Kupiec test — consider recalibration'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="card-title">What this means</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:12px; color:#64748B; line-height:1.7;">
            <p style="margin:0 0 8px;"><b style="color:#1C2B4A;">Basel zones</b> count violations over 250 days:</p>
            <p style="margin:0 0 4px; color:#0F6E56;">0–4 violations → Green — model accurate</p>
            <p style="margin:0 0 4px; color:#92400E;">5–9 violations → Yellow — needs review</p>
            <p style="margin:0 0 8px; color:#DC2626;">10+ violations → Red — model inaccurate</p>
            <p style="margin:0;"><b style="color:#1C2B4A;">Kupiec test</b> — p-value above 0.05 means the violation rate is statistically consistent with a correctly calibrated model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="card-title">Stress Test — Crisis Scenarios</div>', unsafe_allow_html=True)

    with st.spinner("Running stress tests..."):
        stress_results = stress_test(portfolio_ret, portfolio_value)

    st.pyplot(plot_stress_test(portfolio_ret, portfolio_value))

    st.divider()

    st.markdown('<div class="card-title">Stress Test Summary</div>', unsafe_allow_html=True)

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
        st.markdown(f"""
        <div style="
            background: #FEF2F2;
            border: 1px solid #FCA5A5;
            border-left: 4px solid #DC2626;
            border-radius: 8px;
            padding: 14px 16px;
        ">
            <div style="font-size:13px; font-weight:600; color:#DC2626; margin-bottom:4px;">Worst scenario: {worst_scenario[0]}</div>
            <div style="font-size:12px; color:#64748B;">Total return: {worst_scenario[1]['total_return']:+.2%} (£{worst_scenario[1]['total_return'] * portfolio_value:,.0f})</div>
            <div style="font-size:12px; color:#64748B;">Max drawdown: {worst_scenario[1]['max_drawdown']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        color = "#0F6E56" if best_scenario[1]["total_return"] > 0 else "#92400E"
        bg = "#F0FDF4" if best_scenario[1]["total_return"] > 0 else "#FFFBEB"
        border_c = "#86EFAC" if best_scenario[1]["total_return"] > 0 else "#FCD34D"
        st.markdown(f"""
        <div style="
            background: {bg};
            border: 1px solid {border_c};
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 14px 16px;
        ">
            <div style="font-size:13px; font-weight:600; color:{color}; margin-bottom:4px;">Best scenario: {best_scenario[0]}</div>
            <div style="font-size:12px; color:#64748B;">Total return: {best_scenario[1]['total_return']:+.2%} (£{best_scenario[1]['total_return'] * portfolio_value:,.0f})</div>
            <div style="font-size:12px; color:#64748B;">Max drawdown: {best_scenario[1]['max_drawdown']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)