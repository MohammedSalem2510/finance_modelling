import streamlit as st
import pandas as pd
from charts import plot_hedge_analysis
from risk_engine import hedge_analysis


def render(portfolio_ret, portfolio_value, confidence):

    st.markdown('<div class="card-title">Hedge Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Running hedge analysis..."):
        hedge_results, all_returns = hedge_analysis(portfolio_ret)

    best_hedge = max(hedge_results.items(), key=lambda x: x[1]["t_var_reduction"])
    worst_hedge = min(hedge_results.items(), key=lambda x: x[1]["t_var_reduction"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best hedge", best_hedge[0])
    with col2:
        st.metric("T-VaR reduction", f"{best_hedge[1]['t_var_reduction']:.2%}")
    with col3:
        st.metric("Unhedged T-VaR",
                 f"£{abs(best_hedge[1]['t_var_unhedged']) * portfolio_value:,.0f}")
    with col4:
        st.metric("Hedged T-VaR",
                 f"£{abs(best_hedge[1]['t_var_hedged']) * portfolio_value:,.0f}",
                 delta=f"-£{abs(abs(best_hedge[1]['t_var_hedged']) - abs(best_hedge[1]['t_var_unhedged'])) * portfolio_value:,.0f}",
                 delta_color="inverse")

    st.divider()

    st.pyplot(plot_hedge_analysis(portfolio_ret, hedge_results, all_returns, portfolio_value))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card-title">Hedge Summary</div>', unsafe_allow_html=True)

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

    with col2:
        st.markdown('<div class="card-title">Hedge Insights</div>', unsafe_allow_html=True)

        for name, result in hedge_results.items():
            reduction = result["t_var_reduction"]
            corr = result["correlation"]

            if reduction > 0.20:
                color = "#0F6E56"
                bg = "#F0FDF4"
                border = "#86EFAC"
                effectiveness = "Effective hedge"
            elif reduction > 0.05:
                color = "#92400E"
                bg = "#FFFBEB"
                border = "#FCD34D"
                effectiveness = "Partial hedge"
            else:
                color = "#DC2626"
                bg = "#FEF2F2"
                border = "#FCA5A5"
                effectiveness = "Ineffective hedge"

            st.markdown(f"""
            <div style="
                background: {bg};
                border: 1px solid {border};
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 14px 16px;
                margin-bottom: 10px;
            ">
                <div style="font-size:13px; font-weight:600; color:{color}; margin-bottom:6px;">{name} — {effectiveness}</div>
                <div style="font-size:12px; color:#64748B; margin-bottom:2px;">Correlation with portfolio: <b style="color:#1C2B4A;">{corr:.3f}</b></div>
                <div style="font-size:12px; color:#64748B; margin-bottom:2px;">T-dist VaR reduction: <b style="color:#1C2B4A;">{reduction:.2%}</b></div>
                <div style="font-size:12px; color:#64748B; margin-bottom:2px;">
                    Unhedged VaR: <b style="color:#1C2B4A;">£{abs(result['t_var_unhedged']) * portfolio_value:,.0f}</b>
                    → Hedged: <b style="color:#1C2B4A;">£{abs(result['t_var_hedged']) * portfolio_value:,.0f}</b>
                </div>
                <div style="font-size:12px; color:#64748B;">
                    {'Strong negative or positive correlation makes this an effective hedge.' if abs(corr) > 0.5 else 'Low correlation limits hedging effectiveness.' if abs(corr) < 0.2 else 'Moderate correlation provides partial protection.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="card-title">Recommendation</div>', unsafe_allow_html=True)

        best_name = best_hedge[0]
        best_result = best_hedge[1]

        if best_result["t_var_reduction"] > 0.10:
            rec_color = "#0F6E56"
            rec_bg = "#F0FDF4"
            rec_border = "#86EFAC"
            rec_text = f"Consider hedging with {best_name}. A hedge ratio of {abs(best_result['hedge_ratio']):.2f} reduces your 99% VaR by {best_result['t_var_reduction']:.2%}, saving £{abs(abs(best_result['t_var_hedged']) - abs(best_result['t_var_unhedged'])) * portfolio_value:,.0f} of tail risk exposure."
        else:
            rec_color = "#92400E"
            rec_bg = "#FFFBEB"
            rec_border = "#FCD34D"
            rec_text = f"No strong hedge candidate found. The best available hedge ({best_name}) only reduces VaR by {best_result['t_var_reduction']:.2%}. Consider instruments more closely correlated with your portfolio."

        st.markdown(f"""
        <div style="
            background: {rec_bg};
            border: 1px solid {rec_border};
            border-left: 4px solid {rec_color};
            border-radius: 8px;
            padding: 14px 16px;
        ">
            <div style="font-size:13px; font-weight:600; color:{rec_color}; margin-bottom:6px;">Hedging recommendation</div>
            <p style="font-size:12px; color:#64748B; margin:0; line-height:1.6;">{rec_text}</p>
        </div>
        """, unsafe_allow_html=True)