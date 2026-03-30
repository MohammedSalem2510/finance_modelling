def render(portfolio_ret, portfolio_value, drawdown, cumulative,
           max_dd, confidence):

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max Drawdown", f"{max_dd:.2%}",
                 delta=f"{max_dd:.2%}", delta_color="inverse")
    with col2:
        st.metric("Final Portfolio Value",
                 f"£{float(cumulative.iloc[-1]):,.0f}",
                 delta=f"£{float(cumulative.iloc[-1]) - portfolio_value:,.0f}")
    with col3:
        recovery_days = int((drawdown < -0.10).sum())
        st.metric("Days in 10%+ Drawdown", f"{recovery_days}")
    with col4:
        worst_day = float(portfolio_ret.min())
        st.metric("Worst Single Day",
                 f"{worst_day:.2%}", delta_color="inverse")

    st.divider()

    st.markdown('<div class="card-title">Cumulative P&L and Drawdown</div>',
               unsafe_allow_html=True)
    st.pyplot(plot_pnl_drawdown(portfolio_ret, portfolio_value))

    st.divider()

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="card-title">Monte Carlo Simulation</div>',
                   unsafe_allow_html=True)

        with st.spinner("Running Monte Carlo simulation..."):
            max_drawdowns = monte_carlo_drawdown(
                portfolio_ret, portfolio_value, n_simulations=1000
            )

        st.pyplot(plot_monte_carlo(max_drawdowns, max_dd))

    with col2:
        st.markdown('<div class="card-title">Simulation Summary</div>',
                   unsafe_allow_html=True)

        median_dd = float(np.percentile(max_drawdowns, 50))
        pct_95 = float(np.percentile(max_drawdowns, 5))
        worst_sim = float(max_drawdowns.min())
        prob_20 = float((max_drawdowns < -0.20).mean())
        prob_30 = float((max_drawdowns < -0.30).mean())

        import pandas as pd
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

        st.divider()

        st.markdown('<div class="card-title">Drawdown Insights</div>',
                   unsafe_allow_html=True)

        insights = [
            {
                "color": "#DC2626" if max_dd < -0.20 else "#92400E" if max_dd < -0.10 else "#0F6E56",
                "bg": "#FEF2F2" if max_dd < -0.20 else "#FFFBEB" if max_dd < -0.10 else "#F0FDF4",
                "border": "#FCA5A5" if max_dd < -0.20 else "#FCD34D" if max_dd < -0.10 else "#86EFAC",
                "title": f"Actual max drawdown: {max_dd:.2%}",
                "body": f"{'Severe drawdown exceeding 20% — significant peak to trough loss.' if max_dd < -0.20 else 'Moderate drawdown between 10-20%.' if max_dd < -0.10 else 'Mild drawdown below 10%.'}"
            },
            {
                "color": "#DC2626" if prob_20 > 0.30 else "#92400E" if prob_20 > 0.15 else "#0F6E56",
                "bg": "#FEF2F2" if prob_20 > 0.30 else "#FFFBEB" if prob_20 > 0.15 else "#F0FDF4",
                "border": "#FCA5A5" if prob_20 > 0.30 else "#FCD34D" if prob_20 > 0.15 else "#86EFAC",
                "title": f"{prob_20:.1%} chance of 20%+ drawdown in any year",
                "body": f"Based on 1,000 simulated paths using your fitted t-distribution with fat tails."
            },
            {
                "color": "#1C2B4A",
                "bg": "#F8FAFC",
                "border": "#E2E8F0",
                "title": f"Actual drawdown vs median simulation: {abs(max_dd - median_dd):.2%} difference",
                "body": f"Your actual drawdown of {max_dd:.2%} {'exceeded' if max_dd < median_dd else 'was better than'} the simulated median of {median_dd:.2%}."
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