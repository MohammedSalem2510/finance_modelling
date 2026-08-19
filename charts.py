"""Matplotlib figures for the risk dashboard.

Chart rules, applied uniformly:
  - canvas matches the card surface, no white plates
  - hairline single-axis grids only, no background lattice
  - solid flat fills on bars and areas, never alpha glows under lines
  - thin lines (1.0-1.2px) and small tick labels for terminal density
  - titles live on the Streamlit panel header, not inside the axes
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

import theme

theme.apply_chart_theme()


def plot_return_distribution(returns, h_var, p_var, t_v):
    fig, ax = theme.figure((10, 4))

    ax.hist(returns, bins=30, density=True, color=theme.BLUE_FILL,
            label="Actual returns", zorder=2)

    x = np.linspace(returns.min(), returns.max(), 1000)
    nu, mu_t, sigma_t = t_dist.fit(returns)
    mu_n = float(returns.mean())
    sigma_n = float(returns.std())

    ax.plot(x, norm.pdf(x, mu_n, sigma_n), color=theme.BLUE,
            linewidth=1.2, label="Normal distribution", zorder=3)
    ax.plot(x, t_dist.pdf(x, nu, mu_t, sigma_t), color=theme.AMBER,
            linewidth=1.2, label=f"T-distribution (ν={nu:.2f})", zorder=3)

    ax.axvline(h_var, color=theme.RED, linestyle="--", linewidth=1,
               label=f"Historical VaR: {h_var:.2%}", zorder=4)
    ax.axvline(p_var, color=theme.BLUE, linestyle="--", linewidth=1,
               label=f"Parametric VaR: {p_var:.2%}", zorder=4)
    ax.axvline(t_v, color=theme.AMBER, linestyle="--", linewidth=1,
               label=f"T-dist VaR: {t_v:.2%}", zorder=4)

    theme.style_axes(ax, grid="y")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    theme.percent_axis(ax, axis="x", decimals=1)
    theme.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.28)
    return fig


def plot_pnl_drawdown(returns, portfolio_value=10000):
    from risk_engine import drawdown_analysis
    cumulative, drawdown = drawdown_analysis(returns, portfolio_value)
    running_max = cumulative.cummax()

    fig, (ax1, ax2) = theme.figure((10, 6), nrows=2, sharex=True)

    ax1.plot(cumulative.index, running_max, color=theme.TEXT_DIM,
             linewidth=0.8, linestyle="--", label="Peak value", zorder=2)
    ax1.plot(cumulative.index, cumulative, color=theme.BLUE,
             linewidth=1.2, label="Portfolio value", zorder=3)
    theme.style_axes(ax1, grid="y", bottom_spine=False)
    ax1.set_ylabel("Portfolio value (£)")
    theme.currency_axis(ax1)
    ax1.margins(y=0.05)
    ax1.tick_params(axis="x", length=0)
    theme.legend_above(ax1, ncol=2)

    ax2.fill_between(drawdown.index, drawdown, 0, color=theme.RED_FILL,
                     linewidth=0, zorder=1)
    ax2.plot(drawdown.index, drawdown, color=theme.RED, linewidth=1, zorder=2)
    ax2.axhline(drawdown.min(), color=theme.RED, linestyle=":", linewidth=0.8,
                label=f"Max drawdown: {drawdown.min():.2%}", zorder=3)
    theme.style_axes(ax2, grid="y")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(drawdown.min() * 1.3, 0.01)
    theme.percent_axis(ax2, decimals=0)
    theme.legend(ax2, loc="lower left")
    plt.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.11, hspace=0.10)
    return fig


def plot_monte_carlo(max_drawdowns, actual_drawdown):
    fig, ax = theme.figure((10, 4))

    ax.hist(max_drawdowns, bins=50, color=theme.BLUE_FILL, density=True,
            label="Simulated max drawdowns", zorder=2)
    ax.axvline(np.percentile(max_drawdowns, 50), color=theme.BLUE,
               linewidth=1, linestyle="--", zorder=4,
               label=f"Median: {np.percentile(max_drawdowns, 50):.2%}")
    ax.axvline(np.percentile(max_drawdowns, 5), color=theme.AMBER,
               linewidth=1, linestyle="--", zorder=4,
               label=f"95th pct: {np.percentile(max_drawdowns, 5):.2%}")
    ax.axvline(actual_drawdown, color=theme.RED, linewidth=1.4, zorder=5,
               label=f"Actual: {actual_drawdown:.2%}")

    ax.invert_xaxis()
    theme.style_axes(ax, grid="y")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Density")
    theme.percent_axis(ax, axis="x", decimals=0)
    theme.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.28)
    return fig


def plot_backtest(returns, h_var_series, t_var_series,
                  h_violations, t_violations):
    fig, (ax1, ax2) = theme.figure((10, 6), nrows=2, sharex=True)

    ax1.plot(returns[h_var_series.index], color=theme.TEXT_DIM,
             linewidth=0.7, label="Actual returns", zorder=2)
    ax1.plot(h_var_series, color=theme.BLUE, linewidth=1,
             linestyle="--", label="Historical VaR 99%", zorder=3)
    ax1.plot(t_var_series, color=theme.AMBER, linewidth=1,
             linestyle="--", label="T-dist VaR 99%", zorder=3)

    violation_dates = h_violations[h_violations == 1].index
    ax1.scatter(violation_dates, returns[violation_dates],
                color=theme.RED, s=18, linewidths=0, zorder=4,
                label=f"Violations ({len(violation_dates)})")

    theme.style_axes(ax1, grid="y", bottom_spine=False)
    ax1.set_ylabel("Daily return")
    theme.percent_axis(ax1, decimals=1)
    ax1.margins(y=0.05)
    ax1.tick_params(axis="x", length=0)
    theme.legend_above(ax1, ncol=4)

    cumulative_violations = h_violations.cumsum()
    expected_line = pd.Series(
        [(i + 1) * 0.01 for i in range(len(h_violations))],
        index=h_violations.index
    )
    ax2.plot(expected_line, color=theme.RED, linewidth=1,
             linestyle="--", label="Expected (1% rate)", zorder=2)
    ax2.plot(cumulative_violations, color=theme.BLUE, linewidth=1.2,
             label="Actual cumulative violations", zorder=3)

    theme.style_axes(ax2, grid="y")
    ax2.set_ylabel("Cumulative violations")
    ax2.set_xlabel("Date")
    theme.legend(ax2, loc="upper left", ncol=2)
    plt.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.11, hspace=0.10)
    return fig


def plot_stress_test(returns, portfolio_value=10000):
    from risk_engine import stress_test
    results = stress_test(returns, portfolio_value)

    fig, axes = theme.figure((10, 7), nrows=2, ncols=2)
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        cumulative = data["cumulative"]
        total_return = data["total_return"]
        tone = theme.pnl_tone(total_return)
        line = theme.tone_color(tone)
        fill = theme.tone_fill(tone)
        start_value = cumulative.iloc[0]

        ax.fill_between(cumulative.index, cumulative, start_value,
                        color=fill, linewidth=0, zorder=2)
        ax.plot(cumulative.index, cumulative, color=line,
                linewidth=1.2, zorder=3)
        ax.axhline(start_value, color=theme.TEXT_DIM, linewidth=0.8,
                   linestyle="--", zorder=1)

        theme.style_axes(ax, grid="y")
        ax.set_title(f"{name}    {total_return:+.2%}",
                     fontsize=10, color=line, loc="left")
        ax.set_ylabel("Portfolio value (£)", fontsize=8)
        theme.currency_axis(ax)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.1)

    plt.tight_layout(pad=1.6)
    return fig


def plot_hedge_analysis(portfolio_returns, hedge_results, all_returns,
                        portfolio_value=10000):
    fig, axes = theme.figure((10, 7), nrows=2, ncols=2)
    axes = axes.flatten()

    # --- correlation matrix -------------------------------------------------
    ax1 = axes[0]
    corr_matrix = all_returns.corr()
    instruments = list(corr_matrix.columns)
    im = ax1.imshow(corr_matrix, cmap=theme.correlation_cmap(),
                    vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.outline.set_edgecolor(theme.BORDER)
    cbar.ax.tick_params(colors=theme.TEXT_DIM, labelsize=7, length=2)
    ax1.set_xticks(range(len(instruments)))
    ax1.set_yticks(range(len(instruments)))
    ax1.set_xticklabels(instruments, fontsize=8, color=theme.TEXT_MUTED)
    ax1.set_yticklabels(instruments, fontsize=8, color=theme.TEXT_MUTED)
    for i in range(len(instruments)):
        for j in range(len(instruments)):
            ax1.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     fontweight="600", color=theme.TEXT)
    ax1.set_title("Correlation matrix", fontsize=10, loc="left")
    for side in ax1.spines.values():
        side.set_visible(False)
    ax1.tick_params(length=0)

    # --- VaR: hedged vs unhedged ------------------------------------------
    ax2 = axes[1]
    hedge_names = list(hedge_results.keys())
    categories = ["Unhedged"] + hedge_names
    h_vars = [abs(hedge_results[hedge_names[0]]["h_var_unhedged"]) * portfolio_value]
    t_vars = [abs(hedge_results[hedge_names[0]]["t_var_unhedged"]) * portfolio_value]
    for name in hedge_names:
        h_vars.append(abs(hedge_results[name]["h_var_hedged"]) * portfolio_value)
        t_vars.append(abs(hedge_results[name]["t_var_hedged"]) * portfolio_value)
    x = range(len(categories))
    width = 0.38
    ax2.bar([i - width / 2 for i in x], h_vars, width,
            color=theme.BLUE, label="Historical VaR", zorder=2)
    ax2.bar([i + width / 2 for i in x], t_vars, width,
            color=theme.AMBER, label="T-dist VaR", zorder=2)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(categories, fontsize=8, color=theme.TEXT_MUTED)
    ax2.set_ylabel("99% VaR (£)")
    theme.currency_axis(ax2)
    ax2.set_title("VaR - hedged vs unhedged", fontsize=10, loc="left")
    theme.style_axes(ax2, grid="y")
    theme.legend_above(ax2, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))

    # --- per-instrument equity curves -------------------------------------
    for ax_idx, (hedge_name, result) in enumerate(hedge_results.items()):
        if ax_idx + 2 >= len(axes):
            break
        ax = axes[ax_idx + 2]

        port_cum = portfolio_value * (1 + all_returns["Portfolio"]).cumprod()
        hedged_cum = portfolio_value * (1 + result["hedged_returns"]).cumprod()

        ax.plot(port_cum.index, port_cum, color=theme.TEXT_DIM,
                linewidth=1, label="Unhedged", zorder=2)
        ax.plot(hedged_cum.index, hedged_cum, color=theme.BLUE,
                linewidth=1.2, label=f"Hedged ({hedge_name})", zorder=3)

        theme.style_axes(ax, grid="y")
        reduction = result["t_var_reduction"]
        ax.set_title(f"{hedge_name} hedge    T-VaR {reduction:+.2%}",
                     fontsize=10, loc="left")
        ax.set_ylabel("Portfolio value (£)", fontsize=8)
        theme.currency_axis(ax)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.05)
        theme.legend(ax, loc="upper left")

    plt.tight_layout(pad=1.6)
    return fig


def plot_marginal_var(marginal_vars, portfolio_value=10000):
    fig, ax = theme.figure((8, 4))

    tickers = list(marginal_vars.keys())
    values = [abs(v) * portfolio_value for v in marginal_vars.values()]
    colors = [theme.RED if v > 0 else theme.BLUE
              for v in marginal_vars.values()]

    ax.bar(tickers, values, color=colors, width=0.6, zorder=2)
    theme.style_axes(ax, grid="y")
    ax.set_ylabel("Marginal VaR (£)")
    theme.currency_axis(ax, decimals=2)
    ax.tick_params(labelsize=9)
    plt.tight_layout(pad=0.8)
    return fig


def plot_daily_returns(ticker_returns, ticker):
    """Solid up/down bars for a single asset's daily returns."""
    fig, ax = theme.figure((8, 3))

    # One bar per day with its own colour: two overlapping series would
    # blend to grey at sub-pixel bar widths.
    colors = [theme.GREEN if v >= 0 else theme.RED for v in ticker_returns]
    ax.bar(ticker_returns.index, ticker_returns, color=colors, width=1,
           linewidth=0, zorder=2)
    ax.axhline(0, color=theme.BORDER, linewidth=0.8, zorder=3)

    theme.style_axes(ax, grid="y")
    ax.set_ylabel("Daily return")
    theme.percent_axis(ax, decimals=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(pad=0.8)
    return fig


def plot_correlation_matrix(corr_matrix, figsize=(6, 4)):
    """Standalone correlation heatmap on the palette's diverging ramp."""
    fig, ax = theme.figure(figsize)
    instruments = list(corr_matrix.columns)

    im = ax.imshow(corr_matrix, cmap=theme.correlation_cmap(), vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_edgecolor(theme.BORDER)
    cbar.ax.tick_params(colors=theme.TEXT_DIM, labelsize=7, length=2)

    ax.set_xticks(range(len(instruments)))
    ax.set_yticks(range(len(instruments)))
    ax.set_xticklabels(instruments, fontsize=8, color=theme.TEXT_MUTED)
    ax.set_yticklabels(instruments, fontsize=8, color=theme.TEXT_MUTED)
    for i in range(len(instruments)):
        for j in range(len(instruments)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    fontweight="600", color=theme.TEXT)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=0.8)
    return fig
