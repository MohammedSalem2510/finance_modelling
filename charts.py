import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

def plot_return_distribution(returns, h_var, p_var, t_v):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(returns, bins=30, density=True, color="#CFE2F3",
            alpha=0.8, label="Actual returns", zorder=2)

    x = np.linspace(returns.min(), returns.max(), 1000)
    nu, mu_t, sigma_t = t_dist.fit(returns)
    mu_n = float(returns.mean())
    sigma_n = float(returns.std())

    ax.plot(x, norm.pdf(x, mu_n, sigma_n), color="#1565C0",
            linewidth=1.5, label="Normal distribution")
    ax.plot(x, t_dist.pdf(x, nu, mu_t, sigma_t), color="#FF9800",
            linewidth=1.5, label=f"T-distribution (ν={nu:.2f})")

    ax.axvline(h_var, color="#B71C1C", linestyle="--",
               linewidth=1, label=f"Historical VaR: {h_var:.2%}")
    ax.axvline(p_var, color="#1565C0", linestyle="--",
               linewidth=1, label=f"Parametric VaR: {p_var:.2%}")
    ax.axvline(t_v, color="#FF9800", linestyle="--",
               linewidth=1, label=f"T-dist VaR: {t_v:.2%}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_title("Return Distribution vs Fitted Models",
                fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Daily Return", fontsize=9, color="#555555")
    ax.set_ylabel("Density", fontsize=9, color="#555555")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.tick_params(colors="#555555", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
             ncol=3, framealpha=0, fontsize=8, edgecolor="none")
    plt.subplots_adjust(bottom=0.25)
    return fig

def plot_pnl_drawdown(returns, portfolio_value=10000):
    from risk_engine import drawdown_analysis
    cumulative, drawdown = drawdown_analysis(returns, portfolio_value)
    running_max = cumulative.cummax()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    ax1.plot(cumulative.index, cumulative, color="#1565C0",
            linewidth=1.2, label="Portfolio value", zorder=3)
    ax1.plot(cumulative.index, running_max, color="#888888",
            linewidth=0.8, linestyle="--", label="Peak value", zorder=2)
    ax1.fill_between(cumulative.index, cumulative, running_max,
                    alpha=0.08, color="#1565C0", zorder=1)
    ax1.set_ylabel("Portfolio Value (£)", fontsize=9, color="#555555")
    ax1.set_title("Cumulative P&L and Drawdown",
                 fontsize=11, fontweight="bold", loc="left")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"£{y:,.0f}"))
    ax1.margins(y=0.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_color("#CCCCCC")
    ax1.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax1.xaxis.grid(False)
    ax1.tick_params(colors="#555555", labelsize=8)
    ax1.legend(loc="upper left", framealpha=0, fontsize=8, edgecolor="none")

    ax2.fill_between(drawdown.index, drawdown, 0,
                    color="#B71C1C", alpha=0.2, zorder=1)
    ax2.plot(drawdown.index, drawdown, color="#B71C1C",
            linewidth=1, zorder=2)
    ax2.axhline(drawdown.min(), color="#B71C1C", linestyle=":",
               linewidth=0.8,
               label=f"Max drawdown: {drawdown.min():.2%}", zorder=3)
    ax2.set_ylabel("Drawdown (%)", fontsize=9, color="#555555")
    ax2.set_xlabel("Date", fontsize=9, color="#555555")
    ax2.set_ylim(drawdown.min() * 1.3, 0.01)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#CCCCCC")
    ax2.spines["bottom"].set_color("#CCCCCC")
    ax2.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax2.xaxis.grid(False)
    ax2.tick_params(colors="#555555", labelsize=8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0, fontsize=8, edgecolor="none")
    plt.subplots_adjust(hspace=0.15, bottom=0.15)
    return fig

def plot_monte_carlo(max_drawdowns, actual_drawdown):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(max_drawdowns, bins=50, color="#CFE2F3",
            density=True, label="Simulated max drawdowns", zorder=2)
    ax.axvline(np.percentile(max_drawdowns, 50), color="#1565C0",
               linewidth=1, linestyle="--",
               label=f"Median: {np.percentile(max_drawdowns, 50):.2%}",
               zorder=4)
    ax.axvline(np.percentile(max_drawdowns, 5), color="#FF9800",
               linewidth=1, linestyle="--",
               label=f"95th pct: {np.percentile(max_drawdowns, 5):.2%}",
               zorder=4)
    ax.axvline(actual_drawdown, color="#B71C1C", linewidth=1.5,
               label=f"Actual: {actual_drawdown:.2%}", zorder=4)

    ax.invert_xaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_title("Monte Carlo — Max Drawdown Distribution",
                fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel("Max Drawdown", fontsize=9, color="#555555")
    ax.set_ylabel("Density", fontsize=9, color="#555555")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(colors="#555555", labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
             ncol=3, framealpha=0, fontsize=8, edgecolor="none")
    plt.subplots_adjust(bottom=0.25)
    return fig

def plot_backtest(returns, h_var_series, t_var_series,
                 h_violations, t_violations):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    ax1.plot(returns[h_var_series.index], color="#8B949E",
            linewidth=0.8, label="Actual returns", zorder=2)
    ax1.plot(h_var_series, color="#1565C0", linewidth=1,
            linestyle="--", label="Historical VaR 99%", zorder=3)
    ax1.plot(t_var_series, color="#FF9800", linewidth=1,
            linestyle="--", label="T-dist VaR 99%", zorder=3)

    violation_dates = h_violations[h_violations == 1].index
    ax1.scatter(violation_dates, returns[violation_dates],
               color="#B71C1C", s=50, zorder=4,
               label=f"Violations ({len(violation_dates)})")

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#CCCCCC")
    ax1.spines["bottom"].set_visible(False)
    ax1.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax1.xaxis.grid(False)
    ax1.set_title("VaR Backtest — Actual Returns vs Rolling VaR",
                 fontsize=11, fontweight="bold", loc="left")
    ax1.set_ylabel("Daily Return", fontsize=9, color="#555555")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax1.tick_params(colors="#555555", labelsize=8)
    ax1.margins(y=0.05)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, framealpha=0, fontsize=8, edgecolor="none")

    cumulative_violations = h_violations.cumsum()
    expected_line = pd.Series(
        [(i+1) * 0.01 for i in range(len(h_violations))],
        index=h_violations.index
    )
    ax2.plot(cumulative_violations, color="#1565C0", linewidth=1.5,
            label="Actual cumulative violations", zorder=3)
    ax2.plot(expected_line, color="#B71C1C", linewidth=1,
            linestyle="--", label="Expected (1% rate)", zorder=2)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#CCCCCC")
    ax2.spines["bottom"].set_color("#CCCCCC")
    ax2.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax2.xaxis.grid(False)
    ax2.set_ylabel("Cumulative violations", fontsize=9, color="#555555")
    ax2.set_xlabel("Date", fontsize=9, color="#555555")
    ax2.tick_params(colors="#555555", labelsize=8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0, fontsize=8, edgecolor="none")
    plt.subplots_adjust(hspace=0.15, bottom=0.15)
    return fig

def plot_stress_test(returns, portfolio_value=10000):
    from risk_engine import stress_test
    results = stress_test(returns, portfolio_value)

    colors = {
        "COVID Crash":     "#B71C1C",
        "SVB Collapse":    "#E65100",
        "Ukraine War":     "#F57F17",
        "Rate Hike Shock": "#1565C0"
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor("white")
        cumulative = data["cumulative"]
        total_return = data["total_return"]
        color = colors.get(name, "#1565C0")
        start_value = cumulative.iloc[0]

        ax.plot(cumulative.index, cumulative, color=color,
               linewidth=1.5, zorder=3)
        ax.fill_between(cumulative.index, cumulative, start_value,
                       color=color, alpha=0.15, zorder=2)
        ax.axhline(start_value, color="#888888", linewidth=0.8,
                  linestyle="--", zorder=1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)

        pnl_color = "#B71C1C" if total_return < 0 else "#1565C0"
        ax.set_title(f"{name}   {total_return:+.2%}",
                    fontsize=10, fontweight="bold",
                    loc="left", color=pnl_color)
        ax.set_ylabel("Portfolio Value (£)", fontsize=8, color="#555555")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"£{y:,.0f}"))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(colors="#555555", labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.1)

    plt.tight_layout(pad=2.0)
    return fig

def plot_hedge_analysis(portfolio_returns, hedge_results, all_returns,
                        portfolio_value=10000):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    ax1 = axes[0]
    ax1.set_facecolor("white")
    corr_matrix = all_returns.corr()
    instruments = list(corr_matrix.columns)
    im = ax1.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax1)
    ax1.set_xticks(range(len(instruments)))
    ax1.set_yticks(range(len(instruments)))
    ax1.set_xticklabels(instruments, fontsize=8, color="#555555")
    ax1.set_yticklabels(instruments, fontsize=8, color="#555555")
    for i in range(len(instruments)):
        for j in range(len(instruments)):
            ax1.text(j, i, f"{corr_matrix.iloc[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#1A1A1A")
    ax1.set_title("Correlation Matrix",
                 fontsize=10, fontweight="bold", loc="left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2.set_facecolor("white")
    hedge_names = list(hedge_results.keys())
    categories = ["Unhedged"] + hedge_names
    h_vars = [abs(hedge_results[hedge_names[0]]["h_var_unhedged"]) * portfolio_value]
    t_vars = [abs(hedge_results[hedge_names[0]]["t_var_unhedged"]) * portfolio_value]
    for name in hedge_names:
        h_vars.append(abs(hedge_results[name]["h_var_hedged"]) * portfolio_value)
        t_vars.append(abs(hedge_results[name]["t_var_hedged"]) * portfolio_value)
    x = range(len(categories))
    width = 0.35
    ax2.bar([i - width/2 for i in x], h_vars, width,
            color="#1565C0", alpha=0.8, label="Historical VaR")
    ax2.bar([i + width/2 for i in x], t_vars, width,
            color="#FF9800", alpha=0.8, label="T-dist VaR")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(categories, fontsize=8, color="#555555")
    ax2.set_ylabel("99% VaR (£)", fontsize=9, color="#555555")
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"£{y:,.0f}"))
    ax2.set_title("VaR — Hedged vs Unhedged",
                 fontsize=10, fontweight="bold", loc="left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#CCCCCC")
    ax2.spines["bottom"].set_color("#CCCCCC")
    ax2.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax2.xaxis.grid(False)
    ax2.tick_params(colors="#555555", labelsize=8)
    ax2.legend(framealpha=0, fontsize=8, edgecolor="none")

    colors = ["#1565C0", "#FF9800", "#1D9E75", "#B71C1C", "#534AB7"]
    for ax_idx, (hedge_name, result) in enumerate(hedge_results.items()):
        if ax_idx + 2 >= len(axes):
            break
        ax = axes[ax_idx + 2]
        ax.set_facecolor("white")
        color = colors[ax_idx % len(colors)]

        port_cum = portfolio_value * (
            1 + all_returns["Portfolio"]).cumprod()
        hedged_cum = portfolio_value * (
            1 + result["hedged_returns"]).cumprod()

        ax.plot(port_cum.index, port_cum, color="#B71C1C",
               linewidth=1, label="Unhedged", zorder=3)
        ax.plot(hedged_cum.index, hedged_cum, color=color,
               linewidth=1, label=f"Hedged ({hedge_name})", zorder=3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)

        reduction = result["t_var_reduction"]
        ax.set_title(
            f"{hedge_name} hedge   T-VaR: {reduction:.2%}",
            fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Portfolio Value (£)", fontsize=8, color="#555555")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"£{y:,.0f}"))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(colors="#555555", labelsize=7)
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=0.05)
        ax.legend(framealpha=0, fontsize=8, edgecolor="none")

    plt.tight_layout(pad=2.0)
    return fig

def plot_marginal_var(marginal_vars, portfolio_value=10000):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    tickers = list(marginal_vars.keys())
    values = [abs(v) * portfolio_value for v in marginal_vars.values()]
    colors = ["#B71C1C" if v > 0 else "#1565C0"
              for v in marginal_vars.values()]

    ax.bar(tickers, values, color=colors, alpha=0.8, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_title("Marginal VaR by Asset",
                fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Marginal VaR (£)", fontsize=9, color="#555555")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"£{y:,.2f}"))
    ax.tick_params(colors="#555555", labelsize=9)
    return fig
