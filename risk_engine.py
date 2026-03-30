import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t as t_dist

def fetch_data(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, start=start, end=end, progress=False)
    if len(tickers) == 1:
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        return pd.DataFrame({tickers[0]: close})
    close = data["Close"]
    close.columns = [col[0] if isinstance(col, tuple) else col 
                     for col in close.columns]
    return close

def compute_returns(price_df):
    return price_df.pct_change(fill_method=None).dropna()

def compute_portfolio_returns(returns_df, amounts):
    total = sum(amounts.values())
    weights = {ticker: amount / total for ticker, amount in amounts.items()}
    
    if isinstance(returns_df, pd.Series):
        return returns_df, weights
    
    returns_df = returns_df.copy()
    returns_df.columns = [col[0] if isinstance(col, tuple) else col 
                          for col in returns_df.columns]
    
    available = [t for t in weights.keys() if t in returns_df.columns]
    returns_df = returns_df[available].dropna()
    
    portfolio_ret = sum(
        returns_df[ticker] * weight 
        for ticker, weight in weights.items() 
        if ticker in returns_df.columns
    )
    
    return portfolio_ret, weights

def historical_var(returns, confidence=0.99):
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    return float(np.percentile(returns, (1 - confidence) * 100))

def parametric_var(returns, confidence=0.99):
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    mu = float(returns.mean())
    sigma = float(returns.std())
    return float(mu + sigma * norm.ppf(1 - confidence))

def t_var(returns, confidence=0.99):
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    nu, mu, sigma = t_dist.fit(returns)
    return float(t_dist.ppf(1 - confidence, nu, mu, sigma))

def sharpe_ratio(returns, risk_free_rate=0.05):
    returns = returns.dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    annual_return = float(returns.mean()) * 252
    annual_vol = float(returns.std()) * np.sqrt(252)
    if annual_vol == 0:
        return 0.0
    return (annual_return - risk_free_rate) / annual_vol

def marginal_var(returns_df, weights, confidence=0.99):
    portfolio_ret = (returns_df * pd.Series(weights)).sum(axis=1)
    portfolio_vol = float(portfolio_ret.std())
    marginal_vars = {}
    for ticker in returns_df.columns:
        asset_ret = returns_df[ticker]
        correlation = float(asset_ret.corr(portfolio_ret))
        asset_vol = float(asset_ret.std())
        m_var = correlation * asset_vol * norm.ppf(1 - confidence)
        marginal_vars[ticker] = m_var * weights[ticker]
    return marginal_vars

def diversification_benefit(returns_df, weights, confidence=0.99):
    portfolio_ret, _ = compute_portfolio_returns(returns_df, 
                       {t: w for t, w in weights.items()})
    portfolio_var = abs(historical_var(portfolio_ret, confidence))
    sum_individual = sum(
        abs(historical_var(returns_df[ticker], confidence)) * weight
        for ticker, weight in weights.items()
    )
    return sum_individual - portfolio_var

def drawdown_analysis(returns, portfolio_value=10000):
    returns = returns.dropna()
    cumulative = portfolio_value * (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return cumulative, drawdown

def monte_carlo_drawdown(returns, portfolio_value=10000, 
                         n_simulations=1000, n_days=252):
    returns = returns.dropna()
    nu, mu, sigma = t_dist.fit(returns)
    simulated_returns = t_dist.rvs(
        nu, mu, sigma, size=(n_days, n_simulations)
    )
    cumulative_paths = portfolio_value * np.cumprod(
        1 + simulated_returns, axis=0
    )
    max_drawdowns = []
    for i in range(n_simulations):
        path = cumulative_paths[:, i]
        running_max = np.maximum.accumulate(path)
        drawdown = (path - running_max) / running_max
        max_drawdowns.append(drawdown.min())
    return np.array(max_drawdowns)

def backtest_var(returns, window=252, confidence=0.99):
    returns = returns.dropna()
    historical_violations = []
    t_violations = []
    historical_var_series = []
    t_var_series = []
    dates = []
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        actual_return = returns.iloc[i]
        h_var_rolling = historical_var(window_returns, confidence)
        t_var_rolling = t_var(window_returns, confidence)
        historical_var_series.append(h_var_rolling)
        t_var_series.append(t_var_rolling)
        dates.append(returns.index[i])
        historical_violations.append(
            1 if actual_return < h_var_rolling else 0
        )
        t_violations.append(
            1 if actual_return < t_var_rolling else 0
        )
    historical_var_series = pd.Series(historical_var_series, index=dates)
    t_var_series = pd.Series(t_var_series, index=dates)
    historical_violations = pd.Series(historical_violations, index=dates)
    t_violations = pd.Series(t_violations, index=dates)
    return historical_var_series, t_var_series, historical_violations, t_violations

def kupiec_test(n, violations, confidence):
    from scipy.stats import chi2
    p = 1 - confidence
    if violations == 0:
        return 1.0
    p_hat = violations / n
    lr = -2 * (
        violations * np.log(p / p_hat) +
        (n - violations) * np.log((1 - p) / (1 - p_hat))
    )
    return float(chi2.sf(lr, df=1))

def basel_zone(violations):
    if violations <= 4:
        return "🟢 Green"
    elif violations <= 9:
        return "🟡 Yellow"
    else:
        return "🔴 Red"

def stress_test(returns, portfolio_value=10000):
    scenarios = {
        "COVID Crash":     ("2020-02-19", "2020-03-23"),
        "SVB Collapse":    ("2023-03-08", "2023-03-24"),
        "Ukraine War":     ("2022-02-24", "2022-03-15"),
        "Rate Hike Shock": ("2022-01-01", "2022-06-30")
    }
    results = {}
    for name, (start, end) in scenarios.items():
        scenario_returns = returns[start:end]
        if len(scenario_returns) == 0:
            continue
        cumulative = portfolio_value * (1 + scenario_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        results[name] = {
            "returns":      scenario_returns,
            "cumulative":   cumulative,
            "total_return": float((1 + scenario_returns).prod() - 1),
            "worst_day":    float(scenario_returns.min()),
            "max_drawdown": float(drawdown.min()),
            "n_days":       len(scenario_returns),
            "start":        start,
            "end":          end
        }
    return results

def hedge_analysis(returns, hedge_tickers=["^FTSE", "GLD"]):
    start = returns.index[0]
    end = returns.index[-1]
    results = {}
    all_returns = pd.DataFrame({"Portfolio": returns})
    for ticker in hedge_tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        hedge_ret = data["Close"].squeeze().pct_change().dropna()
        all_returns[ticker] = hedge_ret
    all_returns = all_returns.dropna()
    for ticker in hedge_tickers:
        portfolio_ret = all_returns["Portfolio"]
        hedge_ret = all_returns[ticker]
        correlation = float(portfolio_ret.corr(hedge_ret))
        beta = correlation * (
            float(portfolio_ret.std()) / float(hedge_ret.std())
        )
        hedge_ratio = -beta
        hedged_returns = portfolio_ret + hedge_ratio * hedge_ret
        h_var_unhedged = historical_var(portfolio_ret)
        h_var_hedged = historical_var(hedged_returns)
        t_var_unhedged = t_var(portfolio_ret)
        t_var_hedged = t_var(hedged_returns)
        results[ticker] = {
            "correlation":    correlation,
            "beta":           beta,
            "hedge_ratio":    hedge_ratio,
            "hedged_returns": hedged_returns,
            "h_var_unhedged": h_var_unhedged,
            "h_var_hedged":   h_var_hedged,
            "t_var_unhedged": t_var_unhedged,
            "t_var_hedged":   t_var_hedged,
            "var_reduction":  (h_var_hedged - h_var_unhedged) / abs(h_var_unhedged),
            "t_var_reduction":(t_var_hedged - t_var_unhedged) / abs(t_var_unhedged)
        }
    return results, all_returns