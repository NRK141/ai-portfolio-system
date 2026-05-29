from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min(), drawdown


def rolling_compounded_return(returns: pd.Series, window: int):
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def compute_metrics(
    equity: pd.Series,
    benchmark_prices: pd.Series | None = None,
    periods_per_year: int = 52,
    risk_free_rate: float = 0.0,
):
    equity = equity.dropna()
    returns = equity.pct_change().dropna()

    if len(returns) < 2:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / len(returns)) - 1
    vol = returns.std() * np.sqrt(periods_per_year)

    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_period

    sharpe = np.nan
    if returns.std() != 0:
        sharpe = np.sqrt(periods_per_year) * excess.mean() / returns.std()

    downside = excess[excess < 0]
    sortino = np.nan
    if len(downside) > 1 and downside.std() != 0:
        sortino = np.sqrt(periods_per_year) * excess.mean() / downside.std()

    mdd, dd = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    beta = np.nan
    corr = np.nan
    bench_cagr = np.nan
    bench_vol = np.nan
    bench_mdd = np.nan

    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(equity.index).ffill().dropna()
        aligned = pd.concat([returns.rename("portfolio"), bench.pct_change().rename("benchmark")], axis=1).dropna()

        if len(aligned) > 2:
            cov = aligned["portfolio"].cov(aligned["benchmark"])
            var = aligned["benchmark"].var()
            beta = cov / var if var != 0 else np.nan
            corr = aligned["portfolio"].corr(aligned["benchmark"])

            bench_equity = bench.loc[aligned.index]
            bench_returns = aligned["benchmark"]
            bench_cagr = (bench_equity.iloc[-1] / bench_equity.iloc[0]) ** (periods_per_year / len(bench_returns)) - 1
            bench_vol = bench_returns.std() * np.sqrt(periods_per_year)
            bench_mdd, _ = max_drawdown(bench_equity)

    return {
        "Start": equity.index[0].date(),
        "End": equity.index[-1].date(),
        "Weeks": len(returns),
        "Final Equity": equity.iloc[-1],
        "Total Return": total_return,
        "Annual Return / CAGR": cagr,
        "Annual Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "Worst 13W Return": rolling_compounded_return(returns, 13).min(),
        "Worst 26W Return": rolling_compounded_return(returns, 26).min(),
        "Worst 52W Return": rolling_compounded_return(returns, 52).min(),
        "Beta vs SPY": beta,
        "Correlation vs SPY": corr,
        "SPY Annual Return": bench_cagr,
        "SPY Annual Volatility": bench_vol,
        "SPY Max Drawdown": bench_mdd,
    }


def risk_contribution(prices: pd.DataFrame, target_weights: dict[str, float], periods_per_year: int = 52):
    tickers = [t for t in target_weights if t in prices.columns]
    returns = prices[tickers].pct_change().dropna(how="any")

    if returns.empty:
        return pd.DataFrame()

    cov = returns.cov() * periods_per_year
    w = pd.Series({t: target_weights[t] for t in tickers})
    w = w / w.sum()

    port_var = float(w.T @ cov @ w)
    if port_var <= 0:
        return pd.DataFrame()

    marginal = cov @ w
    contrib = w * marginal / port_var

    return pd.DataFrame({"Weight Used": w, "Risk Contribution %": contrib}).sort_values(
        "Risk Contribution %", ascending=False
    )


def rolling_analytics(equity: pd.Series, benchmark_prices: pd.Series | None, windows: list[int], periods_per_year: int = 52):
    returns = equity.pct_change().dropna()
    out = pd.DataFrame(index=returns.index)

    for w in windows:
        out[f"Rolling {w}W Return"] = rolling_compounded_return(returns, w)
        vol = returns.rolling(w).std() * np.sqrt(periods_per_year)
        mean = returns.rolling(w).mean() * periods_per_year
        out[f"Rolling {w}W Volatility"] = vol
        out[f"Rolling {w}W Sharpe"] = mean / vol

        dd_vals = []
        for i in range(len(equity)):
            if i < w:
                dd_vals.append(np.nan)
            else:
                mdd, _ = max_drawdown(equity.iloc[i - w:i])
                dd_vals.append(mdd)

        out[f"Rolling {w}W MaxDD"] = pd.Series(dd_vals, index=equity.index).reindex(out.index)

    if benchmark_prices is not None:
        bench = benchmark_prices.reindex(equity.index).ffill()
        aligned = pd.concat([returns.rename("p"), bench.pct_change().rename("b")], axis=1).dropna()

        for w in windows:
            out[f"Rolling {w}W Beta vs SPY"] = aligned["p"].rolling(w).cov(aligned["b"]) / aligned["b"].rolling(w).var()
            out[f"Rolling {w}W Corr vs SPY"] = aligned["p"].rolling(w).corr(aligned["b"])

    return out
