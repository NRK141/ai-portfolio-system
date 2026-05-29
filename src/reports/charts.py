from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.metrics import max_drawdown


def save_main_charts(
    equity: pd.Series,
    benchmark_prices: pd.Series,
    initial_capital: float,
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spy_equity = benchmark_prices / benchmark_prices.iloc[0] * initial_capital
    _, dd = max_drawdown(equity)
    _, spy_dd = max_drawdown(spy_equity)

    equity_path = output_dir / "equity_curve.png"
    plt.figure(figsize=(11, 5))
    plt.plot(equity, label="Portfolio")
    plt.plot(spy_equity, label="SPY")
    plt.title("Portfolio Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=160)
    plt.close()

    dd_path = output_dir / "drawdown.png"
    plt.figure(figsize=(11, 4))
    plt.plot(dd, label="Portfolio Drawdown")
    plt.plot(spy_dd, label="SPY Drawdown")
    plt.title("Drawdown Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dd_path, dpi=160)
    plt.close()

    return {"equity_curve": str(equity_path), "drawdown": str(dd_path)}
