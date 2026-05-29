from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.engine.backtest import BacktestEngine
from src.engine.execution import ExecutionModel


def _get_rebalance_counts(log: pd.DataFrame) -> dict[str, int]:
    if log is None or log.empty or "Reason" not in log.columns:
        return {
            "Rebalance Count": 0,
            "Scheduled Count": 0,
            "Hard Cap Count": 0,
        }

    reason = log["Reason"].astype(str)
    return {
        "Rebalance Count": len(log),
        "Scheduled Count": int(reason.str.contains("scheduled").sum()),
        "Hard Cap Count": int(reason.str.contains("hard_cap").sum()),
    }


def run_strategy_comparison(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    target_weights: dict[str, float],
    equity_universe: list[str],
    diversifiers: dict[str, float],
    execution_config: dict[str, Any],
    rebalance_config: dict[str, Any],
    initial_capital: float,
    variants: list[dict[str, Any]],
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """Run all configured strategy variants and return a comparison table."""
    rows = []

    for variant in variants:
        name = variant["name"]
        scheduled_frequency = str(variant.get("scheduled_frequency", rebalance_config["scheduled_frequency"]))
        hard_cap_frequency = str(variant.get("hard_cap_check_frequency", rebalance_config["hard_cap_check_frequency"]))

        execution = ExecutionModel(
            whole_shares=bool(execution_config["whole_shares"]),
            commission_per_trade=float(execution_config["commission_per_trade"]),
            slippage_bps=float(execution_config["slippage_bps"]),
            min_trade_dollars=float(execution_config["min_trade_dollars"]),
        )

        engine = BacktestEngine(
            prices=prices,
            target_weights=target_weights,
            equity_universe=equity_universe,
            diversifiers=diversifiers,
            execution=execution,
            cash_return_annual=float(execution_config["cash_return_annual"]),
            periods_per_year=periods_per_year,
        )

        equity, cash, holdings, log = engine.run(
            initial_capital=initial_capital,
            scheduled_frequency=scheduled_frequency,
            hard_cap_check_frequency=hard_cap_frequency,
            stock_hard_trigger=float(rebalance_config["stock_hard_trigger"]),
            stock_hard_trim_to=float(rebalance_config["stock_hard_trim_to"]),
            diversifier_hard_triggers={
                k: float(v) for k, v in rebalance_config["diversifier_hard_triggers"].items()
            },
            diversifier_hard_trim_to={
                k: float(v) for k, v in rebalance_config["diversifier_hard_trim_to"].items()
            },
        )

        metrics = compute_metrics(
            equity=equity,
            benchmark_prices=benchmark_prices,
            periods_per_year=periods_per_year,
        )

        counts = _get_rebalance_counts(log)

        row = {
            "Strategy": name,
            "Scheduled Frequency": scheduled_frequency,
            "Hard Cap Frequency": hard_cap_frequency,
            **counts,
            "Final Equity": metrics.get("Final Equity", np.nan),
            "Total Return": metrics.get("Total Return", np.nan),
            "Annual Return / CAGR": metrics.get("Annual Return / CAGR", np.nan),
            "Annual Volatility": metrics.get("Annual Volatility", np.nan),
            "Sharpe": metrics.get("Sharpe", np.nan),
            "Sortino": metrics.get("Sortino", np.nan),
            "Max Drawdown": metrics.get("Max Drawdown", np.nan),
            "Calmar": metrics.get("Calmar", np.nan),
            "Worst 13W Return": metrics.get("Worst 13W Return", np.nan),
            "Worst 26W Return": metrics.get("Worst 26W Return", np.nan),
            "Worst 52W Return": metrics.get("Worst 52W Return", np.nan),
            "Beta vs SPY": metrics.get("Beta vs SPY", np.nan),
            "Correlation vs SPY": metrics.get("Correlation vs SPY", np.nan),
        }
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if not comparison.empty:
        comparison = comparison.sort_values(
            by=["Sharpe", "Calmar", "Annual Return / CAGR"],
            ascending=False,
        ).reset_index(drop=True)

    return comparison
