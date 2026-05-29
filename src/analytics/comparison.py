from __future__ import annotations

import pandas as pd

from src.analytics.metrics import compute_metrics
from src.engine.backtest import BacktestEngine
from src.engine.execution import ExecutionModel


def run_strategy_comparison(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    initial_capital: float,
    target_weights: dict,
    equity_universe: list[str],
    diversifiers: dict,
    execution: ExecutionModel,
    cash_return_annual: float,
    stock_hard_trigger: float,
    stock_hard_trim_to: float,
    diversifier_hard_triggers: dict,
    diversifier_hard_trim_to: dict,
    target_weights_by_date: dict | None = None,
):
    variants = [
        {"Strategy": "Annual + Weekly Hard Cap", "scheduled": "Y", "hard": "W"},
        {"Strategy": "Annual Only", "scheduled": "Y", "hard": "NONE"},
        {"Strategy": "Annual + Monthly Hard Cap", "scheduled": "Y", "hard": "M"},
        {"Strategy": "Annual + Quarterly Hard Cap", "scheduled": "Y", "hard": "Q"},
        {"Strategy": "Semiannual Only", "scheduled": "SA", "hard": "NONE"},
        {"Strategy": "Quarterly Only", "scheduled": "Q", "hard": "NONE"},
        {"Strategy": "Monthly Only", "scheduled": "M", "hard": "NONE"},
        {"Strategy": "Buy Once / No Rebalance", "scheduled": "NONE", "hard": "NONE"},
    ]

    rows = []
    for variant in variants:
        engine = BacktestEngine(
            prices=prices,
            target_weights=target_weights,
            target_weights_by_date=target_weights_by_date,
            equity_universe=equity_universe,
            diversifiers=diversifiers,
            execution=execution,
            cash_return_annual=cash_return_annual,
        )

        eq, _, _, log = engine.run(
            initial_capital=initial_capital,
            scheduled_frequency=variant["scheduled"],
            hard_cap_check_frequency=variant["hard"],
            stock_hard_trigger=stock_hard_trigger,
            stock_hard_trim_to=stock_hard_trim_to,
            diversifier_hard_triggers=diversifier_hard_triggers,
            diversifier_hard_trim_to=diversifier_hard_trim_to,
        )
        metrics = compute_metrics(eq, benchmark_prices)
        hard_count = 0 if log.empty else log["Reason"].astype(str).str.contains("hard_cap").sum()
        rows.append(
            {
                "Strategy": variant["Strategy"],
                "Scheduled Frequency": variant["scheduled"],
                "Hard Cap Frequency": variant["hard"],
                "Rebalance Count": len(log),
                "Hard Cap Count": hard_count,
                "Final Equity": metrics.get("Final Equity"),
                "Annual Return / CAGR": metrics.get("Annual Return / CAGR"),
                "Annual Volatility": metrics.get("Annual Volatility"),
                "Sharpe": metrics.get("Sharpe"),
                "Max Drawdown": metrics.get("Max Drawdown"),
                "Calmar": metrics.get("Calmar"),
                "Worst 52W Return": metrics.get("Worst 52W Return"),
                "Beta vs SPY": metrics.get("Beta vs SPY"),
            }
        )
    return pd.DataFrame(rows)
