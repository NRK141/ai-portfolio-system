from __future__ import annotations

import pandas as pd

from src.analytics.metrics import compute_metrics
from src.engine.backtest import BacktestEngine
from src.engine.execution import ExecutionModel
from src.strategy.weights import make_target_weights_by_date, latest_target_weights_on_or_before
from src.universe.point_in_time_market_caps import build_point_in_time_universe_by_date


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
    selected_ranking_frequency: str | None = None,
):
    """
    Compare hard-cap / rebalance variants for the currently selected strategy.

    If target_weights_by_date is provided, the labels are PIT-aware. The old
    "Buy Once / No Rebalance" label is not used because PIT target dates
    intentionally trigger ticker rotation.
    """
    if target_weights_by_date:
        selected_label = str(selected_ranking_frequency or "PIT").upper()
        variants = [
            {"Strategy": f"PIT {selected_label} Rotation + Weekly Hard Cap", "scheduled": "NONE", "hard": "W"},
            {"Strategy": f"PIT {selected_label} Rotation Only", "scheduled": "NONE", "hard": "NONE"},
            {"Strategy": f"PIT {selected_label} Rotation + Monthly Hard Cap", "scheduled": "NONE", "hard": "M"},
            {"Strategy": f"PIT {selected_label} Rotation + Quarterly Hard Cap", "scheduled": "NONE", "hard": "Q"},
        ]
    else:
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

        eq, cash, holdings, log = engine.run(
            initial_capital=initial_capital,
            scheduled_frequency=variant["scheduled"],
            hard_cap_check_frequency=variant["hard"],
            stock_hard_trigger=stock_hard_trigger,
            stock_hard_trim_to=stock_hard_trim_to,
            diversifier_hard_triggers=diversifier_hard_triggers,
            diversifier_hard_trim_to=diversifier_hard_trim_to,
        )

        metrics = compute_metrics(eq, benchmark_prices)

        if log.empty:
            rebalance_count = 0
            hard_count = 0
        else:
            rebalance_count = len(log)
            hard_count = log["Reason"].astype(str).str.contains("hard_cap").sum()

        rows.append(
            {
                "Strategy": variant["Strategy"],
                "Scheduled Frequency": variant["scheduled"],
                "Hard Cap Frequency": variant["hard"],
                "Rebalance Count": rebalance_count,
                "Hard Cap Count": hard_count,
                "Final Equity": metrics.get("Final Equity"),
                "Annual Return / CAGR": metrics.get("Annual Return / CAGR"),
                "Annual Volatility": metrics.get("Annual Volatility"),
                "Sharpe": metrics.get("Sharpe"),
                "Sortino": metrics.get("Sortino"),
                "Max Drawdown": metrics.get("Max Drawdown"),
                "Calmar": metrics.get("Calmar"),
                "Worst 52W Return": metrics.get("Worst 52W Return"),
                "Beta vs SPY": metrics.get("Beta vs SPY"),
                "Correlation vs SPY": metrics.get("Correlation vs SPY"),
            }
        )

    return pd.DataFrame(rows)


def run_pit_ranking_frequency_comparison(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    market_caps: pd.DataFrame,
    ranking_frequencies: list[str],
    anchors: list[str],
    total_positions: int,
    equity_weight: float,
    diversifiers: dict,
    execution: ExecutionModel,
    cash_return_annual: float,
    initial_capital: float,
    hard_cap_check_frequency: str,
    stock_hard_trigger: float,
    stock_hard_trim_to: float,
    diversifier_hard_triggers: dict,
    diversifier_hard_trim_to: dict,
    asof_lag_days: int = 0,
    min_market_cap: float = 0.0,
):
    """
    Test PIT ranking refresh frequencies. The strategy rotates tickers whenever
    the selected ranking frequency produces a new universe.
    """
    rows = []
    membership_frames = []

    for freq in [str(f).upper() for f in ranking_frequencies]:
        universe_by_date, membership = build_point_in_time_universe_by_date(
            prices_index=prices.index,
            market_caps=market_caps,
            anchors=anchors,
            total_positions=total_positions,
            available_cols=prices.columns,
            ranking_frequency=freq,
            asof_lag_days=asof_lag_days,
            min_market_cap=min_market_cap,
        )

        target_weights_by_date = make_target_weights_by_date(
            universe_by_date=universe_by_date,
            equity_weight=equity_weight,
            diversifier_targets=diversifiers,
        )

        equity_universe = list(dict.fromkeys(membership["ticker"].dropna().astype(str).tolist()))
        target_weights = latest_target_weights_on_or_before(target_weights_by_date, prices.index[-1])

        engine = BacktestEngine(
            prices=prices,
            target_weights=target_weights,
            target_weights_by_date=target_weights_by_date,
            equity_universe=equity_universe,
            diversifiers=diversifiers,
            execution=execution,
            cash_return_annual=cash_return_annual,
        )

        eq, cash, holdings, log = engine.run(
            initial_capital=initial_capital,
            scheduled_frequency="NONE",
            hard_cap_check_frequency=hard_cap_check_frequency,
            stock_hard_trigger=stock_hard_trigger,
            stock_hard_trim_to=stock_hard_trim_to,
            diversifier_hard_triggers=diversifier_hard_triggers,
            diversifier_hard_trim_to=diversifier_hard_trim_to,
        )

        metrics = compute_metrics(eq, benchmark_prices)

        membership_frames.append(membership.assign(test_ranking_frequency=freq))

        rows.append(
            {
                "Ranking Frequency": freq,
                "Strategy": f"PIT {freq} Ranking Rotation + {hard_cap_check_frequency} Hard Cap",
                "Ranking Dates": membership["ranking_date"].nunique(),
                "Unique Tickers Used": membership["ticker"].nunique(),
                "Rebalance Count": 0 if log.empty else len(log),
                "Hard Cap Count": 0 if log.empty else log["Reason"].astype(str).str.contains("hard_cap").sum(),
                "Final Equity": metrics.get("Final Equity"),
                "Annual Return / CAGR": metrics.get("Annual Return / CAGR"),
                "Annual Volatility": metrics.get("Annual Volatility"),
                "Sharpe": metrics.get("Sharpe"),
                "Sortino": metrics.get("Sortino"),
                "Max Drawdown": metrics.get("Max Drawdown"),
                "Calmar": metrics.get("Calmar"),
                "Worst 52W Return": metrics.get("Worst 52W Return"),
                "Beta vs SPY": metrics.get("Beta vs SPY"),
                "Correlation vs SPY": metrics.get("Correlation vs SPY"),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("Annual Return / CAGR", ascending=False)
    all_membership = pd.concat(membership_frames, ignore_index=True) if membership_frames else pd.DataFrame()

    return comparison, all_membership
