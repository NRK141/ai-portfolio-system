from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.engine.execution import ExecutionModel


def get_rebalance_dates(index: pd.Index, freq: str) -> list[pd.Timestamp]:
    """Return the first available trading date for each requested period."""
    freq = str(freq).upper()
    idx = pd.DatetimeIndex(index)

    if len(idx) == 0:
        return []

    if freq == "NONE":
        return []
    if freq == "ONCE":
        return [idx[0]]
    if freq == "W":
        return list(idx)
    if freq == "M":
        return pd.Series(idx, index=idx).groupby(idx.to_period("M")).first().tolist()
    if freq == "Q":
        return pd.Series(idx, index=idx).groupby(idx.to_period("Q")).first().tolist()
    if freq == "SA":
        df = pd.DataFrame(index=idx)
        df["year"] = idx.year
        df["half"] = np.where(idx.month <= 6, 1, 2)
        return df.groupby(["year", "half"]).apply(lambda x: x.index[0]).tolist()
    if freq in ["Y", "A"]:
        return pd.Series(idx, index=idx).groupby(idx.to_period("Y")).first().tolist()

    raise ValueError(f"Unsupported rebalance frequency: {freq}")


def current_values(row: pd.Series, shares: dict[str, float], cash: float):
    values = {}
    holdings_value = 0.0

    for ticker, qty in shares.items():
        px = row.get(ticker, np.nan)
        val = qty * px if pd.notna(px) else 0.0
        values[ticker] = val
        holdings_value += val

    return values, holdings_value, holdings_value + cash


class BacktestEngine:
    def __init__(
        self,
        prices: pd.DataFrame,
        target_weights: Dict[str, float],
        equity_universe: list[str],
        diversifiers: Dict[str, float],
        execution: ExecutionModel,
        cash_return_annual: float = 0.0,
        periods_per_year: int = 52,
    ):
        self.prices = prices.dropna(how="all").copy()
        self.target_weights = target_weights
        self.tickers = list(target_weights.keys())
        self.equity_universe = set(equity_universe)
        self.diversifiers = diversifiers
        self.execution = execution
        self.cash_return_annual = float(cash_return_annual)
        self.periods_per_year = int(periods_per_year)

    def apply_cash_return(self, cash: float) -> float:
        if self.cash_return_annual == 0:
            return cash

        rate = (1 + self.cash_return_annual) ** (1 / self.periods_per_year) - 1
        return cash * (1 + rate)

    def full_rebalance(self, row: pd.Series, shares: dict[str, float], cash: float):
        shares = shares.copy()
        _, _, portfolio_value = current_values(row, shares, cash)

        if portfolio_value <= 0:
            return shares, cash

        target_values = {
            ticker: portfolio_value * weight
            for ticker, weight in self.target_weights.items()
        }

        # 1) Sell overweight positions first. This makes cash available before buys.
        for ticker, target_value in target_values.items():
            px = row.get(ticker, np.nan)

            if pd.isna(px) or px <= 0:
                continue

            current_value = shares.get(ticker, 0.0) * px

            if current_value > target_value:
                desired_qty = target_value / px
                sell_qty = shares.get(ticker, 0.0) - desired_qty
                shares, cash, _ = self.execution.sell_quantity(ticker, sell_qty, px, shares, cash)

        # 2) Recompute and buy underweight positions.
        _, _, portfolio_value = current_values(row, shares, cash)
        target_values = {
            ticker: portfolio_value * weight
            for ticker, weight in self.target_weights.items()
        }

        deficits = []
        for ticker, target_value in target_values.items():
            px = row.get(ticker, np.nan)

            if pd.isna(px) or px <= 0:
                continue

            current_value = shares.get(ticker, 0.0) * px
            deficit = target_value - current_value

            if deficit > self.execution.min_trade_dollars:
                deficits.append((ticker, deficit))

        deficits = sorted(deficits, key=lambda x: x[1], reverse=True)

        for ticker, deficit in deficits:
            px = row.get(ticker, np.nan)

            if pd.isna(px) or px <= 0:
                continue

            shares, cash, _ = self.execution.buy_with_budget(ticker, deficit, px, shares, cash)

        return shares, cash

    def hard_cap_trim(
        self,
        row: pd.Series,
        shares: dict[str, float],
        cash: float,
        stock_hard_trigger: float,
        stock_hard_trim_to: float,
        diversifier_hard_triggers: Dict[str, float],
        diversifier_hard_trim_to: Dict[str, float],
    ):
        """Trim hard-cap breaches and redeploy freed cash to underweights.

        This is intentionally single-name based. Cluster caps should remain alerts only
        unless you explicitly choose to add cluster trading rules later.
        """
        shares = shares.copy()
        values, _, portfolio_value = current_values(row, shares, cash)

        if portfolio_value <= 0:
            return shares, cash, []

        triggers = []
        cash_before = cash

        # 1) Sell any hard-cap breaches to their trim-to weight.
        for ticker, value in values.items():
            px = row.get(ticker, np.nan)

            if pd.isna(px) or px <= 0:
                continue

            current_weight = value / portfolio_value
            trigger = None
            trim_to = None

            if ticker in self.equity_universe:
                trigger = stock_hard_trigger
                trim_to = stock_hard_trim_to
            elif ticker in self.diversifiers:
                trigger = diversifier_hard_triggers.get(ticker)
                trim_to = diversifier_hard_trim_to.get(ticker)

            if trigger is None or trim_to is None:
                continue

            if current_weight > trigger:
                trim_value = portfolio_value * trim_to

                if value > trim_value:
                    sell_value = value - trim_value
                    sell_qty = sell_value / px
                    shares, cash, _ = self.execution.sell_quantity(ticker, sell_qty, px, shares, cash)
                    triggers.append(
                        f"{ticker}: {current_weight:.2%} > {trigger:.2%}; trim to {trim_to:.2%}"
                    )

        freed_cash = max(cash - cash_before, 0.0)

        if freed_cash <= 0:
            return shares, cash, triggers

        # 2) Redeploy freed cash toward underweight target positions.
        values, _, portfolio_value = current_values(row, shares, cash)
        deficits = []
        total_deficit = 0.0

        for ticker, target_weight in self.target_weights.items():
            px = row.get(ticker, np.nan)

            if pd.isna(px) or px <= 0:
                continue

            target_value = portfolio_value * target_weight
            current_value = values.get(ticker, 0.0)
            deficit = max(target_value - current_value, 0.0)

            if deficit > self.execution.min_trade_dollars:
                deficits.append((ticker, deficit))
                total_deficit += deficit

        if total_deficit > 0:
            for ticker, deficit in deficits:
                px = row.get(ticker, np.nan)

                if pd.isna(px) or px <= 0:
                    continue

                budget = min(cash, freed_cash * (deficit / total_deficit))
                shares, cash, _ = self.execution.buy_with_budget(ticker, budget, px, shares, cash)

        return shares, cash, triggers

    def run(
        self,
        initial_capital: float,
        scheduled_frequency: str,
        hard_cap_check_frequency: str,
        stock_hard_trigger: float,
        stock_hard_trim_to: float,
        diversifier_hard_triggers: Dict[str, float],
        diversifier_hard_trim_to: Dict[str, float],
    ):
        scheduled_frequency = str(scheduled_frequency).upper()
        hard_cap_check_frequency = str(hard_cap_check_frequency).upper()

        scheduled_dates = set(get_rebalance_dates(self.prices.index, scheduled_frequency))
        if scheduled_frequency == "NONE":
            # Buy once at the start, while still allowing new tickers to be bought on their first valid date.
            scheduled_dates = {self.prices.index[0]}

        hard_check_dates = set(get_rebalance_dates(self.prices.index, hard_cap_check_frequency))

        # Invest new assets on their first valid date. This is important for ICOP / shorter-history assets.
        for ticker in self.tickers:
            if ticker in self.prices.columns:
                first_valid = self.prices[ticker].first_valid_index()
                if first_valid is not None:
                    scheduled_dates.add(first_valid)

        shares = {ticker: 0.0 for ticker in self.tickers}
        cash = float(initial_capital)

        equity_curve = []
        cash_curve = []
        holdings_records = []
        rebalance_records = []

        for date in self.prices.index:
            row = self.prices.loc[date]
            cash = self.apply_cash_return(cash)

            reason = None
            trigger_text = ""

            if date in scheduled_dates:
                shares, cash = self.full_rebalance(row, shares, cash)
                reason = "scheduled"

            elif date in hard_check_dates:
                shares, cash, triggers = self.hard_cap_trim(
                    row=row,
                    shares=shares,
                    cash=cash,
                    stock_hard_trigger=stock_hard_trigger,
                    stock_hard_trim_to=stock_hard_trim_to,
                    diversifier_hard_triggers=diversifier_hard_triggers,
                    diversifier_hard_trim_to=diversifier_hard_trim_to,
                )

                if triggers:
                    reason = f"hard_cap_{hard_cap_check_frequency.lower()}"
                    trigger_text = "; ".join(triggers)

            values, _, portfolio_value = current_values(row, shares, cash)

            if reason:
                rebalance_records.append(
                    {
                        "Date": date,
                        "Reason": reason,
                        "Portfolio Value": portfolio_value,
                        "Cash": cash,
                        "Triggers": trigger_text,
                    }
                )

            equity_curve.append(portfolio_value)
            cash_curve.append(cash)

            record = {
                "Date": date,
                "Portfolio Value": portfolio_value,
                "Cash": cash,
            }

            for ticker in self.tickers:
                record[f"{ticker}_Shares"] = shares.get(ticker, 0.0)
                record[f"{ticker}_Value"] = values.get(ticker, 0.0)

            holdings_records.append(record)

        equity = pd.Series(equity_curve, index=self.prices.index, name="Portfolio")
        cash_series = pd.Series(cash_curve, index=self.prices.index, name="Cash")
        holdings = pd.DataFrame(holdings_records).set_index("Date")
        log = pd.DataFrame(rebalance_records)

        return equity, cash_series, holdings, log
