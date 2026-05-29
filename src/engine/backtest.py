from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.engine.execution import ExecutionModel


def get_rebalance_dates(index: pd.Index, freq: str) -> list[pd.Timestamp]:
    freq = str(freq).upper()
    idx = pd.DatetimeIndex(index)

    if freq == "NONE":
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
    for t, sh in shares.items():
        px = row.get(t, np.nan)
        val = sh * px if pd.notna(px) else 0.0
        values[t] = val
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
        self.cash_return_annual = cash_return_annual
        self.periods_per_year = periods_per_year

    def apply_cash_return(self, cash: float) -> float:
        if self.cash_return_annual == 0:
            return cash
        rate = (1 + self.cash_return_annual) ** (1 / self.periods_per_year) - 1
        return cash * (1 + rate)

    def full_rebalance(self, row: pd.Series, shares: dict[str, float], cash: float):
        shares = shares.copy()
        _, _, pv = current_values(row, shares, cash)
        if pv <= 0:
            return shares, cash

        target_values = {t: pv * w for t, w in self.target_weights.items()}

        # Sell overweights first.
        for t, target_val in target_values.items():
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue

            current_val = shares.get(t, 0.0) * px
            if current_val > target_val:
                sell_qty = shares.get(t, 0.0) - (target_val / px)
                shares, cash, _ = self.execution.sell_quantity(t, sell_qty, px, shares, cash)

        # Recompute and buy underweights.
        _, _, pv = current_values(row, shares, cash)
        target_values = {t: pv * w for t, w in self.target_weights.items()}
        deficits = []

        for t, target_val in target_values.items():
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue

            current_val = shares.get(t, 0.0) * px
            deficit = target_val - current_val
            if deficit > self.execution.min_trade_dollars:
                deficits.append((t, deficit))

        deficits = sorted(deficits, key=lambda x: x[1], reverse=True)
        for t, deficit in deficits:
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue
            shares, cash, _ = self.execution.buy_with_budget(t, deficit, px, shares, cash)

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
        shares = shares.copy()
        values, _, pv = current_values(row, shares, cash)
        if pv <= 0:
            return shares, cash, []

        triggers = []
        cash_before = cash

        for t, val in values.items():
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue

            weight = val / pv
            trigger = None
            trim_to = None

            if t in self.equity_universe:
                trigger = stock_hard_trigger
                trim_to = stock_hard_trim_to
            elif t in self.diversifiers:
                trigger = diversifier_hard_triggers.get(t)
                trim_to = diversifier_hard_trim_to.get(t)

            if trigger is None or trim_to is None:
                continue

            if weight > trigger:
                trim_value = pv * trim_to
                if val > trim_value:
                    sell_value = val - trim_value
                    sell_qty = sell_value / px
                    shares, cash, _ = self.execution.sell_quantity(t, sell_qty, px, shares, cash)
                    triggers.append(f"{t}: {weight:.2%} > {trigger:.2%}, trim to {trim_to:.2%}")

        freed_cash = max(cash - cash_before, 0.0)
        if freed_cash <= 0:
            return shares, cash, triggers

        # Redeploy freed cash to underweights, proportional to deficits.
        values, _, pv = current_values(row, shares, cash)
        deficits = []
        total_deficit = 0.0

        for t, target_w in self.target_weights.items():
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue

            target_val = pv * target_w
            current_val = values.get(t, 0.0)
            deficit = max(target_val - current_val, 0.0)

            if deficit > self.execution.min_trade_dollars:
                deficits.append((t, deficit))
                total_deficit += deficit

        if total_deficit > 0:
            for t, deficit in deficits:
                px = row.get(t, np.nan)
                if pd.isna(px) or px <= 0:
                    continue
                budget = min(cash, freed_cash * (deficit / total_deficit))
                shares, cash, _ = self.execution.buy_with_budget(t, budget, px, shares, cash)

        return shares, cash, triggers

    def run(
        self,
        initial_capital: float,
        scheduled_frequency: str,
        monthly_hard_single_name_check: bool,
        stock_hard_trigger: float,
        stock_hard_trim_to: float,
        diversifier_hard_triggers: Dict[str, float],
        diversifier_hard_trim_to: Dict[str, float],
    ):
        scheduled_dates = set(get_rebalance_dates(self.prices.index, scheduled_frequency))
        hard_check_dates = set(get_rebalance_dates(self.prices.index, "M")) if monthly_hard_single_name_check else set()

        # Invest new assets on their first valid date.
        for t in self.tickers:
            if t in self.prices.columns:
                first_valid = self.prices[t].first_valid_index()
                if first_valid is not None:
                    scheduled_dates.add(first_valid)

        shares = {t: 0.0 for t in self.tickers}
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
                    row,
                    shares,
                    cash,
                    stock_hard_trigger,
                    stock_hard_trim_to,
                    diversifier_hard_triggers,
                    diversifier_hard_trim_to,
                )
                if triggers:
                    reason = "monthly_hard_cap"
                    trigger_text = "; ".join(triggers)

            values, _, pv = current_values(row, shares, cash)

            if reason:
                rebalance_records.append(
                    {
                        "Date": date,
                        "Reason": reason,
                        "Portfolio Value": pv,
                        "Cash": cash,
                        "Triggers": trigger_text,
                    }
                )

            equity_curve.append(pv)
            cash_curve.append(cash)

            record = {"Date": date, "Portfolio Value": pv, "Cash": cash}
            for t in self.tickers:
                record[f"{t}_Shares"] = shares.get(t, 0.0)
                record[f"{t}_Value"] = values.get(t, 0.0)
            holdings_records.append(record)

        equity = pd.Series(equity_curve, index=self.prices.index, name="Portfolio")
        cash_series = pd.Series(cash_curve, index=self.prices.index, name="Cash")
        holdings = pd.DataFrame(holdings_records).set_index("Date")
        log = pd.DataFrame(rebalance_records)

        return equity, cash_series, holdings, log
