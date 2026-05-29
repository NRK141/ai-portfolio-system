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
        target_weights_by_date: Optional[dict[pd.Timestamp, Dict[str, float]]] = None,
    ):
        self.prices = prices.dropna(how="all").copy()
        self.target_weights = target_weights
        self.target_weights_by_date = None
        if target_weights_by_date:
            self.target_weights_by_date = {
                pd.Timestamp(k): dict(v)
                for k, v in sorted(target_weights_by_date.items(), key=lambda kv: pd.Timestamp(kv[0]))
            }
        self.tickers = self._build_ticker_union()
        self.equity_universe = set(equity_universe)
        self.diversifiers = diversifiers
        self.execution = execution
        self.cash_return_annual = cash_return_annual
        self.periods_per_year = periods_per_year

    def _build_ticker_union(self) -> list[str]:
        seen = set()
        out = []
        for t in self.target_weights:
            if t not in seen:
                out.append(t)
                seen.add(t)
        if self.target_weights_by_date:
            for _, weights in self.target_weights_by_date.items():
                for t in weights:
                    if t not in seen:
                        out.append(t)
                        seen.add(t)
        return out

    def get_target_weights_for_date(self, date: pd.Timestamp) -> Dict[str, float]:
        if not self.target_weights_by_date:
            return self.target_weights
        date = pd.Timestamp(date)
        valid_dates = [d for d in self.target_weights_by_date if d <= date]
        if not valid_dates:
            return self.target_weights_by_date[min(self.target_weights_by_date)]
        return self.target_weights_by_date[max(valid_dates)]

    def scheduled_dates_from_dynamic_targets(self) -> set[pd.Timestamp]:
        if not self.target_weights_by_date:
            return set()
        return set(pd.Timestamp(d) for d in self.target_weights_by_date.keys())

    def apply_cash_return(self, cash: float) -> float:
        if self.cash_return_annual == 0:
            return cash
        rate = (1 + self.cash_return_annual) ** (1 / self.periods_per_year) - 1
        return cash * (1 + rate)

    def full_rebalance(self, row: pd.Series, shares: dict[str, float], cash: float, date: pd.Timestamp):
        shares = shares.copy()
        active_weights = self.get_target_weights_for_date(date)
        _, _, pv = current_values(row, shares, cash)
        if pv <= 0:
            return shares, cash

        # Sell overweights and sell rotated-out names to zero.
        for t in self.tickers:
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue
            target_val = pv * active_weights.get(t, 0.0)
            current_val = shares.get(t, 0.0) * px
            if current_val > target_val:
                desired_qty = target_val / px if target_val > 0 else 0.0
                sell_qty = shares.get(t, 0.0) - desired_qty
                shares, cash, _ = self.execution.sell_quantity(t, sell_qty, px, shares, cash)

        # Recompute and buy active underweights.
        _, _, pv = current_values(row, shares, cash)
        active_weights = self.get_target_weights_for_date(date)
        deficits = []
        for t, target_w in active_weights.items():
            px = row.get(t, np.nan)
            if pd.isna(px) or px <= 0:
                continue
            current_val = shares.get(t, 0.0) * px
            target_val = pv * target_w
            deficit = target_val - current_val
            if deficit > self.execution.min_trade_dollars:
                deficits.append((t, deficit))

        for t, deficit in sorted(deficits, key=lambda x: x[1], reverse=True):
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
        date: pd.Timestamp,
        stock_hard_trigger: float,
        stock_hard_trim_to: float,
        diversifier_hard_triggers: Dict[str, float],
        diversifier_hard_trim_to: Dict[str, float],
    ):
        shares = shares.copy()
        active_weights = self.get_target_weights_for_date(date)
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

            if t in self.diversifiers:
                trigger = diversifier_hard_triggers.get(t)
                trim_to = diversifier_hard_trim_to.get(t)
            elif t in self.equity_universe or (t in active_weights and t not in self.diversifiers):
                trigger = stock_hard_trigger
                trim_to = stock_hard_trim_to

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

        values, _, pv = current_values(row, shares, cash)
        active_weights = self.get_target_weights_for_date(date)
        deficits = []
        total_deficit = 0.0

        for t, target_w in active_weights.items():
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
        hard_cap_check_frequency: str,
        stock_hard_trigger: float,
        stock_hard_trim_to: float,
        diversifier_hard_triggers: Dict[str, float],
        diversifier_hard_trim_to: Dict[str, float],
    ):
        scheduled_dates = set(get_rebalance_dates(self.prices.index, scheduled_frequency))
        scheduled_dates.update(self.scheduled_dates_from_dynamic_targets())

        hard_freq = str(hard_cap_check_frequency).upper()
        hard_dates = set() if hard_freq == "NONE" else set(get_rebalance_dates(self.prices.index, hard_freq))

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
            active_weights = self.get_target_weights_for_date(date)

            if date in scheduled_dates:
                shares, cash = self.full_rebalance(row, shares, cash, date)
                reason = "scheduled"
            elif date in hard_dates:
                shares, cash, triggers = self.hard_cap_trim(
                    row, shares, cash, date,
                    stock_hard_trigger, stock_hard_trim_to,
                    diversifier_hard_triggers, diversifier_hard_trim_to,
                )
                if triggers:
                    reason = f"{hard_freq.lower()}_hard_cap"
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
                        "Active Target Count": len(active_weights),
                    }
                )

            equity_curve.append(pv)
            cash_curve.append(cash)

            record = {"Date": date, "Portfolio Value": pv, "Cash": cash}
            for t in self.tickers:
                record[f"{t}_Shares"] = shares.get(t, 0.0)
                record[f"{t}_Value"] = values.get(t, 0.0)
                record[f"{t}_TargetWeight"] = active_weights.get(t, 0.0)
            holdings_records.append(record)

        equity = pd.Series(equity_curve, index=self.prices.index, name="Portfolio")
        cash_series = pd.Series(cash_curve, index=self.prices.index, name="Cash")
        holdings = pd.DataFrame(holdings_records).set_index("Date")
        log = pd.DataFrame(rebalance_records)
        return equity, cash_series, holdings, log
