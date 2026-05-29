from __future__ import annotations

import numpy as np


class ExecutionModel:
    def __init__(
        self,
        whole_shares: bool = True,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 2.0,
        min_trade_dollars: float = 25.0,
    ):
        self.whole_shares = whole_shares
        self.commission_per_trade = float(commission_per_trade)
        self.slippage_bps = float(slippage_bps)
        self.min_trade_dollars = float(min_trade_dollars)

    def buy_cost(self, qty: float, px: float) -> float:
        gross = qty * px
        slippage = gross * self.slippage_bps / 10000
        return gross + slippage + self.commission_per_trade

    def sell_proceeds(self, qty: float, px: float) -> float:
        gross = qty * px
        slippage = gross * self.slippage_bps / 10000
        return max(gross - slippage - self.commission_per_trade, 0.0)

    def quantize_buy_qty(self, raw_qty: float) -> float:
        return float(np.floor(raw_qty)) if self.whole_shares else float(raw_qty)

    def quantize_sell_qty(self, raw_qty: float) -> float:
        return float(np.floor(raw_qty)) if self.whole_shares else float(raw_qty)

    def sell_quantity(self, ticker: str, qty: float, px: float, shares: dict[str, float], cash: float):
        qty = min(qty, shares.get(ticker, 0.0))
        qty = self.quantize_sell_qty(qty)

        if qty <= 0:
            return shares, cash, 0.0

        trade_value = qty * px
        if trade_value < self.min_trade_dollars:
            return shares, cash, 0.0

        proceeds = self.sell_proceeds(qty, px)
        shares[ticker] -= qty
        cash += proceeds
        return shares, cash, proceeds

    def buy_with_budget(self, ticker: str, budget: float, px: float, shares: dict[str, float], cash: float):
        budget = min(float(budget), cash)

        if budget < self.min_trade_dollars:
            return shares, cash, 0.0

        approx_qty = (budget - self.commission_per_trade) / (px * (1 + self.slippage_bps / 10000))
        qty = self.quantize_buy_qty(approx_qty)

        if qty <= 0:
            return shares, cash, 0.0

        cost = self.buy_cost(qty, px)

        if cost > cash:
            approx_qty = (cash - self.commission_per_trade) / (px * (1 + self.slippage_bps / 10000))
            qty = self.quantize_buy_qty(max(approx_qty, 0.0))
            if qty <= 0:
                return shares, cash, 0.0
            cost = self.buy_cost(qty, px)

        if cost < self.min_trade_dollars:
            return shares, cash, 0.0

        shares[ticker] = shares.get(ticker, 0.0) + qty
        cash -= cost
        return shares, cash, cost
