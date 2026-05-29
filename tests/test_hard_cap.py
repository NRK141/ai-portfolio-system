import pandas as pd

from src.engine.backtest import BacktestEngine
from src.engine.execution import ExecutionModel


def test_weekly_hard_cap_trims_breach_after_initial_rebalance():
    prices = pd.DataFrame(
        {
            "A": [100.0, 400.0],
            "B": [100.0, 100.0],
        },
        index=pd.to_datetime(["2024-01-05", "2024-01-12"]),
    )
    target_weights = {"A": 0.5, "B": 0.5}
    execution = ExecutionModel(whole_shares=False, commission_per_trade=0.0, slippage_bps=0.0, min_trade_dollars=0.0)

    engine = BacktestEngine(
        prices=prices,
        target_weights=target_weights,
        equity_universe=["A", "B"],
        diversifiers={},
        execution=execution,
    )

    equity, cash, holdings, log = engine.run(
        initial_capital=1000.0,
        scheduled_frequency="NONE",
        hard_cap_check_frequency="W",
        stock_hard_trigger=0.60,
        stock_hard_trim_to=0.50,
        diversifier_hard_triggers={},
        diversifier_hard_trim_to={},
    )

    assert not log.empty
    assert log["Reason"].astype(str).str.contains("hard_cap").any()

    final_a_value = holdings.iloc[-1]["A_Value"]
    final_portfolio_value = equity.iloc[-1]
    final_weight = final_a_value / final_portfolio_value

    # With fractional shares and zero slippage, the trim should land very close to 50%.
    assert final_weight <= 0.51
