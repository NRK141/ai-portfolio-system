import pandas as pd

from src.engine.backtest import get_rebalance_dates


def test_get_rebalance_dates_weekly_returns_all_dates():
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    out = get_rebalance_dates(idx, "W")
    assert list(out) == list(idx)


def test_get_rebalance_dates_monthly_returns_first_date_per_month():
    idx = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-02-02", "2024-02-09"])
    out = get_rebalance_dates(idx, "M")
    assert out == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-02-02")]


def test_get_rebalance_dates_none_returns_empty_list():
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    assert get_rebalance_dates(idx, "NONE") == []
