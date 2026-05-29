import pandas as pd

from src.universe.point_in_time_market_caps import build_point_in_time_universe_by_date


def test_point_in_time_universe_forces_anchors_and_fills_by_rank():
    idx = pd.date_range("2020-01-03", periods=110, freq="W-FRI")
    market_caps = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5,
            "ticker": ["AAA", "BBB", "CCC", "DDD", "ANCHOR"],
            "market_cap": [500, 400, 300, 200, 10],
        }
    )
    market_caps["date"] = pd.to_datetime(market_caps["date"])

    universe_by_date, membership = build_point_in_time_universe_by_date(
        prices_index=idx,
        market_caps=market_caps,
        anchors=["ANCHOR"],
        total_positions=3,
        available_cols=["AAA", "BBB", "CCC", "DDD", "ANCHOR"],
        ranking_frequency="Y",
        asof_lag_days=0,
        min_market_cap=0,
    )
    first_universe = next(iter(universe_by_date.values()))
    assert first_universe[0] == "ANCHOR"
    assert first_universe[1:] == ["AAA", "BBB"]
    assert len(first_universe) == 3
    assert not membership.empty


def test_point_in_time_universe_uses_asof_not_future_data():
    idx = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    market_caps = pd.DataFrame(
        {
            "date": ["2019-12-31", "2020-01-10"],
            "ticker": ["OLD", "FUTURE"],
            "market_cap": [100, 999],
        }
    )
    market_caps["date"] = pd.to_datetime(market_caps["date"])

    universe_by_date, _ = build_point_in_time_universe_by_date(
        prices_index=idx,
        market_caps=market_caps,
        anchors=[],
        total_positions=1,
        available_cols=["OLD", "FUTURE"],
        ranking_frequency="Y",
        asof_lag_days=0,
        min_market_cap=0,
    )
    first_universe = next(iter(universe_by_date.values()))
    assert first_universe == ["OLD"]
