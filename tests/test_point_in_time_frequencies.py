import pandas as pd

from src.universe.point_in_time_market_caps import build_point_in_time_universe_by_date


def test_pit_monthly_has_more_ranking_dates_than_annual():
    idx = pd.date_range("2020-01-03", "2021-12-31", freq="W-FRI")

    rows = []
    for d in ["2019-12-31", "2020-12-31", "2021-12-31"]:
        for ticker, cap in [("AAA", 500), ("BBB", 400), ("CCC", 300), ("DDD", 200), ("EEE", 100), ("ANCHOR", 10)]:
            rows.append({"date": d, "ticker": ticker, "market_cap": cap})

    market_caps = pd.DataFrame(rows)
    market_caps["date"] = pd.to_datetime(market_caps["date"])

    monthly_universe, monthly_membership = build_point_in_time_universe_by_date(
        prices_index=idx,
        market_caps=market_caps,
        anchors=["ANCHOR"],
        total_positions=3,
        available_cols=["AAA", "BBB", "CCC", "DDD", "EEE", "ANCHOR"],
        ranking_frequency="M",
    )

    annual_universe, annual_membership = build_point_in_time_universe_by_date(
        prices_index=idx,
        market_caps=market_caps,
        anchors=["ANCHOR"],
        total_positions=3,
        available_cols=["AAA", "BBB", "CCC", "DDD", "EEE", "ANCHOR"],
        ranking_frequency="Y",
    )

    assert len(monthly_universe) > len(annual_universe)
    assert monthly_membership["ranking_frequency"].iloc[0] == "M"
    assert annual_membership["ranking_frequency"].iloc[0] == "Y"
