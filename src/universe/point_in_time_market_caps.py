from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.yahoo_provider import normalize_ticker, unique_preserve_order
from src.engine.backtest import get_rebalance_dates

REQUIRED_COLUMNS = {"date", "ticker", "market_cap"}


def load_point_in_time_market_caps(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied point-in-time market-cap file.

    Required columns: date,ticker,market_cap
    CSV and Parquet are supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Point-in-time market-cap file not found: {path}. "
            "Expected columns: date,ticker,market_cap"
        )

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing PIT market-cap columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "market_cap"])
    df = df[df["market_cap"] > 0]
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def latest_market_caps_asof(
    market_caps: pd.DataFrame,
    asof_date: pd.Timestamp,
    min_market_cap: float = 0.0,
) -> pd.DataFrame:
    """For each ticker, use latest market-cap observation on or before asof_date."""
    asof_date = pd.Timestamp(asof_date)
    subset = market_caps.loc[market_caps["date"] <= asof_date].copy()

    if min_market_cap and min_market_cap > 0:
        subset = subset.loc[subset["market_cap"] >= float(min_market_cap)]

    if subset.empty:
        return pd.DataFrame(columns=list(market_caps.columns) + ["rank_date", "rank"])

    latest = subset.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest = latest.sort_values("market_cap", ascending=False).reset_index(drop=True)
    latest["rank"] = latest.index + 1
    latest["rank_date"] = asof_date
    return latest


def build_point_in_time_universe_by_date(
    prices_index: Iterable[pd.Timestamp],
    market_caps: pd.DataFrame,
    anchors: list[str],
    total_positions: int,
    available_cols: Iterable[str],
    ranking_frequency: str = "Y",
    asof_lag_days: int = 0,
    min_market_cap: float = 0.0,
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    """Build historical top-market-cap universes without looking forward.

    On each ranking date:
    - Rank using market caps dated <= ranking_date - asof_lag_days.
    - Force anchors into the portfolio.
    - Fill remaining slots with top ranked names available in price data.
    """
    idx = pd.DatetimeIndex(prices_index)
    ranking_dates = get_rebalance_dates(idx, ranking_frequency)

    anchors = [normalize_ticker(t) for t in anchors]
    available = set(normalize_ticker(t) for t in available_cols)

    missing_anchors = [t for t in anchors if t not in available]
    if missing_anchors:
        raise ValueError(f"Missing anchor price data: {missing_anchors}")

    rows = []
    universe_by_date: dict[pd.Timestamp, list[str]] = {}

    for ranking_date in ranking_dates:
        ranking_date = pd.Timestamp(ranking_date)
        asof_date = ranking_date - pd.Timedelta(days=int(asof_lag_days))

        latest = latest_market_caps_asof(market_caps, asof_date, float(min_market_cap))
        latest = latest.loc[latest["ticker"].isin(available)].copy()

        if latest.empty:
            raise ValueError(
                f"No PIT market-cap data available as of {asof_date.date()} "
                f"for ranking date {ranking_date.date()}."
            )

        ranked = latest["ticker"].tolist()
        universe = []

        for t in anchors:
            if t not in universe:
                universe.append(t)

        for t in ranked:
            if t not in universe:
                universe.append(t)
            if len(universe) >= total_positions:
                break

        if len(universe) < total_positions:
            raise ValueError(
                f"Only built {len(universe)} names for {ranking_date.date()}. "
                f"Need {total_positions}."
            )

        universe_by_date[ranking_date] = universe

        rank_map = latest.set_index("ticker")["rank"].to_dict()
        cap_map = latest.set_index("ticker")["market_cap"].to_dict()
        cap_date_map = latest.set_index("ticker")["date"].to_dict()

        for slot, ticker in enumerate(universe, start=1):
            rows.append(
                {
                    "ranking_date": ranking_date,
                    "asof_date": asof_date,
                    "ticker": ticker,
                    "slot": slot,
                    "is_anchor": ticker in anchors,
                    "market_cap_rank": rank_map.get(ticker),
                    "market_cap": cap_map.get(ticker),
                    "market_cap_observation_date": cap_date_map.get(ticker),
                }
            )

    return universe_by_date, pd.DataFrame(rows)


def unique_tickers_from_market_cap_file(path: str | Path) -> list[str]:
    df = load_point_in_time_market_caps(path)
    return unique_preserve_order(df["ticker"].dropna().astype(str).tolist())
