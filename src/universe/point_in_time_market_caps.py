from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.yahoo_provider import normalize_ticker, unique_preserve_order
from src.engine.backtest import get_rebalance_dates


REQUIRED_COLUMNS = {"date", "ticker", "market_cap"}


YFINANCE_TICKER_MAP = {
    # CRSP historical ticker -> Yahoo/current ticker.
    "FB": "META",
    "BRK": "BRK-B",
    "BRK.B": "BRK-B",
    "BRK-B": "BRK-B",
    "BF.B": "BF-B",
    "BF-B": "BF-B",
}


def to_yfinance_ticker(ticker: str) -> str:
    t = normalize_ticker(ticker)
    return YFINANCE_TICKER_MAP.get(t, t)


def normalize_frequency_list(freqs) -> list[str]:
    if freqs is None:
        return []
    if isinstance(freqs, str):
        return [freqs.upper()]
    return [str(f).upper() for f in freqs]


def load_point_in_time_market_caps(path: str | Path) -> pd.DataFrame:
    """
    Load a user-supplied point-in-time market-cap file.

    Required columns:
        date,ticker,market_cap

    The file can be CSV or Parquet. Tickers are normalized to yfinance format.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Point-in-time market-cap file not found: {path}\n"
            "Expected columns: date,ticker,market_cap"
        )

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Point-in-time market-cap file is missing columns: {sorted(missing)}. "
            "Expected at least: date,ticker,market_cap"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(normalize_ticker).map(to_yfinance_ticker)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "market_cap"])
    df = df[df["ticker"].astype(str).str.strip().ne("")]
    df = df[df["ticker"].astype(str).str.upper().ne("N/A")]
    df = df[df["market_cap"] > 0]
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def latest_market_caps_asof(
    market_caps: pd.DataFrame,
    asof_date: pd.Timestamp,
    min_market_cap: float = 0.0,
) -> pd.DataFrame:
    """
    For each ticker, use the latest market-cap observation known on or before asof_date.
    """
    asof_date = pd.Timestamp(asof_date)
    subset = market_caps.loc[market_caps["date"] <= asof_date].copy()

    if min_market_cap and min_market_cap > 0:
        subset = subset.loc[subset["market_cap"] >= float(min_market_cap)]

    if subset.empty:
        return pd.DataFrame(columns=list(market_caps.columns) + ["rank_date"])

    subset = subset.sort_values(["ticker", "date"])
    latest = subset.groupby("ticker", as_index=False).tail(1)
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
    """
    Build a dynamic point-in-time universe for each ranking/rotation date.

    The strategy rotates tickers whenever the selected ranking_frequency produces
    a new ranking date. Example: M rotates monthly, Q rotates quarterly, SA
    rotates semiannually, Y rotates annually.
    """
    idx = pd.DatetimeIndex(prices_index)
    ranking_dates = get_rebalance_dates(idx, ranking_frequency)

    anchors = [to_yfinance_ticker(t) for t in anchors]
    available = set(to_yfinance_ticker(t) for t in available_cols)

    missing_anchors = [t for t in anchors if t not in available]
    if missing_anchors:
        raise ValueError(f"Missing anchor price data: {missing_anchors}")

    rows = []
    universe_by_date: dict[pd.Timestamp, list[str]] = {}

    for ranking_date in ranking_dates:
        ranking_date = pd.Timestamp(ranking_date)
        asof_date = ranking_date - pd.Timedelta(days=int(asof_lag_days))

        latest = latest_market_caps_asof(
            market_caps=market_caps,
            asof_date=asof_date,
            min_market_cap=float(min_market_cap),
        )

        latest = latest.loc[latest["ticker"].isin(available)].copy()

        if latest.empty:
            raise ValueError(
                f"No point-in-time market-cap data available as of {asof_date.date()} "
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
                f"Need {total_positions}. Increase point_in_time_download_top_n."
            )

        universe_by_date[ranking_date] = universe

        rank_map = latest.set_index("ticker")["rank"].to_dict()
        cap_map = latest.set_index("ticker")["market_cap"].to_dict()
        cap_date_map = latest.set_index("ticker")["date"].to_dict()

        for slot, ticker in enumerate(universe, start=1):
            rows.append(
                {
                    "ranking_frequency": str(ranking_frequency).upper(),
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

    membership = pd.DataFrame(rows)
    return universe_by_date, membership


def build_pit_candidate_universe(
    prices_index: Iterable[pd.Timestamp],
    market_caps: pd.DataFrame,
    anchors: list[str],
    diversifiers: list[str],
    benchmark: str,
    ranking_frequencies,
    asof_lag_days: int = 0,
    min_market_cap: float = 0.0,
    top_n_per_ranking_date: int = 100,
) -> tuple[list[str], pd.DataFrame]:
    """
    Build a SMALL yfinance download universe before downloading prices.

    Do NOT download all WRDS tickers from Yahoo. This function downloads only:
    anchors + diversifiers + benchmark + top N names for each ranking date/frequency.
    """
    idx = pd.DatetimeIndex(prices_index)
    freqs = normalize_frequency_list(ranking_frequencies)

    anchors = [to_yfinance_ticker(t) for t in anchors]
    diversifiers = [to_yfinance_ticker(t) for t in diversifiers]
    benchmark = to_yfinance_ticker(benchmark)

    rows = []

    for freq in freqs:
        ranking_dates = get_rebalance_dates(idx, freq)

        for ranking_date in ranking_dates:
            ranking_date = pd.Timestamp(ranking_date)
            asof_date = ranking_date - pd.Timedelta(days=int(asof_lag_days))

            latest = latest_market_caps_asof(
                market_caps=market_caps,
                asof_date=asof_date,
                min_market_cap=float(min_market_cap),
            )

            if latest.empty:
                continue

            latest = latest.head(int(top_n_per_ranking_date)).copy()
            latest["ranking_frequency"] = freq
            latest["ranking_date"] = ranking_date
            latest["asof_date"] = asof_date
            rows.append(latest)

    if rows:
        candidates = pd.concat(rows, ignore_index=True)
    else:
        candidates = pd.DataFrame(columns=["ranking_frequency", "ranking_date", "asof_date", "ticker", "market_cap", "rank"])

    pit_tickers = candidates["ticker"].dropna().astype(str).tolist() if not candidates.empty else []
    tickers = unique_preserve_order(anchors + diversifiers + [benchmark] + pit_tickers)

    return tickers, candidates


def summarize_universe_membership(universe_membership: pd.DataFrame) -> pd.DataFrame:
    if universe_membership is None or universe_membership.empty:
        return pd.DataFrame()

    df = universe_membership.copy()
    rows = []

    for freq, group in df.groupby("ranking_frequency", dropna=False):
        rows.append(
            {
                "Ranking Frequency": freq,
                "Ranking Dates": group["ranking_date"].nunique(),
                "Rows": len(group),
                "Unique Tickers": group["ticker"].nunique(),
                "First Ranking Date": group["ranking_date"].min(),
                "Last Ranking Date": group["ranking_date"].max(),
                "Anchor Rows": int(group["is_anchor"].sum()),
            }
        )

    return pd.DataFrame(rows)
