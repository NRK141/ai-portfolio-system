from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from src.data.yahoo_provider import normalize_ticker, unique_preserve_order


def scrape_current_top_market_caps(n: int = 50) -> list[str]:
    url = "https://stockanalysis.com/list/biggest-companies/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("No tables found when scraping current market caps.")

    df = tables[0]
    df.columns = [str(c).strip().lower() for c in df.columns]

    ticker_col = None
    for col in df.columns:
        if "ticker" in col or "symbol" in col:
            ticker_col = col
            break

    if ticker_col is None:
        raise RuntimeError(f"No ticker/symbol column found. Columns: {df.columns.tolist()}")

    tickers = df[ticker_col].astype(str).map(normalize_ticker).tolist()
    tickers = unique_preserve_order(tickers)

    return tickers[:n]


def load_point_in_time_market_caps(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Point-in-time market cap file not found: {path}. "
            "Expected columns: date,ticker,market_cap"
        )

    df = pd.read_csv(path)
    needed = {"date", "ticker", "market_cap"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in market cap file: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "market_cap"])
    return df


def build_static_equity_universe(
    anchors: list[str],
    fallback_mega_caps: list[str],
    available_cols: Iterable[str],
    total_positions: int,
    use_web_scraper: bool,
    scrape_top_n: int,
) -> list[str]:
    anchors = [normalize_ticker(t) for t in anchors]
    available = set(normalize_ticker(t) for t in available_cols)

    candidates: list[str] = []
    if use_web_scraper:
        try:
            candidates = scrape_current_top_market_caps(scrape_top_n)
            print(f"Scraped {len(candidates)} current mega-cap candidates.")
        except Exception as e:
            print("WARNING: Web scrape failed. Using fallback list.")
            print(f"Scrape error: {e}")

    candidates = unique_preserve_order(anchors + candidates + fallback_mega_caps)
    candidates = [t for t in candidates if t in available]

    missing_anchors = [t for t in anchors if t not in available]
    if missing_anchors:
        raise ValueError(f"Missing anchor price data: {missing_anchors}")

    universe = []
    for t in anchors:
        if t not in universe:
            universe.append(t)

    for t in candidates:
        if t not in universe:
            universe.append(t)
        if len(universe) >= total_positions:
            break

    if len(universe) < total_positions:
        raise ValueError(f"Only built {len(universe)} equity positions. Need {total_positions}.")

    return universe
