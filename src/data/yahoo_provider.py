from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf


CANONICAL_MAP = {
    "GOOGL": "GOOG",
    "BRK.A": "BRK-B",
    "BRK-A": "BRK-B",
}


def normalize_ticker(ticker: str) -> str:
    t = str(ticker).strip().upper().replace(".", "-")
    return CANONICAL_MAP.get(t, t)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        x = normalize_ticker(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extract_close(price_raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(price_raw.columns, pd.MultiIndex):
        if "Close" in price_raw.columns.get_level_values(0):
            close = price_raw["Close"]
        elif "Adj Close" in price_raw.columns.get_level_values(0):
            close = price_raw["Adj Close"]
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")
    else:
        if "Close" in price_raw.columns:
            close = price_raw[["Close"]]
        elif "Adj Close" in price_raw.columns:
            close = price_raw[["Adj Close"]]
        else:
            raise ValueError("Could not find Close or Adj Close in yfinance output.")

    if isinstance(close, pd.Series):
        close = close.to_frame()

    close.columns = [normalize_ticker(c) for c in close.columns]
    close = close.loc[:, ~close.columns.duplicated()]
    return close


def download_prices(
    tickers: Iterable[str],
    start: str,
    end: Optional[str],
    return_frequency: str,
    cache_dir: str | Path = "data/cache",
    cache_prices: bool = True,
) -> pd.DataFrame:
    tickers = unique_preserve_order(tickers)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_name = f"prices_{start}_{end or 'latest'}_{return_frequency}.parquet".replace("/", "-")
    cache_path = cache_dir / cache_name

    if cache_prices and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        prices.index = pd.to_datetime(prices.index)
        existing = set(prices.columns)
        if set(tickers).issubset(existing):
            return prices[tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    close = extract_close(raw)
    weekly = close.resample(return_frequency).last().ffill().dropna(how="all")

    if cache_prices:
        weekly.to_parquet(cache_path)

    return weekly
