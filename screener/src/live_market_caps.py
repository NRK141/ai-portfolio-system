from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_STOCKANALYSIS_URL = "https://stockanalysis.com/list/biggest-companies/"


def normalize_ticker(ticker: str) -> str:
    if ticker is None:
        return ""

    t = str(ticker).strip().upper()
    t = t.replace(".", "-")

    if t == "BRK.B":
        return "BRK-B"

    return t


def parse_market_cap(value: Any) -> float | None:
    """
    Parse StockAnalysis market-cap strings like:
        5.11T, 985.37B, 800.2M

    Returns raw dollars.
    """
    if pd.isna(value):
        return None

    s = str(value).strip().replace(",", "").replace("$", "")
    if not s or s.upper() in {"-", "N/A", "NA", "NAN", "NONE"}:
        return None

    multiplier = 1.0
    suffix = s[-1].upper()

    if suffix == "T":
        multiplier = 1_000_000_000_000
        s = s[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000
        s = s[:-1]
    elif suffix == "M":
        multiplier = 1_000_000
        s = s[:-1]
    elif suffix == "K":
        multiplier = 1_000
        s = s[:-1]

    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([str(x).strip() for x in col if str(x).strip() and str(x).strip() != "nan"]).strip()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]

    return out


def _find_market_cap_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        t = _flatten_columns(table)
        cols = {str(c).strip() for c in t.columns}

        if {"Symbol", "Company Name", "Market Cap"}.issubset(cols):
            return t

    raise RuntimeError("Could not find StockAnalysis market-cap table with Symbol, Company Name, and Market Cap columns.")


def _read_cache(cache_path: str | Path, max_age_hours: float) -> pd.DataFrame | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    try:
        modified = pd.Timestamp(path.stat().st_mtime, unit="s")
        age_hours = (pd.Timestamp.now() - modified).total_seconds() / 3600.0

        if age_hours > float(max_age_hours):
            return None

        df = pd.read_csv(path)
        if df.empty:
            return None

        required = {"ticker", "market_cap", "market_cap_rank", "date"}
        if not required.issubset(df.columns):
            return None

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
        df["market_cap_rank"] = pd.to_numeric(df["market_cap_rank"], errors="coerce")
        df = df.dropna(subset=["ticker", "market_cap", "market_cap_rank"])

        if df.empty:
            return None

        df["rank_source"] = "LIVE_STOCKANALYSIS_CACHE"
        return df

    except Exception:
        return None


def fetch_stockanalysis_market_caps(
    limit: int = 250,
    url: str = DEFAULT_STOCKANALYSIS_URL,
    cache_path: str | Path | None = "screener/data/live_market_caps_stockanalysis.csv",
    cache_max_age_hours: float = 12,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch current U.S.-listed market-cap ranking from StockAnalysis.

    This is intended for the LIVE screener only, not historical backtests.
    The result is cached locally to avoid repeatedly hitting the site.
    """
    if cache_path and not force_refresh:
        cached = _read_cache(cache_path, cache_max_age_hours)
        if cached is not None and not cached.empty:
            return cached.head(limit).reset_index(drop=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    table = _find_market_cap_table(tables)

    df = table.rename(
        columns={
            "No.": "no",
            "Symbol": "ticker",
            "Company Name": "company_name",
            "Market Cap": "market_cap_raw",
            "Stock Price": "stock_price",
            "% Change": "pct_change",
            "Revenue": "revenue_raw",
        }
    )

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["market_cap"] = df["market_cap_raw"].map(parse_market_cap)

    df = df.dropna(subset=["ticker", "market_cap"])
    df = df[df["ticker"].astype(str).str.strip().ne("")]
    df = df.sort_values("market_cap", ascending=False).head(limit).reset_index(drop=True)

    asof = pd.Timestamp.today().normalize()

    df["market_cap_rank"] = range(1, len(df) + 1)
    df["date"] = asof
    df["rank_source"] = "LIVE_STOCKANALYSIS"
    df["source_url"] = url

    cols = [
        "date",
        "market_cap_rank",
        "ticker",
        "company_name",
        "market_cap",
        "market_cap_raw",
        "stock_price",
        "pct_change",
        "revenue_raw",
        "rank_source",
        "source_url",
    ]

    df = df[[c for c in cols if c in df.columns]]

    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return df
