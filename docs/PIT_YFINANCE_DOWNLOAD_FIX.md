# PIT Yahoo Download Fix

## Problem

The first point-in-time market-cap implementation attempted to download every ticker in
`point_in_time_market_caps.csv` from yfinance.

With WRDS data, that can be 10,000+ historical tickers, many of which are delisted or not
available in Yahoo Finance.

This caused:

- thousands of failed yfinance downloads
- empty price matrices in some runs
- `KeyError: "None of ['Date'] are in the columns"`

## Fix

The backtester now:

1. Loads the full PIT market-cap file locally.
2. Builds annual ranking dates first.
3. Selects only the top N market-cap candidates per ranking date.
4. Adds anchors, diversifiers, and SPY.
5. Downloads only that reduced ticker list from yfinance.

Config option:

```yaml
point_in_time_download_top_n: 100
```

This keeps the Yahoo download list small while still allowing enough fallback candidates
if some historical tickers are unavailable.
