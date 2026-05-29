# Point-in-Time Market-Cap Universe Support

This upgrade adds support for point-in-time historical market-cap rankings.

## Why this matters

The default free-data backtest uses current mega-cap candidates and tests them backward. That can introduce survivorship/look-ahead bias.

Point-in-time market-cap data improves the model because the universe is selected using only market-cap information available at each ranking date.

## Required file

Create:

```text
data/raw/point_in_time_market_caps.csv
```

Required columns:

```text
date,ticker,market_cap
```

Optional columns:

```text
company_name,exchange,shares_outstanding,price,source
```

## Config

In `configs/strategy_refined_free.yaml`:

```yaml
universe:
  use_point_in_time_market_caps: true
  point_in_time_market_caps_path: "data/raw/point_in_time_market_caps.csv"
  point_in_time_ranking_frequency: "Y"
  point_in_time_asof_lag_days: 0
```

## Universe logic

On each ranking date:

1. The engine looks at market-cap observations dated on or before the as-of date.
2. Anchors are forced into the portfolio.
3. Remaining slots are filled with the highest-market-cap names.
4. The universe membership is saved to:

```text
outputs/csv/universe_membership.csv
```

## Free-data limitation

This upgrade supports point-in-time data, but it does not magically create historical market-cap data.

Free options:
- manually curate yearly market-cap snapshots
- use public datasets if licensing permits
- export market-cap snapshots from academic or school resources if available

Paid/institutional options later:
- CRSP / Compustat via WRDS
- FactSet / Bloomberg / Refinitiv / Capital IQ
- Financial Modeling Prep historical market cap endpoint
