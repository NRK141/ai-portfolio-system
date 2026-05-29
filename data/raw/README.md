# Raw Data

This folder is for small manual input datasets needed by the backtester.

## Point-in-time market caps

To enable point-in-time market-cap universe rotation, create:

```text
data/raw/point_in_time_market_caps.csv
```

Required columns:

```text
date,ticker,market_cap
```

Example:

```csv
date,ticker,market_cap
2015-01-01,AAPL,643000000000
2015-01-01,MSFT,381000000000
2015-01-01,XOM,382000000000
2016-01-01,AAPL,586000000000
2016-01-01,MSFT,443000000000
2016-01-01,XOM,314000000000
```

Then set this in `configs/strategy_refined_free.yaml`:

```yaml
universe:
  use_point_in_time_market_caps: true
  point_in_time_market_caps_path: "data/raw/point_in_time_market_caps.csv"
  point_in_time_ranking_frequency: "Y"
  point_in_time_asof_lag_days: 0
```

Downloaded market data and price cache files should not be committed to GitHub.
