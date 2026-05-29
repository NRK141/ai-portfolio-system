# Future CRSP Return Mode

Current mode:

```text
WRDS/CRSP point-in-time market caps
+
yfinance adjusted prices
```

This is much better than using current mega-cap tickers backward, but it still has limitations:

- yfinance may not have old/delisted tickers.
- historical ticker mapping may be imperfect.
- delisting returns are not included.
- Yahoo adjusted price histories may differ from CRSP.

Future mode should use:

```text
CRSP market caps
+
CRSP daily/monthly returns
+
CRSP delisting returns
+
PERMNO/PERMCO identifiers
```

Potential design:

```yaml
data:
  price_source: "crsp"
  crsp_returns_path: "data/raw/wrds_crsp_daily_returns.csv"
  crsp_delisting_path: "data/raw/wrds_crsp_delisting_information.csv"
```

Do not implement this until the PIT ranking/yfinance hybrid mode is stable.
