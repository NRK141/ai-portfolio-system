# Full AI-Anchor Q Screener

This is the live screener for the forward-looking portfolio.

## Strategy

- Full AI anchors:
  - ASML
  - MU
  - TSM
  - NVDA
  - GOOG
  - META

- Quarterly market-cap leadership rotation:
  - Ranking frequency: `Q`
  - Total equity positions: 15
  - Equity sleeve: 60%
  - Each equity position target: 4%

- Diversifier sleeve:
  - IAU: 10%
  - SLV: 7.5%
  - ICOP: 7.5%
  - VXUS: 15%

## Ranking source priority

The live screener uses:

1. StockAnalysis live market-cap rankings
2. Local WRDS PIT market-cap rankings if live scraping fails
3. Static fallback list if WRDS is unavailable

Live rankings are for current allocation decisions only. Use WRDS PIT data for historical backtests.

## Install

```powershell
pip install -r screener/requirements-screener.txt
```

## Run with current holdings

```powershell
python screener/ai_portfolio_screener.py --holdings screener/data/current_holdings.csv --cash 0
```

## Force refresh live rankings

```powershell
python screener/ai_portfolio_screener.py --holdings screener/data/current_holdings.csv --cash 0 --refresh-live-ranks
```

## Disable live rankings and use WRDS PIT/fallback

```powershell
python screener/ai_portfolio_screener.py --holdings screener/data/current_holdings.csv --cash 0 --no-live-ranks
```

## Fresh allocation by capital

```powershell
python screener/ai_portfolio_screener.py --capital 25000 --refresh-live-ranks
```

## Outputs

```text
outputs/screener/screener_report.md
outputs/screener/target_allocation.csv
outputs/screener/trade_plan.csv
outputs/screener/alerts.csv
outputs/screener/ranked_candidates.csv
```

## Git policy

Do commit:

```text
screener/ai_portfolio_screener.py
screener/src/live_market_caps.py
screener/src/__init__.py
screener/configs/config_full_ai_anchor_q.yaml
screener/data/current_holdings_template.csv
screener/requirements-screener.txt
docs/SCREENER_README.md
```

Do not commit:

```text
screener/data/current_holdings.csv
screener/data/live_market_caps_stockanalysis.csv
outputs/screener/
data/raw/point_in_time_market_caps.csv
wrds_data/
```
