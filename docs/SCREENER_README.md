# Full AI-Anchor PIT Q Screener

This is the live screener for the forward-looking portfolio.

## Fixes in this version

1. Fractional targets are enabled by default:

```yaml
execution:
  whole_shares: false
```

2. Duplicate Alphabet exposure is prevented:
   - GOOG is preferred.
   - GOOGL is skipped when GOOG is already in the target universe.

3. PIT data age warning is added:
   - The screener alerts if the WRDS-derived market-cap ranking file is stale.

## Run

```powershell
python screener/ai_portfolio_screener.py --config screener/config_full_ai_anchor_q.yaml --capital 25000
```

Or with live holdings:

```powershell
python screener/ai_portfolio_screener.py --config screener/config_full_ai_anchor_q.yaml --holdings screener/current_holdings.csv --cash 500
```

## Outputs

```text
outputs/screener/screener_report.md
outputs/screener/target_allocation.csv
outputs/screener/trade_plan.csv
outputs/screener/alerts.csv
outputs/screener/ranked_candidates.csv
```

Do not commit real WRDS data, current holdings, or screener outputs.
