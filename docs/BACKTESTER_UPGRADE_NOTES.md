# Backtester Upgrade Notes

This patch adds two improvements:

1. Configurable hard-cap check frequency.
   - Old behavior: monthly hard-cap check only.
   - New behavior: `hard_cap_check_frequency` can be `NONE`, `W`, `M`, `Q`, `SA`, or `Y`.
   - Default recommendation: `W`.

2. Strategy comparison runner.
   - Automatically tests multiple rebalance/hard-cap variants.
   - Saves results to `outputs/csv/strategy_comparison.csv`.
   - Adds the comparison table to the PDF report.

## Files changed

Replace these files:

- `configs/strategy_refined_free.yaml`
- `src/engine/backtest.py`
- `src/reports/pdf_report.py`
- `run_backtest.py`

Add these new files:

- `src/analytics/comparison.py`
- `tests/test_rebalance_dates.py`
- `tests/test_hard_cap.py`
- `docs/BACKTESTER_UPGRADE_NOTES.md`

## Run

```powershell
python run_backtest.py --config configs/strategy_refined_free.yaml
```

## Test

```powershell
pip install pytest
pytest
```
