# PIT Ranking Frequency Modes

The backtester now supports point-in-time market-cap ranking refresh modes:

| Mode | Meaning |
|---|---|
| `M` | Monthly ranking refresh and ticker rotation |
| `Q` | Quarterly ranking refresh and ticker rotation |
| `SA` | Semiannual ranking refresh and ticker rotation |
| `Y` | Annual ranking refresh and ticker rotation |

Set the selected strategy mode in:

```yaml
universe:
  point_in_time_ranking_frequency: "Y"
```

The strategy rotates tickers whenever a new PIT ranking date is generated for the selected mode.

## Frequency comparison

The config can test all modes in one run:

```yaml
analysis:
  run_ranking_frequency_comparison: true
  ranking_frequencies_to_test: ["M", "Q", "SA", "Y"]
```

This creates:

```text
outputs/csv/ranking_frequency_comparison.csv
outputs/csv/universe_membership_all_tested_frequencies.csv
```

## Important

The PIT file itself is local/private WRDS-derived data and should not be committed to GitHub.
