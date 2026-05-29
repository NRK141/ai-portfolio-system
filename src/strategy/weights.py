from __future__ import annotations

from typing import Dict

import pandas as pd


def make_target_weights(
    equity_universe: list[str],
    equity_weight: float,
    diversifier_targets: Dict[str, float],
) -> Dict[str, float]:
    if not equity_universe:
        raise ValueError("Equity universe is empty.")

    equity_per_name = equity_weight / len(equity_universe)
    weights = {t: equity_per_name for t in equity_universe}
    weights.update(diversifier_targets)

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Target weights must sum to 1.0, got {total:.6f}")

    return weights


def make_target_weights_by_date(
    universe_by_date: dict[pd.Timestamp, list[str]],
    equity_weight: float,
    diversifier_targets: Dict[str, float],
) -> dict[pd.Timestamp, Dict[str, float]]:
    out: dict[pd.Timestamp, Dict[str, float]] = {}
    for date, universe in universe_by_date.items():
        out[pd.Timestamp(date)] = make_target_weights(universe, equity_weight, diversifier_targets)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def union_tickers_from_target_weights_by_date(
    target_weights_by_date: dict[pd.Timestamp, Dict[str, float]]
) -> list[str]:
    seen = set()
    out = []
    for _, weights in sorted(target_weights_by_date.items(), key=lambda kv: kv[0]):
        for t in weights:
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out


def latest_target_weights_on_or_before(
    target_weights_by_date: dict[pd.Timestamp, Dict[str, float]],
    date: pd.Timestamp,
) -> Dict[str, float]:
    date = pd.Timestamp(date)
    valid_dates = [d for d in target_weights_by_date if d <= date]
    if not valid_dates:
        return target_weights_by_date[min(target_weights_by_date)]
    return target_weights_by_date[max(valid_dates)]
