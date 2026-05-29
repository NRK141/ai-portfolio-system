from __future__ import annotations

from typing import Dict


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
