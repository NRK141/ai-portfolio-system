from __future__ import annotations

import numpy as np
import pandas as pd


def final_holdings_snapshot(holdings: pd.DataFrame, equity_value: float, target_weights: dict[str, float]):
    last = holdings.iloc[-1]
    rows = []

    for t, target in target_weights.items():
        value = last.get(f"{t}_Value", 0.0)
        shares = last.get(f"{t}_Shares", 0.0)
        final_weight = value / equity_value if equity_value != 0 else np.nan

        rows.append(
            {
                "Ticker": t,
                "Target Weight": target,
                "Final Shares": shares,
                "Final Value": value,
                "Final Weight": final_weight,
                "Drift From Target": final_weight - target,
                "Drift % of Target": (final_weight - target) / target if target else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("Drift From Target", ascending=False)


def cluster_exposure_report(holdings_snapshot: pd.DataFrame, direct_semi: list[str], ai_platform: list[str], diversifiers: list[str]):
    weight_map = holdings_snapshot.set_index("Ticker")["Final Weight"].to_dict()
    target_map = holdings_snapshot.set_index("Ticker")["Target Weight"].to_dict()

    total_ai = list(dict.fromkeys(direct_semi + ai_platform))
    groups = {
        "Direct Semi / AI Hardware": direct_semi,
        "AI Platform / Mega-cap Tech": ai_platform,
        "Total AI-Related": total_ai,
        "Diversifiers": diversifiers,
    }

    rows = []
    for group, tickers in groups.items():
        target = sum(target_map.get(t, 0.0) for t in tickers)
        actual = sum(weight_map.get(t, 0.0) for t in tickers)
        rows.append({"Group": group, "Target Weight": target, "Final Weight": actual, "Drift From Target": actual - target})

    return pd.DataFrame(rows)


def cap_breach_report(
    holdings_snapshot: pd.DataFrame,
    equity_universe: list[str],
    diversifiers: dict[str, float],
    stock_hard_trigger: float,
    stock_hard_trim_to: float,
    diversifier_hard_triggers: dict[str, float],
    diversifier_hard_trim_to: dict[str, float],
):
    rows = []
    equity_set = set(equity_universe)

    for _, row in holdings_snapshot.iterrows():
        t = row["Ticker"]
        final_w = row["Final Weight"]

        if t in equity_set:
            trigger = stock_hard_trigger
            trim_to = stock_hard_trim_to
        elif t in diversifiers:
            trigger = diversifier_hard_triggers.get(t, np.nan)
            trim_to = diversifier_hard_trim_to.get(t, np.nan)
        else:
            trigger = np.nan
            trim_to = np.nan

        status = "OK"
        if pd.notna(trigger) and final_w > trigger:
            status = "HARD BREACH"

        rows.append(
            {
                "Ticker": t,
                "Target Weight": row["Target Weight"],
                "Final Weight": final_w,
                "Hard Trigger": trigger,
                "Trim-To Weight": trim_to,
                "Status": status,
            }
        )

    return pd.DataFrame(rows).sort_values(["Status", "Final Weight"], ascending=[True, False])
