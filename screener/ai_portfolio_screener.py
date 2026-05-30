from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: yfinance. Install with:\n"
        "pip install -r screener/requirements-screener.txt"
    ) from exc


YFINANCE_TICKER_MAP = {
    "FB": "META",
    "BRK": "BRK-B",
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}


def normalize_ticker(ticker: str) -> str:
    if ticker is None:
        return ""
    t = str(ticker).strip().upper()
    t = t.replace(".", "-")
    return YFINANCE_TICKER_MAP.get(t, t)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        t = normalize_ticker(item)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def first_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() != 4:
        d += pd.Timedelta(days=1)
    return d


def current_and_next_rebalance_dates(asof: pd.Timestamp, frequency: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    frequency = frequency.upper()

    if frequency == "M":
        months = list(range(1, 13))
    elif frequency == "Q":
        months = [1, 4, 7, 10]
    elif frequency == "SA":
        months = [1, 7]
    elif frequency == "Y":
        months = [1]
    else:
        raise ValueError(f"Unsupported frequency: {frequency}. Use M, Q, SA, or Y.")

    dates = []
    for y in [asof.year - 1, asof.year, asof.year + 1]:
        for m in months:
            dates.append(first_friday(y, m))

    dates = sorted(dates)
    current = max(d for d in dates if d <= asof)
    nxt = min(d for d in dates if d > asof)
    return current, nxt


def download_latest_prices(tickers: list[str]) -> dict[str, float]:
    tickers = unique_preserve_order(tickers)

    data = yf.download(
        tickers,
        period="15d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("yfinance returned no price data.")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data["Close"].to_frame(tickers[0])

    latest = close.ffill().iloc[-1].dropna()
    return {normalize_ticker(k): float(v) for k, v in latest.items() if pd.notna(v) and float(v) > 0}


def load_pit_market_caps(path: str | Path, asof: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), None

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"date", "ticker", "market_cap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PIT market-cap file missing columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "market_cap"])
    df = df[(df["ticker"] != "") & (df["market_cap"] > 0)]

    if df.empty:
        return pd.DataFrame(), None

    global_latest_date = pd.Timestamp(df["date"].max())

    df = df[df["date"] <= asof]
    if df.empty:
        return pd.DataFrame(), global_latest_date

    df = df.sort_values(["ticker", "date"])
    latest = df.groupby("ticker", as_index=False).tail(1)
    latest = latest.sort_values("market_cap", ascending=False).reset_index(drop=True)
    latest["market_cap_rank"] = latest.index + 1

    return latest, global_latest_date


def should_skip_share_class(ticker: str, selected: list[str], cfg: dict) -> bool:
    rules = cfg.get("share_class_rules", {})
    preferred = normalize_ticker(rules.get("alphabet_preferred", "GOOG"))
    alternates = set(unique_preserve_order(rules.get("alphabet_alternates", ["GOOGL"])))
    t = normalize_ticker(ticker)

    if t in alternates and preferred in selected:
        return True

    return False


def build_ranked_source(cfg: dict, ranked_caps: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if ranked_caps.empty:
        ranked = pd.DataFrame({"ticker": unique_preserve_order(cfg["portfolio"]["fallback_mega_caps"])})
        ranked["market_cap_rank"] = range(1, len(ranked) + 1)
        ranked["market_cap"] = pd.NA
        ranked["date"] = pd.NaT
        return ranked, "FALLBACK_STATIC_LIST"

    return ranked_caps.copy(), "PIT_WRDS"


def build_target_universe(
    cfg: dict,
    ranked_caps: pd.DataFrame,
    available_price_tickers: set[str],
) -> tuple[list[str], pd.DataFrame, str]:
    p = cfg["portfolio"]
    anchors = unique_preserve_order(p["anchors"])
    total_positions = int(p["total_equity_positions"])

    ranked, source = build_ranked_source(cfg, ranked_caps)

    universe = []

    for t in anchors:
        if t in available_price_tickers and t not in universe:
            universe.append(t)

    for raw_t in ranked["ticker"].astype(str).tolist():
        t = normalize_ticker(raw_t)

        if not t or t not in available_price_tickers:
            continue

        if t in universe:
            continue

        if should_skip_share_class(t, universe, cfg):
            continue

        universe.append(t)

        if len(universe) >= total_positions:
            break

    if len(universe) < total_positions:
        raise RuntimeError(
            f"Only built {len(universe)} target names. Need {total_positions}. "
            "Increase candidate_top_n or check yfinance prices."
        )

    selected = ranked.copy()
    selected["ticker"] = selected["ticker"].map(normalize_ticker)
    selected["selected"] = selected["ticker"].isin(universe)
    selected["skipped_duplicate_share_class"] = selected["ticker"].apply(
        lambda t: should_skip_share_class(t, universe, cfg)
    )

    return universe, selected, source


def build_target_weights(cfg: dict, universe: list[str]) -> dict[str, float]:
    p = cfg["portfolio"]
    equity_weight = float(p["equity_weight"])
    eq_w = equity_weight / len(universe)

    weights = {t: eq_w for t in universe}

    for t, w in p["diversifiers"].items():
        weights[normalize_ticker(t)] = float(w)

    return weights


def read_holdings(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["ticker", "shares"])

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "shares"])

    df = pd.read_csv(path)
    if "ticker" not in df.columns or "shares" not in df.columns:
        raise ValueError("Holdings CSV must have columns: ticker,shares")

    df = df.copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df = df.groupby("ticker", as_index=False)["shares"].sum()
    df = df[df["ticker"] != ""]
    return df


def make_allocation_and_trade_plan(
    cfg: dict,
    target_weights: dict[str, float],
    prices: dict[str, float],
    holdings: pd.DataFrame,
    capital: float | None,
    cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    min_trade = float(cfg["execution"]["min_trade_dollars"])
    whole_shares = bool(cfg["execution"].get("whole_shares", False))

    holdings_map = dict(zip(holdings["ticker"], holdings["shares"])) if not holdings.empty else {}
    all_tickers = unique_preserve_order(list(target_weights.keys()) + list(holdings_map.keys()))

    if holdings.empty:
        if capital is None:
            raise ValueError("Provide --capital when no holdings file is supplied.")
        portfolio_value = float(capital)
    else:
        invested_value = sum(float(holdings_map.get(t, 0.0)) * float(prices.get(t, 0.0)) for t in all_tickers)
        portfolio_value = invested_value + float(cash)

    rows = []

    for t in all_tickers:
        px = float(prices.get(t, 0.0))
        current_shares = float(holdings_map.get(t, 0.0))
        current_value = current_shares * px
        target_weight = float(target_weights.get(t, 0.0))
        target_dollars = target_weight * portfolio_value

        if px > 0:
            raw_target_shares = target_dollars / px
            target_shares = math.floor(raw_target_shares) if whole_shares else raw_target_shares
        else:
            target_shares = 0.0

        delta_shares = target_shares - current_shares
        trade_dollars = delta_shares * px

        if abs(trade_dollars) < min_trade:
            action = "HOLD"
            delta_shares = 0.0
            trade_dollars = 0.0
        elif delta_shares > 0:
            action = "BUY"
        elif delta_shares < 0:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append(
            {
                "ticker": t,
                "price": px,
                "target_weight": target_weight,
                "current_shares": current_shares,
                "current_value": current_value,
                "current_weight": current_value / portfolio_value if portfolio_value else 0.0,
                "target_dollars": target_dollars,
                "target_shares": target_shares,
                "target_value_estimate": target_shares * px,
                "delta_shares": delta_shares,
                "trade_dollars": trade_dollars,
                "action": action,
            }
        )

    allocation = pd.DataFrame(rows)
    allocation = allocation.sort_values(["target_weight", "ticker"], ascending=[False, True]).reset_index(drop=True)

    trades = allocation[allocation["action"] != "HOLD"].copy()
    trades = trades.sort_values("trade_dollars", key=lambda s: s.abs(), ascending=False)

    return allocation, trades, portfolio_value


def build_alerts(
    cfg: dict,
    allocation: pd.DataFrame,
    latest_rank_date: pd.Timestamp | None,
    asof: pd.Timestamp,
    ranking_source: str,
) -> pd.DataFrame:
    risk = cfg["risk"]
    alerts = []

    actual_weight = dict(zip(allocation["ticker"], allocation["current_weight"]))
    target_weight = dict(zip(allocation["ticker"], allocation["target_weight"]))

    if sum(actual_weight.values()) < 0.01:
        actual_weight = target_weight.copy()

    stale_days = int(cfg.get("data", {}).get("stale_rank_data_days", 45))
    if ranking_source == "PIT_WRDS" and latest_rank_date is not None:
        age_days = int((pd.Timestamp(asof).normalize() - pd.Timestamp(latest_rank_date).normalize()).days)
        if age_days > stale_days:
            alerts.append(
                {
                    "severity": "WATCH",
                    "type": "STALE_RANK_DATA",
                    "ticker": "",
                    "message": (
                        f"PIT market-cap data is {age_days} days old "
                        f"(latest {pd.Timestamp(latest_rank_date).date()}). "
                        "Refresh WRDS-derived market caps when available."
                    ),
                }
            )

    stock_trigger = float(risk["stock_hard_trigger"])
    stock_trim = float(risk["stock_hard_trim_to"])
    diversifier_triggers = {normalize_ticker(k): float(v) for k, v in risk["diversifier_hard_triggers"].items()}
    diversifier_trim = {normalize_ticker(k): float(v) for k, v in risk["diversifier_hard_trim_to"].items()}

    for t, w in actual_weight.items():
        if t in diversifier_triggers:
            trigger = diversifier_triggers[t]
            trim = diversifier_trim[t]
        else:
            trigger = stock_trigger
            trim = stock_trim

        if w >= trigger:
            alerts.append(
                {
                    "severity": "HARD",
                    "type": "POSITION_CAP",
                    "ticker": t,
                    "message": f"{t} is {w:.2%}, above hard trigger {trigger:.2%}. Trim toward {trim:.2%}.",
                }
            )
        elif w >= trim and t not in diversifier_triggers:
            alerts.append(
                {
                    "severity": "WATCH",
                    "type": "POSITION_DRIFT",
                    "ticker": t,
                    "message": f"{t} is {w:.2%}, above trim-to level {trim:.2%} but below hard trigger {trigger:.2%}.",
                }
            )

    direct = set(unique_preserve_order(risk["direct_semi_tickers"]))
    platform = set(unique_preserve_order(risk["ai_platform_tickers"]))
    diversifiers = set(unique_preserve_order(cfg["portfolio"]["diversifiers"].keys()))

    direct_w = sum(w for t, w in actual_weight.items() if t in direct)
    total_ai_w = sum(w for t, w in actual_weight.items() if t in direct or t in platform)
    diversifier_w = sum(w for t, w in actual_weight.items() if t in diversifiers)

    cluster_checks = [
        ("DIRECT_SEMI", direct_w, float(risk["direct_semi_soft_alert"]), float(risk["direct_semi_hard_alert"])),
        ("TOTAL_AI", total_ai_w, float(risk["total_ai_soft_alert"]), float(risk["total_ai_hard_alert"])),
    ]

    for name, w, soft, hard in cluster_checks:
        if w >= hard:
            alerts.append({"severity": "HARD", "type": name, "ticker": "", "message": f"{name} exposure is {w:.2%}, above hard review level {hard:.2%}."})
        elif w >= soft:
            alerts.append({"severity": "WATCH", "type": name, "ticker": "", "message": f"{name} exposure is {w:.2%}, above soft review level {soft:.2%}."})

    min_div = float(risk["diversifier_min_review"])
    if diversifier_w < min_div:
        alerts.append({"severity": "WATCH", "type": "DIVERSIFIERS", "ticker": "", "message": f"Diversifier sleeve is {diversifier_w:.2%}, below review level {min_div:.2%}."})

    if not alerts:
        alerts.append({"severity": "OK", "type": "ALL_CLEAR", "ticker": "", "message": "No hard-cap, stale-data, or cluster alerts triggered."})

    return pd.DataFrame(alerts)


def save_report(
    output_dir: Path,
    cfg: dict,
    asof: pd.Timestamp,
    current_rebalance: pd.Timestamp,
    next_rebalance: pd.Timestamp,
    source: str,
    latest_rank_date: pd.Timestamp | None,
    universe: list[str],
    portfolio_value: float,
    allocation: pd.DataFrame,
    trades: pd.DataFrame,
    alerts: pd.DataFrame,
    ranked_candidates: pd.DataFrame,
):
    lines = []
    lines.append(f"# {cfg['portfolio']['name']}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"As of: {asof.date()}")
    lines.append(f"Ranking frequency: {cfg['portfolio']['ranking_frequency']}")
    lines.append(f"Current ranking date: {current_rebalance.date()}")
    lines.append(f"Next ranking date: {next_rebalance.date()}")
    lines.append(f"Ranking source: {source}")
    lines.append(f"Latest PIT rank data date: {latest_rank_date.date() if latest_rank_date is not None else 'N/A'}")
    lines.append(f"Whole-share targets: {cfg['execution'].get('whole_shares', False)}")
    lines.append(f"Portfolio value used: ${portfolio_value:,.2f}")
    lines.append("")
    lines.append("## Target Universe")
    lines.append(", ".join(universe))
    lines.append("")
    lines.append("## Alerts")
    lines.append(alerts.to_string(index=False))
    lines.append("")
    lines.append("## Trade Plan")
    if trades.empty:
        lines.append("No trades above minimum trade-dollar threshold.")
    else:
        cols = ["ticker", "action", "delta_shares", "trade_dollars", "price", "target_weight"]
        lines.append(trades[cols].to_string(index=False))
    lines.append("")
    lines.append("## Target Allocation")
    cols = ["ticker", "target_weight", "current_weight", "target_shares", "delta_shares", "action"]
    lines.append(allocation[cols].to_string(index=False))
    lines.append("")
    lines.append("## Top Ranked Candidates")
    if not ranked_candidates.empty:
        cols = [c for c in ["ticker", "market_cap_rank", "market_cap", "date", "selected", "skipped_duplicate_share_class"] if c in ranked_candidates.columns]
        lines.append(ranked_candidates[cols].head(40).to_string(index=False))
    else:
        lines.append("No PIT ranked candidates available; fallback static list was used.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is a live screener, not a backtest.")
    lines.append("- Full AI anchors are intentional for the forward-looking portfolio.")
    lines.append("- Quarterly ranking changes rotate the non-anchor top-market-cap sleeve.")
    lines.append("- Fractional share targets are recommended for small account sizes.")
    lines.append("- Weekly hard-cap checks should be used for risk control.")
    lines.append("- WRDS-derived data should stay local/private and ignored by Git.")

    (output_dir / "screener_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Full AI-Anchor PIT Q portfolio screener.")
    parser.add_argument("--config", default="screener/configs/config_full_ai_anchor_q.yaml")
    parser.add_argument("--holdings", default=None, help="Optional CSV with ticker,shares.")
    parser.add_argument("--capital", type=float, default=None, help="Portfolio value to allocate if no holdings file is supplied.")
    parser.add_argument("--cash", type=float, default=0.0, help="Cash balance if holdings file is supplied.")
    parser.add_argument("--asof", default=None, help="YYYY-MM-DD. Defaults to today.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()

    out_dir = Path(cfg["outputs"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    current_rebalance, next_rebalance = current_and_next_rebalance_dates(
        asof=asof,
        frequency=cfg["portfolio"]["ranking_frequency"],
    )

    pit_path = cfg["data"]["point_in_time_market_caps_path"]
    ranked_caps, latest_rank_date = load_pit_market_caps(pit_path, asof=asof)

    candidate_top_n = int(cfg["data"].get("candidate_top_n", 100))
    if not ranked_caps.empty:
        rank_candidates = ranked_caps.head(candidate_top_n)
    else:
        rank_candidates = pd.DataFrame({"ticker": unique_preserve_order(cfg["portfolio"]["fallback_mega_caps"])})
        rank_candidates["market_cap_rank"] = range(1, len(rank_candidates) + 1)
        rank_candidates["market_cap"] = pd.NA
        rank_candidates["date"] = pd.NaT

    tickers_to_price = unique_preserve_order(
        cfg["portfolio"]["anchors"]
        + list(cfg["portfolio"]["diversifiers"].keys())
        + [cfg["portfolio"]["benchmark"]]
        + rank_candidates["ticker"].dropna().astype(str).tolist()
    )

    print(f"Downloading latest prices for {len(tickers_to_price)} tickers...")
    prices = download_latest_prices(tickers_to_price)
    available = set(prices.keys())

    universe, ranked_candidates, source = build_target_universe(cfg, ranked_caps, available)
    target_weights = build_target_weights(cfg, universe)

    final_tickers = unique_preserve_order(list(target_weights.keys()) + [cfg["portfolio"]["benchmark"]])
    missing_prices = [t for t in final_tickers if t not in prices]
    if missing_prices:
        raise RuntimeError(f"Missing prices for required target tickers: {missing_prices}")

    holdings = read_holdings(args.holdings)
    allocation, trades, portfolio_value = make_allocation_and_trade_plan(
        cfg=cfg,
        target_weights=target_weights,
        prices=prices,
        holdings=holdings,
        capital=args.capital,
        cash=float(args.cash),
    )

    alerts = build_alerts(
        cfg=cfg,
        allocation=allocation,
        latest_rank_date=latest_rank_date,
        asof=asof,
        ranking_source=source,
    )

    ranked_candidates = ranked_candidates.copy()
    ranked_candidates["selected"] = ranked_candidates["ticker"].isin(universe)

    allocation.to_csv(out_dir / "target_allocation.csv", index=False)
    trades.to_csv(out_dir / "trade_plan.csv", index=False)
    alerts.to_csv(out_dir / "alerts.csv", index=False)
    ranked_candidates.to_csv(out_dir / "ranked_candidates.csv", index=False)

    save_report(
        output_dir=out_dir,
        cfg=cfg,
        asof=asof,
        current_rebalance=current_rebalance,
        next_rebalance=next_rebalance,
        source=source,
        latest_rank_date=latest_rank_date,
        universe=universe,
        portfolio_value=portfolio_value,
        allocation=allocation,
        trades=trades,
        alerts=alerts,
        ranked_candidates=ranked_candidates,
    )

    print("\n===== SCREENER COMPLETE =====")
    print(f"Ranking source       : {source}")
    print(f"Latest PIT rank date : {latest_rank_date.date() if latest_rank_date is not None else 'N/A'}")
    print(f"Selected frequency   : {cfg['portfolio']['ranking_frequency']}")
    print(f"Whole-share targets  : {cfg['execution'].get('whole_shares', False)}")
    print(f"Current ranking date : {current_rebalance.date()}")
    print(f"Next ranking date    : {next_rebalance.date()}")
    print(f"Portfolio value used : ${portfolio_value:,.2f}")
    print(f"Target universe      : {', '.join(universe)}")
    print(f"Outputs              : {out_dir}")


if __name__ == "__main__":
    main()
