from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_output_dirs, load_config
from src.data.yahoo_provider import download_prices, unique_preserve_order
from src.universe.market_cap_universe import build_static_equity_universe
from src.universe.point_in_time_market_caps import (
    build_point_in_time_universe_by_date,
    load_point_in_time_market_caps,
    unique_tickers_from_market_cap_file,
)
from src.strategy.weights import (
    latest_target_weights_on_or_before,
    make_target_weights,
    make_target_weights_by_date,
    union_tickers_from_target_weights_by_date,
)
from src.engine.execution import ExecutionModel
from src.engine.backtest import BacktestEngine
from src.analytics.metrics import compute_metrics, risk_contribution, rolling_analytics
from src.analytics.risk import cap_breach_report, cluster_exposure_report, final_holdings_snapshot
from src.analytics.comparison import run_strategy_comparison
from src.reports.charts import save_main_charts
from src.reports.pdf_report import export_pdf


def print_metric_block(title, metrics):
    print(f"\n\n===== {title} =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            lk = str(k).lower()
            if "beta" in lk:
                print(f"{k:25}: {v:,.4f}")
            elif "return" in lk or "volatility" in lk or "drawdown" in lk or "correlation" in lk:
                print(f"{k:25}: {v:,.4%}")
            elif "equity" in lk:
                print(f"{k:25}: ${v:,.2f}")
            else:
                print(f"{k:25}: {v:,.4f}")
        else:
            print(f"{k:25}: {v}")


def make_engine(prices, target_weights, target_weights_by_date, equity_universe, diversifiers, execution, cash_return_annual):
    return BacktestEngine(
        prices=prices,
        target_weights=target_weights,
        target_weights_by_date=target_weights_by_date,
        equity_universe=equity_universe,
        diversifiers=diversifiers,
        execution=execution,
        cash_return_annual=cash_return_annual,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategy_refined_free.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    output_dir = Path(cfg["report"]["output_dir"])
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    start = cfg["project"]["start_date"]
    end = cfg["project"].get("end_date")
    initial_capital = float(cfg["project"]["initial_capital"])
    benchmark = cfg["portfolio"]["benchmark"]
    diversifiers = {k: float(v) for k, v in cfg["portfolio"]["diversifiers"].items()}
    equity_weight = float(cfg["portfolio"]["equity_weight"])

    use_pit = bool(cfg["universe"].get("use_point_in_time_market_caps", False))
    pit_path = cfg["universe"].get("point_in_time_market_caps_path")

    seed_tickers = unique_preserve_order(
        cfg["universe"]["anchors"]
        + cfg["universe"]["fallback_mega_caps"]
        + list(diversifiers.keys())
        + [benchmark]
    )

    market_caps = None
    if use_pit:
        print("\nPoint-in-time market-cap mode is ENABLED.")
        market_caps = load_point_in_time_market_caps(pit_path)
        pit_tickers = unique_tickers_from_market_cap_file(pit_path)
        seed_tickers = unique_preserve_order(seed_tickers + pit_tickers)
        print(f"Loaded {len(market_caps):,} market-cap rows and {len(pit_tickers):,} unique PIT tickers.")
    else:
        print("\nPoint-in-time market-cap mode is DISABLED. Using static free-data universe.")

    prices = download_prices(
        seed_tickers,
        start=start,
        end=end,
        return_frequency=cfg["data"]["return_frequency"],
        cache_dir=cfg["data"]["cache_dir"],
        cache_prices=cfg["data"]["cache_prices"],
    )

    target_weights_by_date = None
    universe_membership = pd.DataFrame()

    if use_pit:
        universe_by_date, universe_membership = build_point_in_time_universe_by_date(
            prices_index=prices.index,
            market_caps=market_caps,
            anchors=cfg["universe"]["anchors"],
            total_positions=int(cfg["universe"]["total_equity_positions"]),
            available_cols=prices.columns,
            ranking_frequency=cfg["universe"].get("point_in_time_ranking_frequency", cfg["rebalance"]["scheduled_frequency"]),
            asof_lag_days=int(cfg["universe"].get("point_in_time_asof_lag_days", 0)),
            min_market_cap=float(cfg["universe"].get("point_in_time_min_market_cap", 0)),
        )
        target_weights_by_date = make_target_weights_by_date(universe_by_date, equity_weight, diversifiers)
        target_weights = latest_target_weights_on_or_before(target_weights_by_date, prices.index[-1])
        equity_universe = unique_preserve_order(universe_membership["ticker"].dropna().astype(str).tolist())
        all_needed = unique_preserve_order(union_tickers_from_target_weights_by_date(target_weights_by_date) + [benchmark])
        universe_membership.to_csv(output_dir / "csv" / "universe_membership.csv", index=False)
    else:
        equity_universe = build_static_equity_universe(
            anchors=cfg["universe"]["anchors"],
            fallback_mega_caps=cfg["universe"]["fallback_mega_caps"],
            available_cols=prices.columns,
            total_positions=int(cfg["universe"]["total_equity_positions"]),
            use_web_scraper=bool(cfg["universe"]["use_web_scraper"]),
            scrape_top_n=int(cfg["universe"]["scrape_top_n"]),
        )
        target_weights = make_target_weights(equity_universe, equity_weight, diversifiers)
        all_needed = unique_preserve_order(list(target_weights.keys()) + [benchmark])

    prices = download_prices(
        all_needed,
        start=start,
        end=end,
        return_frequency=cfg["data"]["return_frequency"],
        cache_dir=cfg["data"]["cache_dir"],
        cache_prices=cfg["data"]["cache_prices"],
    )

    benchmark_prices = prices[benchmark]

    execution = ExecutionModel(
        whole_shares=bool(cfg["execution"]["whole_shares"]),
        commission_per_trade=float(cfg["execution"]["commission_per_trade"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]),
        min_trade_dollars=float(cfg["execution"]["min_trade_dollars"]),
    )

    engine = make_engine(
        prices, target_weights, target_weights_by_date, equity_universe, diversifiers, execution, float(cfg["execution"]["cash_return_annual"])
    )

    equity, cash, holdings, log = engine.run(
        initial_capital=initial_capital,
        scheduled_frequency=cfg["rebalance"]["scheduled_frequency"],
        hard_cap_check_frequency=cfg["rebalance"]["hard_cap_check_frequency"],
        stock_hard_trigger=float(cfg["rebalance"]["stock_hard_trigger"]),
        stock_hard_trim_to=float(cfg["rebalance"]["stock_hard_trim_to"]),
        diversifier_hard_triggers={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        diversifier_hard_trim_to={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )

    main_metrics = compute_metrics(equity, benchmark_prices)
    print_metric_block("MAIN STRATEGY METRICS", main_metrics)

    equity.to_csv(output_dir / "csv" / "portfolio_equity.csv")
    cash.to_csv(output_dir / "csv" / "portfolio_cash.csv")
    holdings.to_csv(output_dir / "csv" / "portfolio_holdings.csv")
    log.to_csv(output_dir / "csv" / "rebalance_log.csv", index=False)

    strategy_comparison = run_strategy_comparison(
        prices=prices,
        benchmark_prices=benchmark_prices,
        initial_capital=initial_capital,
        target_weights=target_weights,
        target_weights_by_date=target_weights_by_date,
        equity_universe=equity_universe,
        diversifiers=diversifiers,
        execution=execution,
        cash_return_annual=float(cfg["execution"]["cash_return_annual"]),
        stock_hard_trigger=float(cfg["rebalance"]["stock_hard_trigger"]),
        stock_hard_trim_to=float(cfg["rebalance"]["stock_hard_trim_to"]),
        diversifier_hard_triggers={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        diversifier_hard_trim_to={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )
    strategy_comparison.to_csv(output_dir / "csv" / "strategy_comparison.csv", index=False)

    rows = []
    for name, (w_start, w_end) in cfg["windows"].items():
        wp = prices.loc[prices.index >= pd.to_datetime(w_start)]
        if w_end is not None:
            wp = wp.loc[wp.index <= pd.to_datetime(w_end)]
        if len(wp) < 10:
            continue
        we = make_engine(wp, target_weights, target_weights_by_date, equity_universe, diversifiers, execution, float(cfg["execution"]["cash_return_annual"]))
        eq_w, _, _, _ = we.run(
            initial_capital=initial_capital,
            scheduled_frequency=cfg["rebalance"]["scheduled_frequency"],
            hard_cap_check_frequency=cfg["rebalance"]["hard_cap_check_frequency"],
            stock_hard_trigger=float(cfg["rebalance"]["stock_hard_trigger"]),
            stock_hard_trim_to=float(cfg["rebalance"]["stock_hard_trim_to"]),
            diversifier_hard_triggers={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
            diversifier_hard_trim_to={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
        )
        m = compute_metrics(eq_w, wp[benchmark])
        m["Window"] = name
        rows.append(m)

    window_summary = pd.DataFrame(rows)
    if not window_summary.empty:
        first_cols = ["Window", "Start", "End", "Weeks"]
        rest = [c for c in window_summary.columns if c not in first_cols]
        window_summary = window_summary[first_cols + rest]
    window_summary.to_csv(output_dir / "csv" / "window_summary.csv", index=False)

    final_target_weights = engine.get_target_weights_for_date(prices.index[-1])
    holdings_snapshot = final_holdings_snapshot(holdings, equity.iloc[-1], final_target_weights)
    holdings_snapshot.to_csv(output_dir / "csv" / "final_holdings_snapshot.csv", index=False)

    cluster = cluster_exposure_report(
        holdings_snapshot,
        cfg["risk_alerts"]["direct_semi_tickers"],
        cfg["risk_alerts"]["ai_platform_tickers"],
        list(diversifiers.keys()),
    )
    cluster.to_csv(output_dir / "csv" / "cluster_exposure.csv", index=False)

    breaches = cap_breach_report(
        holdings_snapshot,
        equity_universe,
        diversifiers,
        float(cfg["rebalance"]["stock_hard_trigger"]),
        float(cfg["rebalance"]["stock_hard_trim_to"]),
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )
    breaches.to_csv(output_dir / "csv" / "cap_breach_report.csv", index=False)

    risk_contrib = risk_contribution(prices, final_target_weights)
    risk_contrib.to_csv(output_dir / "csv" / "risk_contribution.csv")

    rolling = rolling_analytics(equity, benchmark_prices, windows=[13, 26, 52, 104, 156, 260])
    rolling.to_csv(output_dir / "csv" / "rolling_metrics.csv")

    chart_paths = save_main_charts(equity, benchmark_prices, initial_capital, output_dir / "charts")

    if bool(cfg["report"]["export_pdf"]):
        try:
            pdf_path = export_pdf(
                output_path=output_dir / "reports" / "portfolio_backtest_report.pdf",
                config=cfg,
                chart_paths=chart_paths,
                target_weights=final_target_weights,
                main_metrics=main_metrics,
                window_summary=window_summary,
                holdings_snapshot=holdings_snapshot,
                cluster_exposure=cluster,
                cap_breach=breaches,
                risk_contrib=risk_contrib,
                strategy_comparison=strategy_comparison,
            )
            print(f"\nPDF report exported: {pdf_path}")
        except TypeError:
            # Backward compatibility if your local pdf_report.py has not been upgraded yet.
            pdf_path = export_pdf(
                output_path=output_dir / "reports" / "portfolio_backtest_report.pdf",
                config=cfg,
                chart_paths=chart_paths,
                target_weights=final_target_weights,
                main_metrics=main_metrics,
                window_summary=window_summary,
                holdings_snapshot=holdings_snapshot,
                cluster_exposure=cluster,
                cap_breach=breaches,
                risk_contrib=risk_contrib,
            )
            print(f"\nPDF report exported: {pdf_path}")

    print("\nDone. Check outputs/ for reports, charts, and CSVs.")


if __name__ == "__main__":
    main()
