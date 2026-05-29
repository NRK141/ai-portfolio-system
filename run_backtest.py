from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analytics.comparison import run_strategy_comparison
from src.analytics.metrics import compute_metrics, risk_contribution, rolling_analytics
from src.analytics.risk import cap_breach_report, cluster_exposure_report, final_holdings_snapshot
from src.config import ensure_output_dirs, load_config
from src.data.yahoo_provider import download_prices, unique_preserve_order
from src.engine.backtest import BacktestEngine
from src.engine.execution import ExecutionModel
from src.reports.charts import save_main_charts
from src.reports.pdf_report import export_pdf
from src.strategy.weights import make_target_weights
from src.universe.market_cap_universe import build_static_equity_universe


def print_metric_block(title, metrics):
    print(f"\n\n===== {title} =====")
    for key, value in metrics.items():
        if isinstance(value, float):
            key_lower = str(key).lower()
            if "beta" in key_lower:
                print(f"{key:25}: {value:,.4f}")
            elif "return" in key_lower or "volatility" in key_lower or "drawdown" in key_lower or "correlation" in key_lower:
                print(f"{key:25}: {value:,.4%}")
            elif "equity" in key_lower:
                print(f"{key:25}: ${value:,.2f}")
            else:
                print(f"{key:25}: {value:,.4f}")
        else:
            print(f"{key:25}: {value}")


def build_execution(cfg):
    return ExecutionModel(
        whole_shares=bool(cfg["execution"]["whole_shares"]),
        commission_per_trade=float(cfg["execution"]["commission_per_trade"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]),
        min_trade_dollars=float(cfg["execution"]["min_trade_dollars"]),
    )


def run_engine(cfg, prices, target_weights, equity_universe, execution, initial_capital):
    engine = BacktestEngine(
        prices=prices,
        target_weights=target_weights,
        equity_universe=equity_universe,
        diversifiers=cfg["portfolio"]["diversifiers"],
        execution=execution,
        cash_return_annual=float(cfg["execution"]["cash_return_annual"]),
    )

    return engine.run(
        initial_capital=initial_capital,
        scheduled_frequency=cfg["rebalance"]["scheduled_frequency"],
        hard_cap_check_frequency=cfg["rebalance"]["hard_cap_check_frequency"],
        stock_hard_trigger=float(cfg["rebalance"]["stock_hard_trigger"]),
        stock_hard_trim_to=float(cfg["rebalance"]["stock_hard_trim_to"]),
        diversifier_hard_triggers={
            k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()
        },
        diversifier_hard_trim_to={
            k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategy_refined_free.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    output_dir = Path(cfg["report"]["output_dir"])
    csv_dir = output_dir / "csv"
    charts_dir = output_dir / "charts"
    reports_dir = output_dir / "reports"
    csv_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["project"]["start_date"]
    end = cfg["project"].get("end_date")
    initial_capital = float(cfg["project"]["initial_capital"])
    benchmark = cfg["portfolio"]["benchmark"]

    # Initial price pull with seed tickers so we can determine which tickers are available.
    seed_tickers = unique_preserve_order(
        cfg["universe"]["anchors"]
        + cfg["universe"]["fallback_mega_caps"]
        + list(cfg["portfolio"]["diversifiers"].keys())
        + [benchmark]
    )

    seed_prices = download_prices(
        seed_tickers,
        start=start,
        end=end,
        return_frequency=cfg["data"]["return_frequency"],
        cache_dir=cfg["data"]["cache_dir"],
        cache_prices=cfg["data"]["cache_prices"],
    )

    equity_universe = build_static_equity_universe(
        anchors=cfg["universe"]["anchors"],
        fallback_mega_caps=cfg["universe"]["fallback_mega_caps"],
        available_cols=seed_prices.columns,
        total_positions=int(cfg["universe"]["total_equity_positions"]),
        use_web_scraper=bool(cfg["universe"]["use_web_scraper"]),
        scrape_top_n=int(cfg["universe"]["scrape_top_n"]),
    )

    target_weights = make_target_weights(
        equity_universe,
        float(cfg["portfolio"]["equity_weight"]),
        {k: float(v) for k, v in cfg["portfolio"]["diversifiers"].items()},
    )

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

    print("\n===== FINAL EQUITY UNIVERSE =====")
    for i, ticker in enumerate(equity_universe, start=1):
        print(f"{i:02d}. {ticker}")

    print("\n===== TARGET WEIGHTS =====")
    for ticker, weight in target_weights.items():
        print(f"{ticker:>6}: {weight:.2%}")

    execution = build_execution(cfg)
    equity, cash, holdings, log = run_engine(
        cfg=cfg,
        prices=prices,
        target_weights=target_weights,
        equity_universe=equity_universe,
        execution=execution,
        initial_capital=initial_capital,
    )

    main_metrics = compute_metrics(equity, benchmark_prices)
    print_metric_block("MAIN STRATEGY METRICS", main_metrics)

    equity.to_csv(csv_dir / "portfolio_equity.csv")
    cash.to_csv(csv_dir / "portfolio_cash.csv")
    holdings.to_csv(csv_dir / "portfolio_holdings.csv")
    log.to_csv(csv_dir / "rebalance_log.csv", index=False)

    # Window tests.
    rows = []
    for name, (window_start, window_end) in cfg["windows"].items():
        window_prices = prices.loc[prices.index >= pd.to_datetime(window_start)]
        if window_end is not None:
            window_prices = window_prices.loc[window_prices.index <= pd.to_datetime(window_end)]

        if len(window_prices) < 10:
            continue

        window_execution = build_execution(cfg)
        eq_window, _, _, _ = run_engine(
            cfg=cfg,
            prices=window_prices,
            target_weights=target_weights,
            equity_universe=equity_universe,
            execution=window_execution,
            initial_capital=initial_capital,
        )
        metrics = compute_metrics(eq_window, window_prices[benchmark])
        metrics["Window"] = name
        rows.append(metrics)

    window_summary = pd.DataFrame(rows)
    if not window_summary.empty:
        first_cols = ["Window", "Start", "End", "Weeks"]
        rest = [col for col in window_summary.columns if col not in first_cols]
        window_summary = window_summary[first_cols + rest]
    window_summary.to_csv(csv_dir / "window_summary.csv", index=False)

    holdings_snapshot = final_holdings_snapshot(holdings, equity.iloc[-1], target_weights)
    holdings_snapshot.to_csv(csv_dir / "final_holdings_snapshot.csv", index=False)

    cluster = cluster_exposure_report(
        holdings_snapshot,
        cfg["risk_alerts"]["direct_semi_tickers"],
        cfg["risk_alerts"]["ai_platform_tickers"],
        list(cfg["portfolio"]["diversifiers"].keys()),
    )
    cluster.to_csv(csv_dir / "cluster_exposure.csv", index=False)

    breaches = cap_breach_report(
        holdings_snapshot,
        equity_universe,
        cfg["portfolio"]["diversifiers"],
        float(cfg["rebalance"]["stock_hard_trigger"]),
        float(cfg["rebalance"]["stock_hard_trim_to"]),
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )
    breaches.to_csv(csv_dir / "cap_breach_report.csv", index=False)

    risk_contrib = risk_contribution(prices, target_weights)
    risk_contrib.to_csv(csv_dir / "risk_contribution.csv")

    rolling = rolling_analytics(equity, benchmark_prices, windows=[13, 26, 52, 104, 156, 260])
    rolling.to_csv(csv_dir / "rolling_metrics.csv")

    strategy_comparison = pd.DataFrame()
    if bool(cfg.get("strategy_comparison", {}).get("enabled", False)):
        strategy_comparison = run_strategy_comparison(
            prices=prices,
            benchmark_prices=benchmark_prices,
            target_weights=target_weights,
            equity_universe=equity_universe,
            diversifiers=cfg["portfolio"]["diversifiers"],
            execution_config=cfg["execution"],
            rebalance_config=cfg["rebalance"],
            initial_capital=initial_capital,
            variants=cfg["strategy_comparison"].get("variants", []),
        )
        strategy_comparison.to_csv(csv_dir / "strategy_comparison.csv", index=False)
        print("\n===== STRATEGY COMPARISON =====")
        if not strategy_comparison.empty:
            print(strategy_comparison.to_string(index=False))
        else:
            print("No strategy comparison rows generated.")

    chart_paths = save_main_charts(equity, benchmark_prices, initial_capital, charts_dir)

    if bool(cfg["report"]["export_pdf"]):
        pdf_path = export_pdf(
            output_path=reports_dir / "portfolio_backtest_report.pdf",
            config=cfg,
            chart_paths=chart_paths,
            target_weights=target_weights,
            main_metrics=main_metrics,
            window_summary=window_summary,
            holdings_snapshot=holdings_snapshot,
            cluster_exposure=cluster,
            cap_breach=breaches,
            risk_contrib=risk_contrib,
            strategy_comparison=strategy_comparison,
        )
        print(f"\nPDF report exported: {pdf_path}")

    print("\nDone. Check outputs/ for reports, charts, and CSVs.")


if __name__ == "__main__":
    main()
