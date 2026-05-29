from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ensure_output_dirs, load_config
from src.data.yahoo_provider import unique_preserve_order
from src.data.yahoo_provider import download_prices
from src.universe.market_cap_universe import build_static_equity_universe
from src.strategy.weights import make_target_weights
from src.engine.execution import ExecutionModel
from src.engine.backtest import BacktestEngine
from src.analytics.metrics import compute_metrics, risk_contribution, rolling_analytics
from src.analytics.risk import cap_breach_report, cluster_exposure_report, final_holdings_snapshot
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/strategy_refined_free.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    start = cfg["project"]["start_date"]
    end = cfg["project"].get("end_date")
    initial_capital = float(cfg["project"]["initial_capital"])
    benchmark = cfg["portfolio"]["benchmark"]

    all_seed_tickers = unique_preserve_order(
        cfg["universe"]["anchors"]
        + cfg["universe"]["fallback_mega_caps"]
        + list(cfg["portfolio"]["diversifiers"].keys())
        + [benchmark]
    )

    prices = download_prices(
        all_seed_tickers,
        start=start,
        end=end,
        return_frequency=cfg["data"]["return_frequency"],
        cache_dir=cfg["data"]["cache_dir"],
        cache_prices=cfg["data"]["cache_prices"],
    )

    equity_universe = build_static_equity_universe(
        anchors=cfg["universe"]["anchors"],
        fallback_mega_caps=cfg["universe"]["fallback_mega_caps"],
        available_cols=prices.columns,
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

    execution = ExecutionModel(
        whole_shares=bool(cfg["execution"]["whole_shares"]),
        commission_per_trade=float(cfg["execution"]["commission_per_trade"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]),
        min_trade_dollars=float(cfg["execution"]["min_trade_dollars"]),
    )

    engine = BacktestEngine(
        prices=prices,
        target_weights=target_weights,
        equity_universe=equity_universe,
        diversifiers=cfg["portfolio"]["diversifiers"],
        execution=execution,
        cash_return_annual=float(cfg["execution"]["cash_return_annual"]),
    )

    equity, cash, holdings, log = engine.run(
        initial_capital=initial_capital,
        scheduled_frequency=cfg["rebalance"]["scheduled_frequency"],
        monthly_hard_single_name_check=bool(cfg["rebalance"]["monthly_hard_single_name_check"]),
        stock_hard_trigger=float(cfg["rebalance"]["stock_hard_trigger"]),
        stock_hard_trim_to=float(cfg["rebalance"]["stock_hard_trim_to"]),
        diversifier_hard_triggers={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        diversifier_hard_trim_to={k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )

    main_metrics = compute_metrics(equity, benchmark_prices)
    print_metric_block("MAIN STRATEGY METRICS", main_metrics)

    output_dir = Path(cfg["report"]["output_dir"])
    equity.to_csv(output_dir / "csv" / "portfolio_equity.csv")
    cash.to_csv(output_dir / "csv" / "portfolio_cash.csv")
    holdings.to_csv(output_dir / "csv" / "portfolio_holdings.csv")
    log.to_csv(output_dir / "csv" / "rebalance_log.csv", index=False)

    # Window tests
    rows = []
    for name, (w_start, w_end) in cfg["windows"].items():
        wp = prices.loc[prices.index >= pd.to_datetime(w_start)]
        if w_end is not None:
            wp = wp.loc[wp.index <= pd.to_datetime(w_end)]
        if len(wp) < 10:
            continue

        we = BacktestEngine(
            prices=wp,
            target_weights=target_weights,
            equity_universe=equity_universe,
            diversifiers=cfg["portfolio"]["diversifiers"],
            execution=execution,
            cash_return_annual=float(cfg["execution"]["cash_return_annual"]),
        )
        eq_w, _, _, _ = we.run(
            initial_capital=initial_capital,
            scheduled_frequency=cfg["rebalance"]["scheduled_frequency"],
            monthly_hard_single_name_check=bool(cfg["rebalance"]["monthly_hard_single_name_check"]),
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

    holdings_snapshot = final_holdings_snapshot(holdings, equity.iloc[-1], target_weights)
    holdings_snapshot.to_csv(output_dir / "csv" / "final_holdings_snapshot.csv", index=False)

    direct_semi = cfg["risk_alerts"]["direct_semi_tickers"]
    ai_platform = cfg["risk_alerts"]["ai_platform_tickers"]
    cluster = cluster_exposure_report(holdings_snapshot, direct_semi, ai_platform, list(cfg["portfolio"]["diversifiers"].keys()))
    cluster.to_csv(output_dir / "csv" / "cluster_exposure.csv", index=False)

    breaches = cap_breach_report(
        holdings_snapshot,
        equity_universe,
        cfg["portfolio"]["diversifiers"],
        float(cfg["rebalance"]["stock_hard_trigger"]),
        float(cfg["rebalance"]["stock_hard_trim_to"]),
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_triggers"].items()},
        {k: float(v) for k, v in cfg["rebalance"]["diversifier_hard_trim_to"].items()},
    )
    breaches.to_csv(output_dir / "csv" / "cap_breach_report.csv", index=False)

    risk_contrib = risk_contribution(prices, target_weights)
    risk_contrib.to_csv(output_dir / "csv" / "risk_contribution.csv")

    rolling = rolling_analytics(equity, benchmark_prices, windows=[13, 26, 52, 104, 156, 260])
    rolling.to_csv(output_dir / "csv" / "rolling_metrics.csv")

    chart_paths = save_main_charts(equity, benchmark_prices, initial_capital, output_dir / "charts")

    if bool(cfg["report"]["export_pdf"]):
        pdf_path = export_pdf(
            output_path=output_dir / "reports" / "portfolio_backtest_report.pdf",
            config=cfg,
            chart_paths=chart_paths,
            target_weights=target_weights,
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
