from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def fmt_pct(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def fmt_money(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.{decimals}f}"


def metric_value(key, value):
    if isinstance(value, float):
        k = str(key).lower()
        if "beta" in k:
            return fmt_num(value, 4)
        if "return" in k or "volatility" in k or "drawdown" in k or "correlation" in k or "weight" in k:
            return fmt_pct(value)
        if "equity" in k or "value" in k:
            return fmt_money(value)
        return fmt_num(value, 4)
    return str(value)


def make_table(data, font_size=8, col_widths=None):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    return table


def df_to_table(df, columns=None, pct_cols=None, money_cols=None, max_rows=30, decimals=2):
    if df is None or df.empty:
        return [["No data available"]]

    temp = df.copy()
    if columns is not None:
        temp = temp[[c for c in columns if c in temp.columns]]
    temp = temp.head(max_rows)

    pct_cols = pct_cols or []
    money_cols = money_cols or []

    data = [list(temp.columns)]
    for _, row in temp.iterrows():
        out = []
        for col in temp.columns:
            val = row[col]
            if col in pct_cols and pd.notna(val):
                out.append(fmt_pct(val, decimals))
            elif col in money_cols and pd.notna(val):
                out.append(fmt_money(val, decimals))
            elif isinstance(val, float):
                out.append(fmt_num(val, decimals))
            else:
                out.append(str(val))
        data.append(out)
    return data


def universe_summary_table(universe_membership: pd.DataFrame):
    if universe_membership is None or universe_membership.empty:
        return pd.DataFrame()

    df = universe_membership.copy()
    rows = []
    for freq, group in df.groupby("ranking_frequency", dropna=False):
        rows.append(
            {
                "Ranking Frequency": freq,
                "Ranking Dates": group["ranking_date"].nunique(),
                "Rows": len(group),
                "Unique Tickers": group["ticker"].nunique(),
                "First Ranking Date": group["ranking_date"].min(),
                "Last Ranking Date": group["ranking_date"].max(),
                "Anchor Rows": int(group["is_anchor"].sum()),
            }
        )
    return pd.DataFrame(rows)


def export_pdf(
    output_path: str | Path,
    config: dict,
    chart_paths: dict,
    target_weights: dict,
    main_metrics: dict,
    window_summary: pd.DataFrame,
    holdings_snapshot: pd.DataFrame,
    cluster_exposure: pd.DataFrame,
    cap_breach: pd.DataFrame,
    risk_contrib: pd.DataFrame,
    strategy_comparison: pd.DataFrame | None = None,
    universe_membership: pd.DataFrame | None = None,
    pit_download_candidates: pd.DataFrame | None = None,
    ranking_frequency_comparison: pd.DataFrame | None = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=22, leading=26)
    h1 = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=15, leading=18)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9, leading=12)

    story = []
    story.append(Paragraph("Portfolio Backtest Report", title))
    story.append(Paragraph(config["project"]["name"], body))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    story.append(Spacer(1, 0.2 * inch))

    uni_cfg = config.get("universe", {})
    settings = [
        ["Setting", "Value"],
        ["Initial Capital", fmt_money(config["project"]["initial_capital"])],
        ["Start Date", config["project"]["start_date"]],
        ["Whole Shares", str(config["execution"]["whole_shares"])],
        ["Slippage BPS", fmt_num(config["execution"]["slippage_bps"])],
        ["Scheduled Rebalance", config["rebalance"]["scheduled_frequency"]],
        ["Hard Cap Check Frequency", config["rebalance"]["hard_cap_check_frequency"]],
        ["Stock Hard Trigger", fmt_pct(config["rebalance"]["stock_hard_trigger"])],
        ["Stock Trim To", fmt_pct(config["rebalance"]["stock_hard_trim_to"])],
        ["PIT Market Caps Enabled", str(uni_cfg.get("use_point_in_time_market_caps", False))],
        ["PIT Ranking Frequency", str(uni_cfg.get("point_in_time_ranking_frequency", "N/A"))],
        ["PIT Candidate Frequencies", str(uni_cfg.get("point_in_time_candidate_frequencies", "N/A"))],
        ["PIT Download Top N", str(uni_cfg.get("point_in_time_download_top_n", "N/A"))],
        ["PIT Asof Lag Days", str(uni_cfg.get("point_in_time_asof_lag_days", "N/A"))],
        ["PIT Min Market Cap", str(uni_cfg.get("point_in_time_min_market_cap", "N/A"))],
    ]
    story.append(make_table(settings, font_size=8, col_widths=[2.7 * inch, 6.3 * inch]))

    if universe_membership is not None and not universe_membership.empty:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Point-in-Time Universe Summary", h1))
        usum = universe_summary_table(universe_membership)
        story.append(make_table(df_to_table(usum, max_rows=10), font_size=7.5))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Target Weights", h1))
    weights_df = pd.DataFrame([{"Ticker": t, "Target Weight": w} for t, w in target_weights.items()])
    story.append(make_table(df_to_table(weights_df, pct_cols=["Target Weight"], max_rows=40), font_size=8))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Main Metrics", h1))
    metric_rows = [["Metric", "Value"]] + [[k, metric_value(k, v)] for k, v in main_metrics.items()]
    story.append(make_table(metric_rows, font_size=8, col_widths=[3.0 * inch, 2.0 * inch]))

    if ranking_frequency_comparison is not None and not ranking_frequency_comparison.empty:
        story.append(PageBreak())
        story.append(Paragraph("PIT Ranking Frequency Comparison", h1))
        cols = [
            "Ranking Frequency",
            "Ranking Dates",
            "Unique Tickers Used",
            "Rebalance Count",
            "Hard Cap Count",
            "Final Equity",
            "Annual Return / CAGR",
            "Annual Volatility",
            "Sharpe",
            "Max Drawdown",
            "Calmar",
            "Worst 52W Return",
            "Beta vs SPY",
        ]
        story.append(
            make_table(
                df_to_table(
                    ranking_frequency_comparison,
                    columns=cols,
                    pct_cols=["Annual Return / CAGR", "Annual Volatility", "Max Drawdown", "Worst 52W Return"],
                    money_cols=["Final Equity"],
                    max_rows=10,
                ),
                font_size=5.8,
            )
        )

    if strategy_comparison is not None and not strategy_comparison.empty:
        story.append(PageBreak())
        story.append(Paragraph("Strategy Comparison", h1))
        comp_cols = [
            "Strategy",
            "Scheduled Frequency",
            "Hard Cap Frequency",
            "Rebalance Count",
            "Hard Cap Count",
            "Final Equity",
            "Annual Return / CAGR",
            "Annual Volatility",
            "Sharpe",
            "Max Drawdown",
            "Calmar",
            "Worst 52W Return",
            "Beta vs SPY",
        ]
        story.append(
            make_table(
                df_to_table(
                    strategy_comparison,
                    columns=comp_cols,
                    pct_cols=["Annual Return / CAGR", "Annual Volatility", "Max Drawdown", "Worst 52W Return"],
                    money_cols=["Final Equity"],
                    max_rows=20,
                ),
                font_size=5.5,
            )
        )

    if universe_membership is not None and not universe_membership.empty:
        story.append(PageBreak())
        story.append(Paragraph("Point-in-Time Universe Membership Sample", h1))
        story.append(Paragraph("This table shows the first rows from the selected PIT ranking audit file.", body))
        membership_cols = [
            "ranking_frequency",
            "ranking_date",
            "asof_date",
            "ticker",
            "slot",
            "is_anchor",
            "market_cap_rank",
            "market_cap",
            "market_cap_observation_date",
        ]
        story.append(
            make_table(
                df_to_table(
                    universe_membership,
                    columns=membership_cols,
                    money_cols=["market_cap"],
                    max_rows=35,
                ),
                font_size=6.0,
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Equity Curve and Drawdown", h1))
    if "equity_curve" in chart_paths:
        story.append(Image(chart_paths["equity_curve"], width=9.5 * inch, height=4.2 * inch))
    story.append(Spacer(1, 0.15 * inch))
    if "drawdown" in chart_paths:
        story.append(Image(chart_paths["drawdown"], width=9.5 * inch, height=3.4 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Window Summary", h1))
    story.append(make_table(
        df_to_table(
            window_summary,
            pct_cols=[
                "Annual Return / CAGR",
                "Annual Volatility",
                "Max Drawdown",
                "Worst 52W Return",
                "Correlation vs SPY",
                "SPY Annual Return",
                "SPY Annual Volatility",
                "SPY Max Drawdown",
            ],
            max_rows=20,
        ),
        font_size=6.5,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Final Holdings", h1))
    story.append(make_table(
        df_to_table(
            holdings_snapshot,
            columns=["Ticker", "Target Weight", "Final Weight", "Drift From Target", "Final Shares", "Final Value"],
            pct_cols=["Target Weight", "Final Weight", "Drift From Target"],
            money_cols=["Final Value"],
            max_rows=40,
        ),
        font_size=7.0,
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Cluster Exposure", h1))
    story.append(make_table(
        df_to_table(
            cluster_exposure,
            pct_cols=["Target Weight", "Final Weight", "Drift From Target"],
            max_rows=10,
        ),
        font_size=7.5,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Cap Breach Report", h1))
    story.append(make_table(
        df_to_table(
            cap_breach,
            pct_cols=["Target Weight", "Final Weight", "Hard Trigger", "Trim-To Weight"],
            max_rows=40,
        ),
        font_size=7.0,
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Risk Contribution", h1))
    story.append(make_table(
        df_to_table(
            risk_contrib.reset_index().rename(columns={"index": "Ticker"}),
            pct_cols=["Weight Used", "Risk Contribution %"],
            max_rows=40,
        ),
        font_size=7.0,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Notes", h1))
    notes = [
        "PIT mode uses a local WRDS/CRSP-derived point-in-time market-cap file when enabled.",
        "The selected PIT ranking frequency controls how often the stock universe rotates.",
        "M, Q, SA, and Y ranking frequencies can be tested through ranking_frequency_comparison.csv.",
        "Cluster caps are report-only alerts in this version; only hard single-name/diversifier caps actively trade.",
        "This version still uses yfinance prices. A future CRSP-return mode should use CRSP returns and delisting returns.",
        "Do not commit raw or derived WRDS data to GitHub. Keep WRDS files local/private and ignored by Git.",
        "Results are backtests and can be overstated by data limitations, taxes, and survivorship bias.",
    ]
    for note in notes:
        story.append(Paragraph(f"- {note}", body))

    doc.build(story)
    return output_path
