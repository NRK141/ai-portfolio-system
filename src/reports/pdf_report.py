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
        if "return" in k or "volatility" in k or "drawdown" in k or "correlation" in k:
            return fmt_pct(value)
        if "equity" in k:
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

    settings = [
        ["Setting", "Value"],
        ["Initial Capital", fmt_money(config["project"]["initial_capital"])],
        ["Start Date", config["project"]["start_date"]],
        ["Whole Shares", str(config["execution"]["whole_shares"])],
        ["Slippage BPS", fmt_num(config["execution"]["slippage_bps"])],
        ["Annual Rebalance", config["rebalance"]["scheduled_frequency"]],
        ["Monthly Hard Cap", str(config["rebalance"]["monthly_hard_single_name_check"])],
        ["Stock Hard Trigger", fmt_pct(config["rebalance"]["stock_hard_trigger"])],
        ["Stock Trim To", fmt_pct(config["rebalance"]["stock_hard_trim_to"])],
    ]
    story.append(make_table(settings, font_size=8, col_widths=[2.5 * inch, 6.5 * inch]))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Target Weights", h1))
    weights_df = pd.DataFrame([{"Ticker": t, "Target Weight": w} for t, w in target_weights.items()])
    story.append(make_table(df_to_table(weights_df, pct_cols=["Target Weight"], max_rows=30), font_size=8))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Main Metrics", h1))
    metric_rows = [["Metric", "Value"]] + [[k, metric_value(k, v)] for k, v in main_metrics.items()]
    story.append(make_table(metric_rows, font_size=8, col_widths=[3.0 * inch, 2.0 * inch]))

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
            max_rows=30,
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
            max_rows=30,
        ),
        font_size=7.0,
    ))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Risk Contribution", h1))
    story.append(make_table(
        df_to_table(
            risk_contrib.reset_index().rename(columns={"index": "Ticker"}),
            pct_cols=["Weight Used", "Risk Contribution %"],
            max_rows=30,
        ),
        font_size=7.0,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Notes", h1))
    notes = [
        "This free version uses current mega-cap candidates unless you provide a point-in-time market-cap CSV.",
        "Point-in-time market caps are the biggest remaining accuracy upgrade.",
        "Cluster caps are report-only alerts in this version; only monthly hard single-name caps actively trade.",
        "Results are backtests and can be overstated by data limitations, taxes, and survivorship bias.",
    ]
    for note in notes:
        story.append(Paragraph(f"- {note}", body))

    doc.build(story)
    return output_path
