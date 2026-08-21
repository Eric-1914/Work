"""Build Week 8 final analysis, documentation, dashboard, and reproducibility outputs."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.week8_utils import (  # noqa: E402
    REQUIRED_STRATEGIES,
    SCORECARD_METRICS,
    Week8Paths,
    discover_project_modules,
    environment_manifest,
    format_num,
    format_pct,
    markdown_table,
    read_csv,
    read_json,
    require_columns,
    sha256_file,
)

PATHS = Week8Paths(PROJECT_ROOT)

SCORECARD_OUT = PATHS.tables_dir / "week8_final_scorecard.csv"
ROBUSTNESS_OUT = PATHS.tables_dir / "week8_robustness_summary.csv"
ALLOCATION_OUT = PATHS.tables_dir / "week8_latest_allocation.csv"
QUALITY_OUT = PATHS.tables_dir / "week8_data_quality_audit.csv"
MODULES_OUT = PATHS.tables_dir / "week8_module_inventory.csv"
DASHBOARD_OUT = PATHS.dashboard_dir / "week8_dashboard.html"
CONFIG_OUT = PATHS.reporting_dir / "week8_reporting_config.json"
MANIFEST_OUT = PATHS.reporting_dir / "week8_reproducibility_manifest.json"
REPORT_OUT = PATHS.docs_dir / "week8_final_report.md"
PROJECT_DOC_OUT = PATHS.docs_dir / "week8_project_documentation.md"

FIG_CAGR = PATHS.figures_dir / "week8_cagr_comparison.png"
FIG_RISK = PATHS.figures_dir / "week8_risk_adjusted_comparison.png"
FIG_ROLLING = PATHS.figures_dir / "week8_rolling_sharpe.png"
FIG_COST = PATHS.figures_dir / "week8_cost_robustness.png"
FIG_ALLOC = PATHS.figures_dir / "week8_latest_allocation.png"


def load_inputs() -> dict[str, object]:
    summary = read_csv(PATHS.week7_summary, "Week 7 performance summary")
    require_columns(summary, {"Strategy", *SCORECARD_METRICS}, "Week 7 performance summary")
    found = set(summary["Strategy"].astype(str))
    missing = REQUIRED_STRATEGIES - found
    if missing:
        raise ValueError(f"Week 7 performance summary is missing strategies: {sorted(missing)}")

    daily = read_csv(PATHS.week7_daily, "Week 7 daily backtest")
    require_columns(daily, {"Date", "Net_Return", "Equity", "Drawdown"}, "Week 7 daily backtest")
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")

    benchmarks = read_csv(PATHS.week7_benchmarks, "Week 7 benchmark series")
    require_columns(
        benchmarks,
        {"Date", "SPY", "SPY_Equity", "Equal_Weighted_Sector_Return", "Equal_Weighted_Sectors_Equity"},
        "Week 7 benchmark series",
    )
    benchmarks["Date"] = pd.to_datetime(benchmarks["Date"], errors="raise")

    rolling = read_csv(PATHS.week7_rolling, "Week 7 rolling validation")
    require_columns(rolling, {"Window_Start", "Window_End", "Sharpe_0RF", "CAGR", "Max_Drawdown"}, "Week 7 rolling validation")
    rolling["Window_End"] = pd.to_datetime(rolling["Window_End"], errors="raise")

    cost = read_csv(PATHS.week7_cost, "Week 7 cost sensitivity")
    require_columns(cost, {"Total_Friction_Bps_Per_Risky_Dollar_Traded", "Total_Return", "CAGR", "Sharpe_0RF"}, "Week 7 cost sensitivity")

    rebalance = read_csv(PATHS.week7_rebalance, "Week 7 rebalance log")
    require_columns(rebalance, {"Signal_Date", "Execution_Date", "One_Way_Turnover", "Total_Trading_Cost_Fraction"}, "Week 7 rebalance log")

    week6 = read_csv(PATHS.week6_portfolio, "Week 6 optimized portfolio")
    require_columns(week6, {"Signal_Date", "Asset", "Target_Weight"}, "Week 6 optimized portfolio")
    week6["Signal_Date"] = pd.to_datetime(week6["Signal_Date"], errors="raise")

    week7_config = read_json(PATHS.week7_config, "Week 7 backtest config")
    return {
        "summary": summary,
        "daily": daily,
        "benchmarks": benchmarks,
        "rolling": rolling,
        "cost": cost,
        "rebalance": rebalance,
        "week6": week6,
        "week7_config": week7_config,
    }


def build_scorecard(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.set_index("Strategy").loc[
        ["Week7_Strategy_Net", "SPY_Benchmark", "Equal_Weighted_Sectors"], SCORECARD_METRICS
    ]
    rows = []
    for metric in SCORECARD_METRICS:
        strategy = float(ordered.loc["Week7_Strategy_Net", metric])
        spy = float(ordered.loc["SPY_Benchmark", metric])
        equal = float(ordered.loc["Equal_Weighted_Sectors", metric])
        rows.append(
            {
                "Metric": metric,
                "Week7_Strategy_Net": strategy,
                "SPY_Benchmark": spy,
                "Equal_Weighted_Sectors": equal,
                "Strategy_minus_SPY": strategy - spy,
                "Strategy_minus_EqualWeight": strategy - equal,
            }
        )
    return pd.DataFrame(rows)


def build_robustness(rolling: pd.DataFrame, cost: pd.DataFrame, rebalance: pd.DataFrame) -> pd.DataFrame:
    sharpe = pd.to_numeric(rolling["Sharpe_0RF"], errors="coerce").dropna()
    cagr = pd.to_numeric(rolling["CAGR"], errors="coerce").dropna()
    drawdown = pd.to_numeric(rolling["Max_Drawdown"], errors="coerce").dropna()
    friction = cost.sort_values("Total_Friction_Bps_Per_Risky_Dollar_Traded").copy()
    base_idx = (friction["Total_Friction_Bps_Per_Risky_Dollar_Traded"] - 10.0).abs().idxmin()
    zero_idx = friction["Total_Friction_Bps_Per_Risky_Dollar_Traded"].idxmin()
    high_idx = friction["Total_Friction_Bps_Per_Risky_Dollar_Traded"].idxmax()

    total_turnover = float(pd.to_numeric(rebalance["One_Way_Turnover"], errors="coerce").fillna(0.0).sum())
    total_cost = float(pd.to_numeric(rebalance["Total_Trading_Cost_Fraction"], errors="coerce").fillna(0.0).sum())

    rows = [
        {"Measure": "Rolling_Window_Count", "Value": float(len(rolling))},
        {"Measure": "Rolling_Sharpe_Median", "Value": float(sharpe.median())},
        {"Measure": "Rolling_Sharpe_Min", "Value": float(sharpe.min())},
        {"Measure": "Positive_Rolling_Sharpe_Share", "Value": float((sharpe > 0).mean())},
        {"Measure": "Positive_Rolling_CAGR_Share", "Value": float((cagr > 0).mean())},
        {"Measure": "Worst_Rolling_Max_Drawdown", "Value": float(drawdown.min())},
        {"Measure": "Zero_Friction_Total_Return", "Value": float(friction.loc[zero_idx, "Total_Return"])},
        {"Measure": "Base_10bps_Total_Return", "Value": float(friction.loc[base_idx, "Total_Return"])},
        {"Measure": "Highest_Friction_Total_Return", "Value": float(friction.loc[high_idx, "Total_Return"])},
        {
            "Measure": "Return_Drag_Zero_to_Highest_Friction",
            "Value": float(friction.loc[zero_idx, "Total_Return"] - friction.loc[high_idx, "Total_Return"]),
        },
        {"Measure": "Total_One_Way_Turnover", "Value": total_turnover},
        {"Measure": "Total_Trading_Cost_Fraction", "Value": total_cost},
    ]
    return pd.DataFrame(rows)


def build_latest_allocation(week6: pd.DataFrame) -> pd.DataFrame:
    latest_date = week6["Signal_Date"].max()
    latest = week6.loc[week6["Signal_Date"] == latest_date, ["Signal_Date", "Asset", "Target_Weight"]].copy()
    latest["Signal_Date"] = latest["Signal_Date"].dt.strftime("%Y-%m-%d")
    latest["Target_Weight"] = pd.to_numeric(latest["Target_Weight"], errors="raise")
    latest = latest.sort_values(["Target_Weight", "Asset"], ascending=[False, True]).reset_index(drop=True)
    if not np.isclose(latest["Target_Weight"].sum(), 1.0, atol=1e-8):
        raise ValueError("Latest Week 6 allocation does not sum to 100%.")
    return latest


def build_quality_audit(inputs: dict[str, object]) -> pd.DataFrame:
    summary = inputs["summary"]
    daily = inputs["daily"]
    benchmarks = inputs["benchmarks"]
    rolling = inputs["rolling"]
    cost = inputs["cost"]
    rebalance = inputs["rebalance"]
    week6 = inputs["week6"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(benchmarks, pd.DataFrame)
    assert isinstance(rolling, pd.DataFrame)
    assert isinstance(cost, pd.DataFrame)
    assert isinstance(rebalance, pd.DataFrame)
    assert isinstance(week6, pd.DataFrame)

    checks = [
        ("Required_Strategies_Present", REQUIRED_STRATEGIES.issubset(set(summary["Strategy"].astype(str)))),
        ("Daily_Dates_Unique", not daily["Date"].duplicated().any()),
        ("Benchmark_Dates_Unique", not benchmarks["Date"].duplicated().any()),
        ("Daily_and_Benchmark_Date_Range_Match", daily["Date"].min() == benchmarks["Date"].min() and daily["Date"].max() == benchmarks["Date"].max()),
        ("Daily_Equity_Positive", bool((pd.to_numeric(daily["Equity"], errors="coerce") > 0).all())),
        ("Daily_Returns_Finite", bool(np.isfinite(pd.to_numeric(daily["Net_Return"], errors="coerce")).all())),
        ("Rolling_Windows_Available", len(rolling) > 0),
        ("Cost_Sensitivity_Available", len(cost) >= 2),
        ("Rebalance_Log_Available", len(rebalance) > 0),
        ("Week6_Targets_Available", len(week6) > 0),
    ]
    audit = pd.DataFrame(checks, columns=["Check", "Passed"])
    audit["Passed"] = audit["Passed"].astype(bool)
    return audit


def _strategy_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.set_index("Strategy").loc[
        ["Week7_Strategy_Net", "SPY_Benchmark", "Equal_Weighted_Sectors"]
    ].reset_index()


def save_figures(summary: pd.DataFrame, rolling: pd.DataFrame, cost: pd.DataFrame, allocation: pd.DataFrame) -> None:
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = _strategy_rows(summary)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ordered["Strategy"], ordered["CAGR"] * 100.0)
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Week 8 Final CAGR Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_CAGR, dpi=160)
    plt.close(fig)

    positions = np.arange(len(ordered))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(positions - width / 2, ordered["Sharpe_0RF"], width, label="Sharpe")
    ax.bar(positions + width / 2, ordered["Sortino_0MAR"], width, label="Sortino")
    ax.set_xticks(positions, ordered["Strategy"], rotation=15)
    ax.set_ylabel("Ratio")
    ax.set_title("Week 8 Risk-Adjusted Performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_RISK, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pd.to_datetime(rolling["Window_End"]), rolling["Sharpe_0RF"], marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Rolling Sharpe")
    ax.set_title("Week 8 Rolling Sharpe Stability")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_ROLLING, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cost["Total_Friction_Bps_Per_Risky_Dollar_Traded"], cost["Total_Return"] * 100.0, marker="o")
    ax.set_xlabel("Total Friction (bps per risky dollar traded)")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Week 8 Trading-Cost Robustness")
    fig.tight_layout()
    fig.savefig(FIG_COST, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(allocation["Asset"], allocation["Target_Weight"] * 100.0)
    ax.set_ylabel("Target Weight (%)")
    ax.set_title(f"Latest Week 6 Allocation ({allocation['Signal_Date'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(FIG_ALLOC, dpi=160)
    plt.close(fig)


def save_config() -> dict:
    config = {
        "week8_scope": [
            "final_analysis",
            "code_reproducibility_inventory",
            "visual_reporting",
            "static_html_dashboard",
            "module_and_process_documentation",
        ],
        "input_files": {
            "week7_performance_summary": str(PATHS.week7_summary.relative_to(PROJECT_ROOT)),
            "week7_daily_backtest": str(PATHS.week7_daily.relative_to(PROJECT_ROOT)),
            "week7_benchmark_daily": str(PATHS.week7_benchmarks.relative_to(PROJECT_ROOT)),
            "week7_rolling_validation": str(PATHS.week7_rolling.relative_to(PROJECT_ROOT)),
            "week7_cost_sensitivity": str(PATHS.week7_cost.relative_to(PROJECT_ROOT)),
            "week7_rebalance_log": str(PATHS.week7_rebalance.relative_to(PROJECT_ROOT)),
            "week6_optimized_portfolio": str(PATHS.week6_portfolio.relative_to(PROJECT_ROOT)),
        },
        "dashboard": str(DASHBOARD_OUT.relative_to(PROJECT_ROOT)),
        "final_report": str(REPORT_OUT.relative_to(PROJECT_ROOT)),
        "project_documentation": str(PROJECT_DOC_OUT.relative_to(PROJECT_ROOT)),
        "design_note": "Week 8 reports existing Week 6/7 outputs without retuning model or strategy parameters.",
    }
    PATHS.reporting_dir.mkdir(parents=True, exist_ok=True)
    with CONFIG_OUT.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return config


def save_manifest(modules: pd.DataFrame, config: dict) -> dict:
    input_paths = [
        PATHS.week7_summary,
        PATHS.week7_daily,
        PATHS.week7_benchmarks,
        PATHS.week7_rolling,
        PATHS.week7_cost,
        PATHS.week7_rebalance,
        PATHS.week7_config,
        PATHS.week6_portfolio,
    ]
    manifest = {
        "environment": environment_manifest(PROJECT_ROOT),
        "week8_config": config,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in input_paths
        },
        "source_module_count": int(len(modules)),
        "source_sha256": {
            row["Path"]: row["SHA256"] for _, row in modules.iterrows()
        },
        "reproduction_command": "python3 run_week8.py",
    }
    with MANIFEST_OUT.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _metric_lookup(scorecard: pd.DataFrame, metric: str, strategy_col: str = "Week7_Strategy_Net") -> float:
    row = scorecard.loc[scorecard["Metric"] == metric]
    if row.empty:
        raise ValueError(f"Missing scorecard metric: {metric}")
    return float(row.iloc[0][strategy_col])


def save_final_report(scorecard: pd.DataFrame, robustness: pd.DataFrame, allocation: pd.DataFrame, quality: pd.DataFrame) -> None:
    cagr = _metric_lookup(scorecard, "CAGR")
    vol = _metric_lookup(scorecard, "Annualized_Volatility")
    sharpe = _metric_lookup(scorecard, "Sharpe_0RF")
    sortino = _metric_lookup(scorecard, "Sortino_0MAR")
    drawdown = _metric_lookup(scorecard, "Max_Drawdown")
    total_return = _metric_lookup(scorecard, "Total_Return")
    spy_cagr_delta = float(scorecard.loc[scorecard["Metric"] == "CAGR", "Strategy_minus_SPY"].iloc[0])
    equal_cagr_delta = float(scorecard.loc[scorecard["Metric"] == "CAGR", "Strategy_minus_EqualWeight"].iloc[0])
    robustness_map = dict(zip(robustness["Measure"], robustness["Value"]))

    report = f"""# Week 8: Final Analysis & Reporting

## Scope

Week 8 consolidates the validated Week 6 portfolio and Week 7 backtest into a reproducible reporting layer. It does not retrain models or retune strategy parameters after observing final performance.

## Final strategy scorecard

- Total return: {format_pct(total_return)}
- CAGR: {format_pct(cagr)}
- Annualized volatility: {format_pct(vol)}
- Sharpe ratio: {format_num(sharpe)}
- Sortino ratio: {format_num(sortino)}
- Maximum drawdown: {format_pct(drawdown)}
- CAGR difference vs. SPY: {format_pct(spy_cagr_delta)}
- CAGR difference vs. equal-weight sectors: {format_pct(equal_cagr_delta)}

{markdown_table(scorecard, float_digits=4)}

## Robustness checks

- Rolling windows evaluated: {int(robustness_map['Rolling_Window_Count'])}
- Median rolling Sharpe: {format_num(robustness_map['Rolling_Sharpe_Median'])}
- Share of rolling windows with positive Sharpe: {format_pct(robustness_map['Positive_Rolling_Sharpe_Share'])}
- Share of rolling windows with positive CAGR: {format_pct(robustness_map['Positive_Rolling_CAGR_Share'])}
- Worst rolling maximum drawdown: {format_pct(robustness_map['Worst_Rolling_Max_Drawdown'])}
- Return drag from zero friction to the highest tested friction: {format_pct(robustness_map['Return_Drag_Zero_to_Highest_Friction'])}

{markdown_table(robustness, float_digits=4)}

## Latest optimized allocation

{markdown_table(allocation, float_digits=4)}

## Data-quality status

All Week 8 reporting outputs are generated only after the required Week 6/7 inputs pass structural checks. Current audit result: **{'PASS' if quality['Passed'].all() else 'FAIL'}**.

{markdown_table(quality, float_digits=4)}

## Reporting outputs

- Static dashboard: `results/dashboard/week8_dashboard.html`
- Final scorecard: `results/tables/week8_final_scorecard.csv`
- Robustness summary: `results/tables/week8_robustness_summary.csv`
- Latest allocation: `results/tables/week8_latest_allocation.csv`
- Module inventory: `results/tables/week8_module_inventory.csv`
- Reproducibility manifest: `data/reporting/week8_reproducibility_manifest.json`
- Project process documentation: `docs/week8_project_documentation.md`

## Interpretation note

Week 8 is a reporting and reproducibility layer. It summarizes already-produced out-of-sample/backtest results and deliberately avoids changing model, allocation, risk, or backtest parameters after seeing final performance.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def save_project_documentation(modules: pd.DataFrame) -> None:
    week_counts = modules.groupby(["Week", "Category"], dropna=False).size().reset_index(name="Files")
    process = """# Week 8: Project Documentation

## End-to-end process

1. **Week 1 — Data infrastructure:** collect and store sector and macroeconomic data.
2. **Week 2 — Feature engineering:** transform raw data into model-ready technical and macro features.
3. **Week 3 — Baseline modeling:** define the outperformance target, train baseline classifiers, and evaluate with time-aware splits.
4. **Week 4 — Robust model development:** add ensemble/regime-aware models, sector ranking, and model selection.
5. **Week 5 — Strategy construction:** translate model probabilities into sector signals, target weights, and monthly rebalancing instructions.
6. **Week 6 — Risk management:** apply portfolio constraints, MVO/inverse-volatility logic, volatility targeting, turnover controls, and defensive overlays.
7. **Week 7 — Backtesting:** apply conservative execution timing, transaction costs/slippage, performance metrics, benchmark comparisons, rolling validation, and cost sensitivity.
8. **Week 8 — Final reporting:** consolidate validated outputs, document the codebase, create reproducibility metadata, and generate visual reports/dashboard outputs.

## Reproducibility

The Week 8 entry point is:

```bash
python3 run_week8.py
```

The command builds all Week 8 reports and then independently validates them. Week 8 does not retrain or retune earlier stages.

## Detected project modules

"""
    process += markdown_table(modules[["Path", "Week", "Category", "Lines"]], float_digits=0)
    process += "\n\n## Module counts by stage\n\n"
    process += markdown_table(week_counts, float_digits=0)
    process += "\n\nExact SHA-256 hashes are stored in `results/tables/week8_module_inventory.csv` and `data/reporting/week8_reproducibility_manifest.json`.\n"
    PROJECT_DOC_OUT.write_text(process, encoding="utf-8")


def _html_table(frame: pd.DataFrame, percent_metrics: set[str] | None = None) -> str:
    percent_metrics = percent_metrics or set()
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        metric = str(row.get("Metric", ""))
        for col, value in row.items():
            if isinstance(value, (float, np.floating)) and not pd.isna(value):
                if metric in percent_metrics and col != "Metric":
                    text = f"{float(value):.2%}"
                else:
                    text = f"{float(value):.4f}"
            else:
                text = str(value)
            cells.append(f"<td>{escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def save_dashboard(scorecard: pd.DataFrame, robustness: pd.DataFrame, allocation: pd.DataFrame, quality: pd.DataFrame) -> None:
    PATHS.dashboard_dir.mkdir(parents=True, exist_ok=True)
    cagr = _metric_lookup(scorecard, "CAGR")
    sharpe = _metric_lookup(scorecard, "Sharpe_0RF")
    sortino = _metric_lookup(scorecard, "Sortino_0MAR")
    drawdown = _metric_lookup(scorecard, "Max_Drawdown")
    quality_status = "PASS" if quality["Passed"].all() else "FAIL"
    pct_metrics = {"Total_Return", "CAGR", "Annualized_Volatility", "Max_Drawdown", "Daily_Win_Rate"}

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Week 8 Final Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f4f6f8; color: #202124; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1, h2 {{ margin-bottom: 10px; }}
.note {{ color: #5f6368; margin-bottom: 24px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0 28px; }}
.card {{ background: white; border: 1px solid #dfe3e8; border-radius: 10px; padding: 16px; }}
.card .label {{ font-size: 13px; color: #5f6368; }}
.card .value {{ font-size: 26px; font-weight: 650; margin-top: 5px; }}
.panel {{ background: white; border: 1px solid #dfe3e8; border-radius: 10px; padding: 18px; margin: 16px 0; overflow-x: auto; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 16px; }}
img {{ width: 100%; height: auto; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.status {{ font-weight: 700; }}
</style>
</head>
<body><main>
<h1>Week 8 — Final Analysis Dashboard</h1>
<p class="note">Final reporting layer built from validated Week 6 portfolio targets and Week 7 backtest outputs. No model or strategy retuning is performed here.</p>
<div class="cards">
<div class="card"><div class="label">CAGR</div><div class="value">{format_pct(cagr)}</div></div>
<div class="card"><div class="label">Sharpe</div><div class="value">{format_num(sharpe)}</div></div>
<div class="card"><div class="label">Sortino</div><div class="value">{format_num(sortino)}</div></div>
<div class="card"><div class="label">Max Drawdown</div><div class="value">{format_pct(drawdown)}</div></div>
<div class="card"><div class="label">Data Quality</div><div class="value status">{quality_status}</div></div>
</div>
<div class="panel"><h2>Final Scorecard</h2>{_html_table(scorecard, pct_metrics)}</div>
<div class="grid">
<div class="panel"><h2>CAGR Comparison</h2><img src="../figures/{FIG_CAGR.name}" alt="CAGR comparison"></div>
<div class="panel"><h2>Risk-Adjusted Comparison</h2><img src="../figures/{FIG_RISK.name}" alt="Risk-adjusted comparison"></div>
<div class="panel"><h2>Rolling Sharpe</h2><img src="../figures/{FIG_ROLLING.name}" alt="Rolling Sharpe"></div>
<div class="panel"><h2>Cost Robustness</h2><img src="../figures/{FIG_COST.name}" alt="Cost robustness"></div>
<div class="panel"><h2>Latest Allocation</h2><img src="../figures/{FIG_ALLOC.name}" alt="Latest allocation"></div>
</div>
<div class="panel"><h2>Latest Allocation Table</h2>{_html_table(allocation)}</div>
<div class="panel"><h2>Robustness Summary</h2>{_html_table(robustness)}</div>
<div class="panel"><h2>Data Quality Audit</h2>{_html_table(quality)}</div>
</main></body></html>"""
    DASHBOARD_OUT.write_text(html, encoding="utf-8")


def save_tables(scorecard: pd.DataFrame, robustness: pd.DataFrame, allocation: pd.DataFrame, quality: pd.DataFrame, modules: pd.DataFrame) -> None:
    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(SCORECARD_OUT, index=False)
    robustness.to_csv(ROBUSTNESS_OUT, index=False)
    allocation.to_csv(ALLOCATION_OUT, index=False)
    quality.to_csv(QUALITY_OUT, index=False)
    modules.to_csv(MODULES_OUT, index=False)


def main() -> int:
    try:
        PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
        PATHS.dashboard_dir.mkdir(parents=True, exist_ok=True)
        PATHS.reporting_dir.mkdir(parents=True, exist_ok=True)
        PATHS.docs_dir.mkdir(parents=True, exist_ok=True)

        inputs = load_inputs()
        summary = inputs["summary"]
        rolling = inputs["rolling"]
        cost = inputs["cost"]
        rebalance = inputs["rebalance"]
        week6 = inputs["week6"]
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(rolling, pd.DataFrame)
        assert isinstance(cost, pd.DataFrame)
        assert isinstance(rebalance, pd.DataFrame)
        assert isinstance(week6, pd.DataFrame)

        scorecard = build_scorecard(summary)
        robustness = build_robustness(rolling, cost, rebalance)
        allocation = build_latest_allocation(week6)
        quality = build_quality_audit(inputs)
        if not quality["Passed"].all():
            failed = quality.loc[~quality["Passed"], "Check"].tolist()
            raise ValueError(f"Week 8 data-quality audit failed: {failed}")

        modules = discover_project_modules(PROJECT_ROOT)
        if modules.empty:
            raise ValueError("No project modules were discovered for the Week 8 reproducibility inventory.")

        save_tables(scorecard, robustness, allocation, quality, modules)
        save_figures(summary, rolling, cost, allocation)
        config = save_config()
        save_manifest(modules, config)
        save_final_report(scorecard, robustness, allocation, quality)
        save_project_documentation(modules)
        save_dashboard(scorecard, robustness, allocation, quality)

        print(f"[PASS] Consolidated {len(summary)} Week 7 strategy/benchmark rows into a final scorecard.")
        print(f"[PASS] Summarized {len(rolling)} rolling windows and {len(cost)} trading-cost scenarios.")
        print(f"[PASS] Captured the latest Week 6 allocation and {len(modules)} detected project modules.")
        print("[PASS] Generated static dashboard, final report, project documentation, and reproducibility manifest.")
        print("[PASS] Week 8 reporting layer did not retune model, strategy, risk, or backtest parameters.")
        print("[DONE] Week 8 final analysis and reporting assets completed successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 8 reporting failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
