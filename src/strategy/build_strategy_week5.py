"""Week 5 - Strategy Design & Signal Processing.

This module converts the leakage-safe Week 4 selected-model probabilities into
long-only monthly sector-rotation portfolios. It implements the three requested
portfolio-construction approaches (Top-N, threshold filtering, and score
weighting) and a combined default strategy that uses all three.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "results" / "tables" / "week4_selected_model_signals.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "data" / "strategy"
CONFIG_PATH = CONFIG_DIR / "week5_strategy_config.json"

DATE_COL = "Date"
SECTOR_COL = "Sector"
PROB_COL = "Outperformance_Probability"
RANK_COL = "Sector_Rank"
RANK_SCORE_COL = "Rank_Score_0_1"
REGIME_COL = "Macro_Regime"
MODEL_COL = "model"
PREDICTED_CLASS_COL = "Predicted_Class"

EXPECTED_SECTORS = ("XLF", "XLK", "XLE")
CASH_ASSET = "CASH"
TOP_N = 2
PROBABILITY_THRESHOLD = 0.50
MAX_SECTOR_WEIGHT = 0.60
TOLERANCE = 1e-10

STRATEGY_TOP_N = "Top_N_Equal_Weight"
STRATEGY_THRESHOLD = "Threshold_Equal_Weight"
STRATEGY_SCORE = "Probability_Score_Weighted"
STRATEGY_DEFAULT = "Combined_Default"
STRATEGIES = (
    STRATEGY_TOP_N,
    STRATEGY_THRESHOLD,
    STRATEGY_SCORE,
    STRATEGY_DEFAULT,
)


REQUIRED_SIGNAL_COLUMNS = [
    DATE_COL,
    SECTOR_COL,
    REGIME_COL,
    MODEL_COL,
    PROB_COL,
    PREDICTED_CLASS_COL,
    RANK_COL,
    RANK_SCORE_COL,
]


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Week 4 signal file is missing required columns: {missing}")


def load_week4_signals(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the Week 4 signal handoff used by Week 5."""
    if not path.exists():
        raise FileNotFoundError(
            "results/tables/week4_selected_model_signals.csv is missing. Complete Week 4 first."
        )

    data = pd.read_csv(path, parse_dates=[DATE_COL])
    require_columns(data, REQUIRED_SIGNAL_COLUMNS)
    if data.empty:
        raise ValueError("week4_selected_model_signals.csv is empty.")
    if data[DATE_COL].isna().any():
        raise ValueError("Week 4 signal file contains an invalid or missing Date value.")

    forbidden = {"Target_Outperform", "Relative_Return_21D"}
    if forbidden.intersection(data.columns):
        raise ValueError(
            "Week 5 requires the Week 4 signal handoff without realized forward outcomes."
        )

    # Normalize and validate numeric fields before any sorting or ranking checks.
    probabilities = pd.to_numeric(data[PROB_COL], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities.to_numpy(dtype=float)).all():
        raise ValueError("Outperformance_Probability must contain finite numeric values.")
    if not probabilities.between(0, 1).all():
        raise ValueError("Outperformance_Probability must be inside [0, 1].")
    data[PROB_COL] = probabilities.astype(float)

    ranks = pd.to_numeric(data[RANK_COL], errors="coerce")
    if ranks.isna().any() or not np.isfinite(ranks.to_numpy(dtype=float)).all():
        raise ValueError("Sector_Rank must contain finite numeric values.")
    if not np.allclose(ranks.to_numpy(dtype=float), np.round(ranks.to_numpy(dtype=float)), atol=1e-12):
        raise ValueError("Sector_Rank must contain integer ranks.")
    data[RANK_COL] = ranks.astype(int)

    predicted = pd.to_numeric(data[PREDICTED_CLASS_COL], errors="coerce")
    if predicted.isna().any() or not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError("Predicted_Class must contain finite numeric values.")
    if not np.allclose(
        predicted.to_numpy(dtype=float),
        np.round(predicted.to_numpy(dtype=float)),
        atol=1e-12,
    ):
        raise ValueError("Predicted_Class must contain integer class labels.")
    predicted = predicted.astype(int)
    if not set(predicted.unique()).issubset({0, 1}):
        raise ValueError("Predicted_Class must contain only 0 and 1.")
    expected_predicted = (data[PROB_COL] >= PROBABILITY_THRESHOLD).astype(int)
    if not np.array_equal(predicted.to_numpy(), expected_predicted.to_numpy()):
        raise ValueError(
            f"Predicted_Class is inconsistent with the {PROBABILITY_THRESHOLD:.2f} Week 4 decision threshold."
        )
    data[PREDICTED_CLASS_COL] = predicted

    scores = pd.to_numeric(data[RANK_SCORE_COL], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Rank_Score_0_1 must contain finite numeric values.")
    if not scores.between(0, 1).all():
        raise ValueError("Rank_Score_0_1 must contain values in [0, 1].")
    expected_scores = 1.0 - (data[RANK_COL].astype(float) - 1.0) / 2.0
    if not np.allclose(scores.to_numpy(dtype=float), expected_scores.to_numpy(dtype=float), atol=1e-12):
        raise ValueError("Rank_Score_0_1 is inconsistent with Sector_Rank.")
    data[RANK_SCORE_COL] = scores.astype(float)

    data[SECTOR_COL] = data[SECTOR_COL].astype(str).str.strip()
    data = data.sort_values([DATE_COL, RANK_COL, SECTOR_COL]).reset_index(drop=True)
    if data[[DATE_COL, SECTOR_COL]].duplicated().any():
        raise ValueError("Duplicate Date/Sector rows found in Week 4 selected-model signals.")

    actual_sectors = set(data[SECTOR_COL].unique())
    if actual_sectors != set(EXPECTED_SECTORS):
        raise ValueError(
            f"Week 5 expects sectors {list(EXPECTED_SECTORS)}, found {sorted(actual_sectors)}."
        )

    for date, group in data.groupby(DATE_COL, sort=True):
        if set(group[SECTOR_COL]) != set(EXPECTED_SECTORS) or len(group) != len(EXPECTED_SECTORS):
            raise ValueError(f"Signal date {date.date()} must contain exactly XLF, XLK, and XLE.")
        if group[REGIME_COL].isna().any() or group[REGIME_COL].astype(str).str.strip().eq("").any():
            raise ValueError(f"Signal date {date.date()} contains a missing Macro_Regime value.")
        if group[REGIME_COL].nunique(dropna=False) != 1:
            raise ValueError(f"Signal date {date.date()} contains inconsistent Macro_Regime values.")
        if group[MODEL_COL].isna().any() or group[MODEL_COL].astype(str).str.strip().eq("").any():
            raise ValueError(f"Signal date {date.date()} contains a missing selected-model label.")
        if group[MODEL_COL].nunique(dropna=False) != 1:
            raise ValueError(f"Signal date {date.date()} contains inconsistent selected-model labels.")

        if set(group[RANK_COL]) != {1, 2, 3}:
            raise ValueError(f"Signal date {date.date()} must contain sector ranks 1, 2, and 3.")

        ordered = group.sort_values([PROB_COL, RANK_COL, SECTOR_COL], ascending=[False, True, True])
        if not ordered[RANK_COL].is_monotonic_increasing:
            raise ValueError(f"Sector ranks are inconsistent with model probabilities on {date.date()}.")

    if data[DATE_COL].nunique() < 2:
        raise ValueError("Week 5 requires at least two monthly signal dates.")
    return data


def cap_and_redistribute(
    raw_weights: dict[str, float],
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
) -> dict[str, float]:
    """Normalize long-only raw scores, cap sector weights, and leave residual in cash.

    If the selected set is too small to absorb 100% exposure under the position cap,
    the unallocated portion remains in CASH rather than adding leverage or forcing a
    lower-conviction sector into the portfolio.
    """
    if not (0 < max_sector_weight <= 1):
        raise ValueError("max_sector_weight must be in (0, 1].")

    clean = {
        sector: max(0.0, float(raw_weights.get(sector, 0.0)))
        for sector in EXPECTED_SECTORS
    }
    total = sum(clean.values())
    if total <= TOLERANCE:
        return {**{sector: 0.0 for sector in EXPECTED_SECTORS}, CASH_ASSET: 1.0}

    active = {sector for sector, value in clean.items() if value > TOLERANCE}
    fixed: dict[str, float] = {}

    while active:
        fixed_sum = sum(fixed.values())
        budget = max(0.0, 1.0 - fixed_sum)
        active_raw_total = sum(clean[sector] for sector in active)
        if active_raw_total <= TOLERANCE or budget <= TOLERANCE:
            break

        proposed = {
            sector: clean[sector] / active_raw_total * budget
            for sector in active
        }
        offenders = [sector for sector, value in proposed.items() if value > max_sector_weight + TOLERANCE]
        if not offenders:
            for sector, value in proposed.items():
                fixed[sector] = value
            active.clear()
            break

        for sector in offenders:
            fixed[sector] = max_sector_weight
            active.remove(sector)

    final_sector_weights = {
        sector: min(max_sector_weight, max(0.0, fixed.get(sector, 0.0)))
        for sector in EXPECTED_SECTORS
    }
    invested = sum(final_sector_weights.values())
    cash = max(0.0, 1.0 - invested)
    if cash < TOLERANCE:
        cash = 0.0

    # Keep any meaningful residual in cash; tiny numerical residue is suppressed.
    if cash > 0.0 and abs((invested + cash) - 1.0) > 1e-12:
        cash = 1.0 - invested
    return {**final_sector_weights, CASH_ASSET: cash}


def raw_weights_for_strategy(month: pd.DataFrame, strategy: str) -> dict[str, float]:
    """Construct raw sector scores for one requested portfolio rule."""
    raw = {sector: 0.0 for sector in EXPECTED_SECTORS}
    month = month.sort_values([RANK_COL, SECTOR_COL]).copy()

    if strategy == STRATEGY_TOP_N:
        selected = month[month[RANK_COL].astype(int) <= TOP_N]
        for sector in selected[SECTOR_COL].astype(str):
            raw[sector] = 1.0
        return raw

    if strategy == STRATEGY_THRESHOLD:
        selected = month[month[PROB_COL] >= PROBABILITY_THRESHOLD]
        for sector in selected[SECTOR_COL].astype(str):
            raw[sector] = 1.0
        return raw

    if strategy == STRATEGY_SCORE:
        for _, row in month.iterrows():
            raw[str(row[SECTOR_COL])] = float(row[PROB_COL])
        return raw

    if strategy == STRATEGY_DEFAULT:
        selected = month[
            (month[PROB_COL] >= PROBABILITY_THRESHOLD)
            & (month[RANK_COL].astype(int) <= TOP_N)
        ]
        for _, row in selected.iterrows():
            # Probability itself is the sizing score. Thresholding and Top-N
            # determine eligibility; score weighting determines relative size.
            raw[str(row[SECTOR_COL])] = float(row[PROB_COL])
        return raw

    raise ValueError(f"Unknown Week 5 strategy: {strategy}")


def build_portfolio_weights(signals: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Build monthly target weights for one strategy, including explicit cash."""
    rows: list[dict[str, object]] = []

    for date, month in signals.groupby(DATE_COL, sort=True):
        raw = raw_weights_for_strategy(month, strategy)
        weights = cap_and_redistribute(raw)
        lookup = month.set_index(SECTOR_COL)
        regime = str(month[REGIME_COL].iloc[0])
        model = str(month[MODEL_COL].iloc[0])

        for asset in (*EXPECTED_SECTORS, CASH_ASSET):
            if asset == CASH_ASSET:
                probability = np.nan
                rank = np.nan
                rank_score = np.nan
                selected = weights[asset] > TOLERANCE
            else:
                row = lookup.loc[asset]
                probability = float(row[PROB_COL])
                rank = int(row[RANK_COL])
                rank_score = float(row[RANK_SCORE_COL])
                selected = weights[asset] > TOLERANCE

            rows.append(
                {
                    "Signal_Date": pd.Timestamp(date),
                    "Strategy": strategy,
                    "Asset": asset,
                    "Target_Weight": float(weights[asset]),
                    "Selected": bool(selected),
                    "Outperformance_Probability": probability,
                    "Sector_Rank": rank,
                    "Rank_Score_0_1": rank_score,
                    "Macro_Regime": regime,
                    "model": model,
                }
            )

    return pd.DataFrame(rows).sort_values(["Signal_Date", "Strategy", "Asset"]).reset_index(drop=True)


def build_all_portfolios(signals: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [build_portfolio_weights(signals, strategy) for strategy in STRATEGIES],
        ignore_index=True,
    )


def add_turnover(portfolios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate monthly one-way turnover from prior target weights.

    Every strategy starts from 100% cash. One-way turnover is
    0.5 * sum(abs(target - previous)) across XLF, XLK, XLE, and CASH.
    """
    detailed_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        block = portfolios[portfolios["Strategy"] == strategy].copy()
        dates = sorted(block["Signal_Date"].unique())
        previous = {sector: 0.0 for sector in EXPECTED_SECTORS}
        previous[CASH_ASSET] = 1.0

        for date in dates:
            month = block[block["Signal_Date"] == date].copy()
            current = month.set_index("Asset")["Target_Weight"].astype(float).to_dict()
            month["Previous_Target_Weight"] = month["Asset"].map(previous).astype(float)
            month["Trade_Weight"] = month["Target_Weight"] - month["Previous_Target_Weight"]
            month["Trade_Direction"] = np.select(
                [month["Trade_Weight"] > TOLERANCE, month["Trade_Weight"] < -TOLERANCE],
                ["BUY", "SELL"],
                default="HOLD",
            )
            month["Turnover_Contribution"] = 0.5 * month["Trade_Weight"].abs()
            turnover = float(month["Turnover_Contribution"].sum())
            detailed_parts.append(month)

            sector_rows = month[month["Asset"].isin(EXPECTED_SECTORS)]
            selected_sectors = sector_rows.loc[sector_rows["Target_Weight"] > TOLERANCE, "Asset"].tolist()
            invested = float(sector_rows["Target_Weight"].sum())
            cash = float(month.loc[month["Asset"] == CASH_ASSET, "Target_Weight"].iloc[0])
            summary_rows.append(
                {
                    "Signal_Date": pd.Timestamp(date),
                    "Strategy": strategy,
                    "Selected_Sectors": ",".join(selected_sectors),
                    "Selected_Sector_Count": len(selected_sectors),
                    "Invested_Weight": invested,
                    "Cash_Weight": cash,
                    "One_Way_Turnover": turnover,
                    "Execution_Rule": "Next trading session after signal date",
                }
            )
            previous = current

    detailed = pd.concat(detailed_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return detailed, summary


def build_actionable_signals(signals: pd.DataFrame, default_portfolio: pd.DataFrame) -> pd.DataFrame:
    """Translate probabilities and ranks into a clean decision table."""
    sector_weights = default_portfolio[default_portfolio["Asset"].isin(EXPECTED_SECTORS)][
        ["Signal_Date", "Asset", "Target_Weight"]
    ].rename(columns={"Signal_Date": DATE_COL, "Asset": SECTOR_COL, "Target_Weight": "Default_Target_Weight"})

    output = signals.copy()
    output["Pass_Threshold"] = output[PROB_COL] >= PROBABILITY_THRESHOLD
    output["Top_N_Selected"] = output[RANK_COL].astype(int) <= TOP_N
    output["Signal_Score"] = output[PROB_COL].astype(float)
    output = output.merge(sector_weights, on=[DATE_COL, SECTOR_COL], how="left", validate="one_to_one")
    output["Default_Selected"] = output["Default_Target_Weight"] > TOLERANCE
    output["Actionable_Signal"] = np.where(output["Default_Selected"], "TARGET_LONG", "NO_POSITION")

    columns = [
        DATE_COL,
        SECTOR_COL,
        REGIME_COL,
        MODEL_COL,
        PROB_COL,
        PREDICTED_CLASS_COL,
        RANK_COL,
        RANK_SCORE_COL,
        "Pass_Threshold",
        "Top_N_Selected",
        "Signal_Score",
        "Default_Selected",
        "Default_Target_Weight",
        "Actionable_Signal",
    ]
    return output[columns].sort_values([DATE_COL, RANK_COL, SECTOR_COL]).reset_index(drop=True)


def build_strategy_summary(rebalance_summary: pd.DataFrame, all_portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        rebal = rebalance_summary[rebalance_summary["Strategy"] == strategy]
        weights = all_portfolios[
            (all_portfolios["Strategy"] == strategy)
            & (all_portfolios["Asset"].isin(EXPECTED_SECTORS))
        ]
        rows.append(
            {
                "Strategy": strategy,
                "Months": int(rebal["Signal_Date"].nunique()),
                "Average_Selected_Sectors": float(rebal["Selected_Sector_Count"].mean()),
                "Average_Invested_Weight": float(rebal["Invested_Weight"].mean()),
                "Average_Cash_Weight": float(rebal["Cash_Weight"].mean()),
                "Average_One_Way_Turnover": float(rebal["One_Way_Turnover"].mean()),
                "Maximum_Sector_Weight": float(weights["Target_Weight"].max()),
            }
        )
    return pd.DataFrame(rows)


def save_figures(all_portfolios: pd.DataFrame, rebalance_summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    latest_date = all_portfolios["Signal_Date"].max()
    latest = all_portfolios[all_portfolios["Signal_Date"] == latest_date].copy()
    pivot = latest.pivot(index="Strategy", columns="Asset", values="Target_Weight")
    pivot = pivot.reindex(index=STRATEGIES, columns=[*EXPECTED_SECTORS, CASH_ASSET]).fillna(0.0)

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title(f"Week 5 Target Weights - {pd.Timestamp(latest_date).date()}")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Target Weight")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week5_latest_target_weights.png", dpi=160)
    plt.close()

    turnover_pivot = rebalance_summary.pivot(
        index="Signal_Date", columns="Strategy", values="One_Way_Turnover"
    ).sort_index()
    ax = turnover_pivot.plot(figsize=(10, 5), marker="o")
    ax.set_title("Week 5 Monthly One-Way Turnover")
    ax.set_xlabel("Signal Date")
    ax.set_ylabel("Turnover")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week5_turnover_comparison.png", dpi=160)
    plt.close()


def save_config(signals: pd.DataFrame) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "week": 5,
        "input_signals": "results/tables/week4_selected_model_signals.csv",
        "signal_source": "Week 4 selected-model outperformance probabilities",
        "sectors": list(EXPECTED_SECTORS),
        "cash_asset": CASH_ASSET,
        "rebalance_frequency": "Monthly",
        "execution_timing": "Next trading session after each signal date",
        "top_n": TOP_N,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "max_sector_weight": MAX_SECTOR_WEIGHT,
        "constraints": {
            "long_only": True,
            "leverage": False,
            "shorting": False,
            "cash_absorbs_unallocated_weight": True,
        },
        "strategies": {
            STRATEGY_TOP_N: "Select top 2 sectors by model rank and equal-weight them.",
            STRATEGY_THRESHOLD: "Select sectors with probability >= 0.50 and equal-weight them.",
            STRATEGY_SCORE: "Weight all sectors in proportion to model probability.",
            STRATEGY_DEFAULT: "Apply probability >= 0.50, keep top 2, then weight by probability.",
        },
        "selected_week4_model_labels": sorted(signals[MODEL_COL].astype(str).unique().tolist()),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def format_report_table(data: pd.DataFrame, float_digits: int = 4) -> str:
    """Return a dependency-free fixed-width table for the Markdown report."""
    display = data.copy()
    for column in display.select_dtypes(include=["float", "float64", "float32"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
        )
    return "```text\n" + display.to_string(index=False) + "\n```"


def save_report(
    signals: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    default_portfolio: pd.DataFrame,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    latest_date = pd.Timestamp(signals[DATE_COL].max())
    latest_default = default_portfolio[default_portfolio["Signal_Date"] == latest_date].copy()
    latest_sector = latest_default[latest_default["Asset"].isin(EXPECTED_SECTORS)]
    latest_cash = float(latest_default.loc[latest_default["Asset"] == CASH_ASSET, "Target_Weight"].iloc[0])

    lines = [
        "# Week 5 Strategy Design Report",
        "",
        "## Scope",
        "",
        "Week 5 converts the Week 4 selected-model probabilities into monthly long-only sector portfolio targets. The implementation covers Top-N selection, probability-threshold filtering, score weighting, monthly rebalancing, and position sizing.",
        "",
        "## Strategy Parameters",
        "",
        f"- Top-N: {TOP_N}",
        f"- Probability threshold: {PROBABILITY_THRESHOLD:.2f}",
        f"- Maximum sector weight: {MAX_SECTOR_WEIGHT:.0%}",
        "- Rebalancing frequency: monthly",
        "- Execution rule: next trading session after each signal date",
        "- Portfolio constraints: long-only, no leverage, no short positions; residual allocation remains in cash",
        "",
        "## Strategy Summary",
        "",
        format_report_table(strategy_summary),
        "",
        "## Latest Combined-Default Target",
        "",
        f"Signal date: {latest_date.date()}",
        "",
        format_report_table(
            latest_sector[["Asset", "Outperformance_Probability", "Sector_Rank", "Target_Weight"]]
            .sort_values("Sector_Rank")
            .assign(Sector_Rank=lambda frame: frame["Sector_Rank"].astype(int))
        ),
        "",
        f"Cash target: {latest_cash:.4f}",
        "",
        "## Rebalancing Logic",
        "",
        "Each month the strategy computes new target weights from the current Week 4 signal snapshot. Trade weights are the difference between the new target and the previous month's target. Reported one-way turnover is half of the absolute weight changes across XLF, XLK, XLE, and cash. The first rebalance assumes the portfolio starts fully in cash.",
        "",
        "## Data Handling",
        "",
        "Week 5 consumes only the leakage-safe Week 4 signal file. Realized forward returns and classification targets are not used in portfolio construction.",
        "",
        "## Main Outputs",
        "",
        "- `results/tables/week5_actionable_signals.csv`",
        "- `results/tables/week5_top_n_weights.csv`",
        "- `results/tables/week5_threshold_weights.csv`",
        "- `results/tables/week5_score_weighted_weights.csv`",
        "- `results/tables/week5_default_portfolio.csv`",
        "- `results/tables/week5_all_portfolio_weights.csv`",
        "- `results/tables/week5_rebalance_trades.csv`",
        "- `results/tables/week5_rebalance_summary.csv`",
        "- `results/tables/week5_strategy_summary.csv`",
    ]
    (DOCS_DIR / "week5_strategy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    signals = load_week4_signals()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_portfolios = build_all_portfolios(signals)
    trades, rebalance_summary = add_turnover(all_portfolios)

    strategy_to_filename = {
        STRATEGY_TOP_N: "week5_top_n_weights.csv",
        STRATEGY_THRESHOLD: "week5_threshold_weights.csv",
        STRATEGY_SCORE: "week5_score_weighted_weights.csv",
        STRATEGY_DEFAULT: "week5_default_portfolio.csv",
    }
    for strategy, filename in strategy_to_filename.items():
        all_portfolios[all_portfolios["Strategy"] == strategy].to_csv(
            TABLE_DIR / filename, index=False
        )

    default_portfolio = all_portfolios[all_portfolios["Strategy"] == STRATEGY_DEFAULT].copy()
    actionable = build_actionable_signals(signals, default_portfolio)
    strategy_summary = build_strategy_summary(rebalance_summary, all_portfolios)

    actionable.to_csv(TABLE_DIR / "week5_actionable_signals.csv", index=False)
    all_portfolios.to_csv(TABLE_DIR / "week5_all_portfolio_weights.csv", index=False)
    trades.to_csv(TABLE_DIR / "week5_rebalance_trades.csv", index=False)
    rebalance_summary.to_csv(TABLE_DIR / "week5_rebalance_summary.csv", index=False)
    strategy_summary.to_csv(TABLE_DIR / "week5_strategy_summary.csv", index=False)

    save_config(signals)
    save_figures(all_portfolios, rebalance_summary)
    save_report(signals, strategy_summary, default_portfolio)

    print(f"[PASS] Loaded {signals[DATE_COL].nunique()} monthly Week 4 signal dates.")
    print("[PASS] Translated model probabilities into actionable long/no-position signals.")
    print("[PASS] Built Top-N, threshold, probability-score, and combined portfolio rules.")
    print("[PASS] Applied monthly rebalancing, long-only sizing, a 60% sector cap, and explicit cash.")
    print("[PASS] Saved Week 5 portfolio targets, trades, summaries, figures, and report.")
    print("[DONE] Week 5 strategy construction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
