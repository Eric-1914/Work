"""Validate Week 5 signal-processing, portfolio construction, and rebalancing outputs."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strategy.build_strategy_week5 import (
    CASH_ASSET,
    CONFIG_PATH,
    DATE_COL,
    DOCS_DIR,
    EXPECTED_SECTORS,
    FIGURE_DIR,
    INPUT_PATH,
    MAX_SECTOR_WEIGHT,
    PROBABILITY_THRESHOLD,
    PROB_COL,
    RANK_COL,
    SECTOR_COL,
    STRATEGIES,
    STRATEGY_DEFAULT,
    STRATEGY_SCORE,
    STRATEGY_THRESHOLD,
    STRATEGY_TOP_N,
    TABLE_DIR,
    TOLERANCE,
    TOP_N,
    add_turnover,
    build_actionable_signals,
    build_all_portfolios,
    build_strategy_summary,
    load_week4_signals,
)


WEIGHT_FILES = {
    STRATEGY_TOP_N: TABLE_DIR / "week5_top_n_weights.csv",
    STRATEGY_THRESHOLD: TABLE_DIR / "week5_threshold_weights.csv",
    STRATEGY_SCORE: TABLE_DIR / "week5_score_weighted_weights.csv",
    STRATEGY_DEFAULT: TABLE_DIR / "week5_default_portfolio.csv",
}


def check_files() -> None:
    required = [
        INPUT_PATH,
        CONFIG_PATH,
        TABLE_DIR / "week5_actionable_signals.csv",
        TABLE_DIR / "week5_all_portfolio_weights.csv",
        TABLE_DIR / "week5_rebalance_trades.csv",
        TABLE_DIR / "week5_rebalance_summary.csv",
        TABLE_DIR / "week5_strategy_summary.csv",
        FIGURE_DIR / "week5_latest_target_weights.png",
        FIGURE_DIR / "week5_turnover_comparison.png",
        DOCS_DIR / "week5_strategy_report.md",
        *WEIGHT_FILES.values(),
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week 5 outputs: {missing}")


def load_saved_portfolios() -> pd.DataFrame:
    data = pd.read_csv(TABLE_DIR / "week5_all_portfolio_weights.csv", parse_dates=["Signal_Date"])
    if data.empty:
        raise ValueError("week5_all_portfolio_weights.csv is empty.")
    required = {
        "Signal_Date", "Strategy", "Asset", "Target_Weight", "Selected",
        "Outperformance_Probability", "Sector_Rank", "Rank_Score_0_1",
        "Macro_Regime", "model",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"week5_all_portfolio_weights.csv is missing columns: {missing}")
    return data


def check_weight_integrity(portfolios: pd.DataFrame, signals: pd.DataFrame) -> None:
    expected_assets = set(EXPECTED_SECTORS) | {CASH_ASSET}
    if set(portfolios["Strategy"].astype(str).unique()) != set(STRATEGIES):
        raise ValueError("Week 5 all-portfolio file is missing a required strategy.")

    weights = pd.to_numeric(portfolios["Target_Weight"], errors="coerce")
    if weights.isna().any() or not weights.between(0, 1).all():
        raise ValueError("Week 5 target weights must be finite and inside [0, 1].")

    expected_dates = set(signals[DATE_COL])
    for (strategy, date), group in portfolios.groupby(["Strategy", "Signal_Date"]):
        if set(group["Asset"].astype(str)) != expected_assets or len(group) != 4:
            raise ValueError(f"{strategy} {pd.Timestamp(date).date()} must contain XLF, XLK, XLE, and CASH.")
        total = float(group["Target_Weight"].sum())
        if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-9):
            raise ValueError(f"{strategy} {pd.Timestamp(date).date()} weights sum to {total}, not 1.0.")
        sector = group[group["Asset"].isin(EXPECTED_SECTORS)]
        if (sector["Target_Weight"] > MAX_SECTOR_WEIGHT + 1e-10).any():
            raise ValueError(f"{strategy} exceeds the {MAX_SECTOR_WEIGHT:.0%} sector cap.")

    for strategy in STRATEGIES:
        block = portfolios[portfolios["Strategy"] == strategy]
        if set(block["Signal_Date"]) != expected_dates:
            raise ValueError(f"{strategy} is missing one or more signal dates.")


def check_strategy_rules(portfolios: pd.DataFrame, signals: pd.DataFrame) -> None:
    for date in sorted(signals[DATE_COL].unique()):
        month = signals[signals[DATE_COL] == date]

        top = portfolios[
            (portfolios["Strategy"] == STRATEGY_TOP_N)
            & (portfolios["Signal_Date"] == date)
            & (portfolios["Asset"].isin(EXPECTED_SECTORS))
        ]
        positive_top = set(top.loc[top["Target_Weight"] > TOLERANCE, "Asset"])
        expected_top = set(month.loc[month[RANK_COL].astype(int) <= TOP_N, SECTOR_COL])
        if positive_top != expected_top:
            raise ValueError(f"Top-N selection mismatch on {pd.Timestamp(date).date()}.")
        expected_equal = 1.0 / TOP_N
        if not np.allclose(top.loc[top["Asset"].isin(expected_top), "Target_Weight"], expected_equal):
            raise ValueError("Top-N selected sectors must be equally weighted.")

        threshold = portfolios[
            (portfolios["Strategy"] == STRATEGY_THRESHOLD)
            & (portfolios["Signal_Date"] == date)
        ]
        threshold_sector = threshold[threshold["Asset"].isin(EXPECTED_SECTORS)]
        actual_selected = set(threshold_sector.loc[threshold_sector["Target_Weight"] > TOLERANCE, "Asset"])
        expected_selected = set(month.loc[month[PROB_COL] >= PROBABILITY_THRESHOLD, SECTOR_COL])
        if actual_selected != expected_selected:
            raise ValueError(f"Threshold selection mismatch on {pd.Timestamp(date).date()}.")
        n = len(expected_selected)
        if n == 0:
            cash = float(threshold.loc[threshold["Asset"] == CASH_ASSET, "Target_Weight"].iloc[0])
            if not np.isclose(cash, 1.0):
                raise ValueError("Threshold strategy must hold 100% cash when no sector passes.")
        elif n == 1:
            selected_weight = float(
                threshold_sector.loc[threshold_sector["Asset"].isin(expected_selected), "Target_Weight"].iloc[0]
            )
            if not np.isclose(selected_weight, MAX_SECTOR_WEIGHT):
                raise ValueError("Single threshold-selected sector should be capped at 60%.")
        else:
            expected_weight = 1.0 / n
            actual = threshold_sector.loc[threshold_sector["Asset"].isin(expected_selected), "Target_Weight"]
            if not np.allclose(actual, expected_weight):
                raise ValueError("Threshold-selected sectors must be equally weighted when the cap does not bind.")

        default = portfolios[
            (portfolios["Strategy"] == STRATEGY_DEFAULT)
            & (portfolios["Signal_Date"] == date)
            & (portfolios["Asset"].isin(EXPECTED_SECTORS))
        ]
        expected_default = set(
            month.loc[
                (month[PROB_COL] >= PROBABILITY_THRESHOLD)
                & (month[RANK_COL].astype(int) <= TOP_N),
                SECTOR_COL,
            ]
        )
        actual_default = set(default.loc[default["Target_Weight"] > TOLERANCE, "Asset"])
        if actual_default != expected_default:
            raise ValueError(f"Combined-default eligibility mismatch on {pd.Timestamp(date).date()}.")

        # Score-weighted strategy must assign larger uncapped weights to larger probabilities.
        score = portfolios[
            (portfolios["Strategy"] == STRATEGY_SCORE)
            & (portfolios["Signal_Date"] == date)
            & (portfolios["Asset"].isin(EXPECTED_SECTORS))
        ].copy()
        # Pairwise monotonicity can only be required when neither member is at the cap.
        uncapped = score[score["Target_Weight"] < MAX_SECTOR_WEIGHT - 1e-10]
        for _, row_i in uncapped.iterrows():
            for _, row_j in uncapped.iterrows():
                if (
                    row_i[PROB_COL] > row_j[PROB_COL] + 1e-12
                    and row_i["Target_Weight"] < row_j["Target_Weight"] - 1e-10
                ):
                    raise ValueError("Probability score weighting is not monotonic with model probability.")


def check_recomputed_outputs(signals: pd.DataFrame, saved: pd.DataFrame) -> None:
    expected = build_all_portfolios(signals)
    expected = expected.sort_values(["Signal_Date", "Strategy", "Asset"]).reset_index(drop=True)
    actual = saved.sort_values(["Signal_Date", "Strategy", "Asset"]).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=1e-9)

    expected_trades, expected_rebalance = add_turnover(expected)
    actual_trades = pd.read_csv(TABLE_DIR / "week5_rebalance_trades.csv", parse_dates=["Signal_Date"])
    actual_rebalance = pd.read_csv(TABLE_DIR / "week5_rebalance_summary.csv", parse_dates=["Signal_Date"])
    actual_rebalance["Selected_Sectors"] = actual_rebalance["Selected_Sectors"].fillna("")
    expected_rebalance["Selected_Sectors"] = expected_rebalance["Selected_Sectors"].fillna("")
    assert_frame_equal(
        actual_trades.sort_values(["Signal_Date", "Strategy", "Asset"]).reset_index(drop=True),
        expected_trades.sort_values(["Signal_Date", "Strategy", "Asset"]).reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )
    assert_frame_equal(
        actual_rebalance.sort_values(["Signal_Date", "Strategy"]).reset_index(drop=True),
        expected_rebalance.sort_values(["Signal_Date", "Strategy"]).reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )

    default = expected[expected["Strategy"] == STRATEGY_DEFAULT]
    expected_actionable = build_actionable_signals(signals, default)
    actual_actionable = pd.read_csv(TABLE_DIR / "week5_actionable_signals.csv", parse_dates=[DATE_COL])
    assert_frame_equal(
        actual_actionable.sort_values([DATE_COL, RANK_COL, SECTOR_COL]).reset_index(drop=True),
        expected_actionable.sort_values([DATE_COL, RANK_COL, SECTOR_COL]).reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )

    expected_summary = build_strategy_summary(expected_rebalance, expected)
    actual_summary = pd.read_csv(TABLE_DIR / "week5_strategy_summary.csv")
    assert_frame_equal(
        actual_summary.sort_values("Strategy").reset_index(drop=True),
        expected_summary.sort_values("Strategy").reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-9,
    )


def check_individual_weight_files(saved_all: pd.DataFrame) -> None:
    for strategy, path in WEIGHT_FILES.items():
        actual = pd.read_csv(path, parse_dates=["Signal_Date"])
        expected = saved_all[saved_all["Strategy"] == strategy].copy()
        assert_frame_equal(
            actual.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True),
            expected.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
            rtol=1e-9,
        )


def check_config() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "week": 5,
        "top_n": TOP_N,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "max_sector_weight": MAX_SECTOR_WEIGHT,
        "rebalance_frequency": "Monthly",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Week 5 config field {key} is inconsistent.")
    if config.get("cash_asset") != CASH_ASSET:
        raise ValueError("Week 5 config cash asset is inconsistent.")


def main() -> int:
    check_files()
    signals = load_week4_signals()
    saved = load_saved_portfolios()
    check_weight_integrity(saved, signals)
    check_strategy_rules(saved, signals)
    check_recomputed_outputs(signals, saved)
    check_individual_weight_files(saved)
    check_config()

    print("[PASS] Required Week 5 files and the Week 4 signal interface are valid.")
    print("[PASS] All portfolio weights are long-only, capped, and sum to 100% including cash.")
    print("[PASS] Top-N, threshold-filter, score-weighted, and combined rules were recomputed.")
    print("[PASS] Monthly rebalance trades and one-way turnover were recomputed from prior targets.")
    print("[PASS] Actionable signals, strategy summaries, and individual weight files match exactly.")
    print("[DONE] All Week 5 outputs passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
