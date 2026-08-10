"""Run Week 4 on an isolated exact-schema synthetic Week 3 dataset.

IMPORTANT: this test never writes to the user's real data/modeling directory.
Everything is copied into a temporary project directory and deleted afterward.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["XLF", "XLK", "XLE"]

NUMERIC_FEATURES = [
    "Daily_Return",
    "Rolling_Return_5D",
    "Rolling_Return_20D",
    "Rolling_Return_60D",
    "Momentum_20D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "RSI_14",
    "MACD_Pct",
    "MACD_Signal_Pct",
    "MACD_Hist_Pct",
    "Market_Mean_Rolling_Return_20D",
    "Relative_Rolling_Return_20D",
    "Rank_Rolling_Return_20D",
    "Market_Mean_Rolling_Return_60D",
    "Relative_Rolling_Return_60D",
    "Rank_Rolling_Return_60D",
    "Market_Mean_Momentum_20D",
    "Relative_Momentum_20D",
    "Rank_Momentum_20D",
    "Market_Mean_Rolling_Volatility_20D",
    "Relative_Rolling_Volatility_20D",
    "Rank_Rolling_Volatility_20D",
    "Market_Mean_RSI_14",
    "Relative_RSI_14",
    "Rank_RSI_14",
    "Market_Mean_MACD_Hist_Pct",
    "Relative_MACD_Hist_Pct",
    "Rank_MACD_Hist_Pct",
    "GDP_Growth",
    "Inflation_YoY",
    "FedFunds_Rate",
]


def make_dataset(path: Path) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-31", periods=108, freq="ME")
    rows: list[dict[str, object]] = []

    for month_index, date in enumerate(dates):
        gdp = 2.2 + 1.0 * np.sin(month_index / 11.0) + rng.normal(0, 0.12)
        inflation = 2.1 + 0.8 * np.sin(month_index / 15.0 + 0.9) + rng.normal(0, 0.10)
        fed = 2.0 + 1.2 * np.sin(month_index / 19.0 - 0.5) + rng.normal(0, 0.08)

        growth_regime = "High_Growth" if gdp >= 2.2 else "Low_Growth"
        inflation_regime = "High_Inflation" if inflation >= 2.1 else "Low_Inflation"
        rate_delta = np.cos(month_index / 19.0 - 0.5)
        rate_regime = "Rising_Rates" if rate_delta >= 0 else "Falling_Rates"

        latent_values = []
        sector_rows = []
        for sector_index, ticker in enumerate(TICKERS):
            mom = 0.045 * np.sin(month_index / 7.0 + sector_index) + rng.normal(0, 0.025)
            rr20 = mom + rng.normal(0, 0.012)
            rr60 = 1.8 * mom + rng.normal(0, 0.025)
            vol20 = 0.15 + 0.025 * sector_index + abs(rng.normal(0, 0.02))
            vol60 = vol20 * 1.08 + abs(rng.normal(0, 0.01))
            rsi = 50 + 14 * np.sin(month_index / 8.0 + sector_index) + rng.normal(0, 3)
            macd_hist = 0.30 * mom + rng.normal(0, 0.006)
            latent = (
                4.8 * mom
                + 1.6 * rr60
                - 1.2 * vol20
                + 0.018 * (rsi - 50)
                + 0.14 * gdp
                - 0.10 * inflation
                - 0.04 * fed
                + [0.01, 0.05, -0.02][sector_index]
                + rng.normal(0, 0.12)
            )
            latent_values.append(latent)
            sector_rows.append((ticker, mom, rr20, rr60, vol20, vol60, rsi, macd_hist))

        benchmark_latent = float(np.mean(latent_values))
        realized = np.asarray(
            [(value - benchmark_latent) * 0.04 + rng.normal(0, 0.006) for value in latent_values]
        )
        realized = realized - realized.mean()

        for sector_index, (ticker, mom, rr20, rr60, vol20, vol60, rsi, macd_hist) in enumerate(sector_rows):
            rows.append(
                {
                    "Date": date,
                    "Sector": ticker,
                    "Adj_Close": 100 + 20 * sector_index + month_index * 0.2,
                    "Daily_Return": rng.normal(0, 0.012),
                    "Rolling_Return_5D": rng.normal(0, 0.025),
                    "Rolling_Return_20D": rr20,
                    "Rolling_Return_60D": rr60,
                    "Momentum_20D": mom,
                    "Rolling_Volatility_20D": vol20,
                    "Rolling_Volatility_60D": vol60,
                    "RSI_14": rsi,
                    "MACD_Pct": 0.6 * mom + rng.normal(0, 0.004),
                    "MACD_Signal_Pct": 0.45 * mom + rng.normal(0, 0.004),
                    "MACD_Hist_Pct": macd_hist,
                    "GDP_Growth": gdp,
                    "Inflation_YoY": inflation,
                    "FedFunds_Rate": fed,
                    "Growth_Regime": growth_regime,
                    "Inflation_Regime": inflation_regime,
                    "Rate_Regime": rate_regime,
                    "Next_21D_Sector_Return": 0.02 + realized[sector_index],
                    "Next_21D_EqualWeight_Return": 0.02,
                    "Relative_Return_21D": realized[sector_index],
                    "Target_Outperform": int(realized[sector_index] > 0),
                }
            )

    data = pd.DataFrame(rows).sort_values(["Date", "Sector"]).reset_index(drop=True)
    relative_sources = [
        "Rolling_Return_20D",
        "Rolling_Return_60D",
        "Momentum_20D",
        "Rolling_Volatility_20D",
        "RSI_14",
        "MACD_Hist_Pct",
    ]
    for feature in relative_sources:
        date_mean = data.groupby("Date")[feature].transform("mean")
        data[f"Market_Mean_{feature}"] = date_mean
        data[f"Relative_{feature}"] = data[feature] - date_mean
        data[f"Rank_{feature}"] = data.groupby("Date")[feature].rank(method="average", pct=True)

    missing = [column for column in NUMERIC_FEATURES if column not in data.columns]
    if missing:
        raise AssertionError(f"Synthetic data is missing Week 3 features: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")


def copy_runtime_files(temp_root: Path) -> None:
    (temp_root / "src" / "modeling").mkdir(parents=True, exist_ok=True)
    (temp_root / "src" / "validation").mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "run_week4.py", temp_root / "run_week4.py")
    shutil.copy2(
        PROJECT_ROOT / "src" / "modeling" / "week4_estimators.py",
        temp_root / "src" / "modeling" / "week4_estimators.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "src" / "modeling" / "train_models_week4.py",
        temp_root / "src" / "modeling" / "train_models_week4.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "src" / "validation" / "validate_week4.py",
        temp_root / "src" / "validation" / "validate_week4.py",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="week4_sector_rotation_test_") as temp_dir:
        temp_root = Path(temp_dir)
        copy_runtime_files(temp_root)
        make_dataset(temp_root / "data" / "modeling" / "model_dataset.csv")

        result = subprocess.run([sys.executable, "run_week4.py"], cwd=temp_root)
        if result.returncode != 0:
            return result.returncode

        required = [
            temp_root / "results" / "tables" / "week4_selected_model_rankings.csv",
            temp_root / "results" / "tables" / "week4_selected_model_signals.csv",
            temp_root / "results" / "tables" / "week4_regime_model_usage.csv",
            temp_root / "results" / "figures" / "week4_model_comparison.png",
            temp_root / "results" / "models" / "week4_selected_model.joblib",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise AssertionError(f"Smoke test missing outputs: {missing}")

    print("[SMOKE TEST PASS] Isolated Week 4 exact-schema test completed without touching repository data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
