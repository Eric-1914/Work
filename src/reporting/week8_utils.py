"""Shared utilities for Week 8 reporting and reproducibility outputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Week8Paths:
    root: Path

    @property
    def week7_summary(self) -> Path:
        return self.root / "results" / "tables" / "week7_performance_summary.csv"

    @property
    def week7_daily(self) -> Path:
        return self.root / "results" / "tables" / "week7_daily_backtest.csv"

    @property
    def week7_benchmarks(self) -> Path:
        return self.root / "results" / "tables" / "week7_benchmark_daily.csv"

    @property
    def week7_rolling(self) -> Path:
        return self.root / "results" / "tables" / "week7_rolling_validation.csv"

    @property
    def week7_cost(self) -> Path:
        return self.root / "results" / "tables" / "week7_cost_sensitivity.csv"

    @property
    def week7_rebalance(self) -> Path:
        return self.root / "results" / "tables" / "week7_rebalance_log.csv"

    @property
    def week7_config(self) -> Path:
        return self.root / "data" / "backtesting" / "week7_backtest_config.json"

    @property
    def week6_portfolio(self) -> Path:
        return self.root / "results" / "tables" / "week6_optimized_portfolio.csv"

    @property
    def tables_dir(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def dashboard_dir(self) -> Path:
        return self.root / "results" / "dashboard"

    @property
    def reporting_dir(self) -> Path:
        return self.root / "data" / "reporting"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"


REQUIRED_STRATEGIES = {
    "Week7_Strategy_Net",
    "SPY_Benchmark",
    "Equal_Weighted_Sectors",
}

SCORECARD_METRICS = [
    "Total_Return",
    "CAGR",
    "Annualized_Volatility",
    "Sharpe_0RF",
    "Sortino_0MAR",
    "Max_Drawdown",
    "Daily_Win_Rate",
]


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def read_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    if path.suffix.lower() not in {".py", ".md", ".txt", ".json", ".csv"}:
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def infer_week(path: Path) -> str:
    match = re.search(r"week(\d+)", path.as_posix(), flags=re.IGNORECASE)
    return f"Week {match.group(1)}" if match else "Shared"


def module_category(path: Path) -> str:
    parts = set(path.parts)
    if path.name.startswith("run_week"):
        return "Runner"
    if "tests" in parts:
        return "Test"
    if "docs" in parts:
        return "Documentation"
    if "validation" in parts:
        return "Validation"
    if "src" in parts:
        return "Source Module"
    return "Project File"


def discover_project_modules(root: Path) -> pd.DataFrame:
    candidates: set[Path] = set()
    candidates.update(root.glob("run_week*.py"))
    candidates.update(root.glob("src/**/*.py"))
    candidates.update(root.glob("tests/test_week*.py"))
    candidates.update(root.glob("docs/week*_methodology.md"))

    rows = []
    for path in sorted(p for p in candidates if p.is_file()):
        relative = path.relative_to(root)
        rows.append(
            {
                "Path": relative.as_posix(),
                "Week": infer_week(relative),
                "Category": module_category(relative),
                "Lines": line_count(path),
                "SHA256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["Path", "Week", "Category", "Lines", "SHA256"])


def safe_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def environment_manifest(root: Path) -> dict:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "git_commit_at_run": safe_git_commit(root),
    }


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def format_num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "_No data available._"
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{float(value):.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
