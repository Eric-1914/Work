"""Run the complete Week 7 backtesting and validation pipeline."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
STEPS = [
    PROJECT_ROOT / "src" / "backtesting" / "backtest_week7.py",
    PROJECT_ROOT / "src" / "validation" / "validate_week7.py",
]


def main() -> int:
    for step in STEPS:
        print(f"\n=== Running {step.relative_to(PROJECT_ROOT)} ===", flush=True)
        result = subprocess.run([sys.executable, str(step)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Week 7 pipeline stopped at {step.name}.")
            return result.returncode
    print("\n[DONE] Full Week 7 backtesting pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
