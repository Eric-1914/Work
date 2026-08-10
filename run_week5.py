"""Run the complete Week 5 strategy-design and signal-processing pipeline."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
STEPS = [
    PROJECT_ROOT / "src" / "strategy" / "build_strategy_week5.py",
    PROJECT_ROOT / "src" / "validation" / "validate_week5.py",
]


def main() -> int:
    for step in STEPS:
        print(f"\n=== Running {step.relative_to(PROJECT_ROOT)} ===", flush=True)
        result = subprocess.run([sys.executable, str(step)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Week 5 pipeline stopped at {step.name}.")
            return result.returncode

    print("\n[DONE] Full Week 5 strategy pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
