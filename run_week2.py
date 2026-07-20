from __future__ import annotations
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
STEPS = [
    PROJECT_ROOT/"src"/"data_cleaning"/"clean_data.py",
    PROJECT_ROOT/"src"/"feature_engineering"/"build_features.py",
    PROJECT_ROOT/"src"/"eda"/"run_eda.py",
    PROJECT_ROOT/"src"/"validation"/"validate_week2.py",
]

def main():
    for step in STEPS:
        print(f"\n=== Running {step.relative_to(PROJECT_ROOT)} ===")
        result = subprocess.run([sys.executable, str(step)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[ERROR] Pipeline stopped at {step.name}.")
            return result.returncode
    print("\n[DONE] Full Week 2 pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
