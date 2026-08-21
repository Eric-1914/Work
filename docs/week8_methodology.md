# Week 8: Final Analysis, Reporting & Reproducibility

## Objective

Week 8 converts the validated Week 6 portfolio and Week 7 backtest into a final reporting and reproducibility layer. It focuses on code clarity, visual reporting, documentation, and repeatable outputs. It does **not** retrain models or retune strategy/risk/backtest parameters after observing final performance.

## Scope implemented in code

### 1. Refactor for clarity and reproducibility

- A small shared utility module (`src/reporting/week8_utils.py`) centralizes paths, file validation, hashing, formatting, module discovery, and environment metadata.
- `run_week8.py` provides one reproducible entry point that builds outputs and then runs independent validation.
- Source/input SHA-256 hashes and package versions are recorded in `data/reporting/week8_reproducibility_manifest.json`.
- `results/tables/week8_module_inventory.csv` inventories detected weekly runners, source modules, tests, and methodology files.

### 2. Final analysis and visual reporting

Week 8 reads validated Week 7 outputs and produces:

- a final strategy/benchmark scorecard;
- rolling-window robustness statistics;
- trading-cost sensitivity statistics;
- the latest optimized Week 6 allocation;
- data-quality checks;
- five Week 8 summary figures;
- a static HTML dashboard that can be opened locally without Streamlit, Dash, or a server.

A static dashboard is intentionally used because the assignment marks Streamlit/Dash as optional. This keeps the project portable and avoids adding a new runtime dependency.

### 3. Document modules and processes

`docs/week8_project_documentation.md` records the Week 1–8 process and dynamically lists detected project modules. Exact source hashes are saved separately so the code used for a report can be verified later.


## Inputs

- `results/tables/week7_performance_summary.csv`
- `results/tables/week7_daily_backtest.csv`
- `results/tables/week7_benchmark_daily.csv`
- `results/tables/week7_rolling_validation.csv`
- `results/tables/week7_cost_sensitivity.csv`
- `results/tables/week7_rebalance_log.csv`
- `data/backtesting/week7_backtest_config.json`
- `results/tables/week6_optimized_portfolio.csv`

## Main outputs

### Tables

- `results/tables/week8_final_scorecard.csv`
- `results/tables/week8_robustness_summary.csv`
- `results/tables/week8_latest_allocation.csv`
- `results/tables/week8_data_quality_audit.csv`
- `results/tables/week8_module_inventory.csv`

### Figures

- `results/figures/week8_cagr_comparison.png`
- `results/figures/week8_risk_adjusted_comparison.png`
- `results/figures/week8_rolling_sharpe.png`
- `results/figures/week8_cost_robustness.png`
- `results/figures/week8_latest_allocation.png`

### Dashboard and documentation

- `results/dashboard/week8_dashboard.html`
- `docs/week8_final_report.md`
- `docs/week8_project_documentation.md`

### Reproducibility metadata

- `data/reporting/week8_reporting_config.json`
- `data/reporting/week8_reproducibility_manifest.json`

## Validation design

The independent Week 8 validator recomputes scorecard values, benchmark differences, latest allocation, rolling robustness statistics, turnover/cost summaries, input hashes, module hashes, and dashboard references. This prevents the final report from becoming a disconnected reporting layer that can silently disagree with the validated Week 6/7 data.

## Run command

```bash
python3 run_week8.py
```

For the isolated smoke test:

```bash
python3 tests/test_week8_smoke.py
```


## Week 7 trading-cost handoff

Week 8 reads `Total_Trading_Cost_Fraction` from the validated Week 7 rebalance and cost-sensitivity outputs. The backtest is normalized to initial equity 1.0, so trading friction is reported as a portfolio-value fraction rather than mislabeled as dollars.
