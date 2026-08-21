# Week 8: Project Documentation

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

| Path | Week | Category | Lines |
| --- | --- | --- | --- |
| docs/week2_methodology.md | Week 2 | Documentation | 138 |
| docs/week3_methodology.md | Week 3 | Documentation | 158 |
| docs/week4_methodology.md | Week 4 | Documentation | 183 |
| docs/week5_methodology.md | Week 5 | Documentation | 140 |
| docs/week6_methodology.md | Week 6 | Documentation | 175 |
| docs/week7_methodology.md | Week 7 | Documentation | 150 |
| docs/week8_methodology.md | Week 8 | Documentation | 94 |
| run_week2.py | Week 2 | Runner | 25 |
| run_week3.py | Week 3 | Runner | 36 |
| run_week4.py | Week 4 | Runner | 30 |
| run_week5.py | Week 5 | Runner | 30 |
| run_week6.py | Week 6 | Runner | 30 |
| run_week7.py | Week 7 | Runner | 28 |
| run_week8.py | Week 8 | Runner | 28 |
| src/backtesting/backtest_week7.py | Week 7 | Source Module | 851 |
| src/data_cleaning/clean_data.py | Shared | Source Module | 139 |
| src/data_collection/collect_week1_data.py | Week 1 | Source Module | 222 |
| src/data_collection/validate_week1_data.py | Week 1 | Source Module | 111 |
| src/eda/run_eda.py | Shared | Source Module | 203 |
| src/feature_engineering/build_features.py | Shared | Source Module | 106 |
| src/modeling/build_model_dataset.py | Shared | Source Module | 327 |
| src/modeling/train_models.py | Shared | Source Module | 678 |
| src/modeling/train_models_week4.py | Week 4 | Source Module | 877 |
| src/modeling/week4_estimators.py | Week 4 | Source Module | 148 |
| src/optimization/optimize_week6.py | Week 6 | Source Module | 1086 |
| src/reporting/build_week8_report.py | Week 8 | Source Module | 565 |
| src/reporting/week8_utils.py | Week 8 | Source Module | 228 |
| src/strategy/build_strategy_week5.py | Week 5 | Source Module | 608 |
| src/validation/validate_week2.py | Week 2 | Validation | 148 |
| src/validation/validate_week3.py | Week 3 | Validation | 162 |
| src/validation/validate_week4.py | Week 4 | Validation | 381 |
| src/validation/validate_week5.py | Week 5 | Validation | 288 |
| src/validation/validate_week6.py | Week 6 | Validation | 266 |
| src/validation/validate_week7.py | Week 7 | Validation | 225 |
| src/validation/validate_week8.py | Week 8 | Validation | 195 |
| tests/test_week4_smoke.py | Week 4 | Test | 199 |
| tests/test_week5_smoke.py | Week 5 | Test | 110 |
| tests/test_week6_smoke.py | Week 6 | Test | 238 |
| tests/test_week7_smoke.py | Week 7 | Test | 152 |
| tests/test_week8_smoke.py | Week 8 | Test | 176 |

## Module counts by stage

| Week | Category | Files |
| --- | --- | --- |
| Shared | Source Module | 5 |
| Week 1 | Source Module | 2 |
| Week 2 | Documentation | 1 |
| Week 2 | Runner | 1 |
| Week 2 | Validation | 1 |
| Week 3 | Documentation | 1 |
| Week 3 | Runner | 1 |
| Week 3 | Validation | 1 |
| Week 4 | Documentation | 1 |
| Week 4 | Runner | 1 |
| Week 4 | Source Module | 2 |
| Week 4 | Test | 1 |
| Week 4 | Validation | 1 |
| Week 5 | Documentation | 1 |
| Week 5 | Runner | 1 |
| Week 5 | Source Module | 1 |
| Week 5 | Test | 1 |
| Week 5 | Validation | 1 |
| Week 6 | Documentation | 1 |
| Week 6 | Runner | 1 |
| Week 6 | Source Module | 1 |
| Week 6 | Test | 1 |
| Week 6 | Validation | 1 |
| Week 7 | Documentation | 1 |
| Week 7 | Runner | 1 |
| Week 7 | Source Module | 1 |
| Week 7 | Test | 1 |
| Week 7 | Validation | 1 |
| Week 8 | Documentation | 1 |
| Week 8 | Runner | 1 |
| Week 8 | Source Module | 2 |
| Week 8 | Test | 1 |
| Week 8 | Validation | 1 |

Exact SHA-256 hashes are stored in `results/tables/week8_module_inventory.csv` and `data/reporting/week8_reproducibility_manifest.json`.
