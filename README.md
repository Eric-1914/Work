# Machine Learning-Based Sector Rotation Strategy

Innovation AI Internship Project

## Baseline Assets

- XLF — Financials
- XLK — Technology
- XLE — Energy

## Week 1

Week 1 collects ETF historical prices and FRED macroeconomic data.

```bash
python src/data_collection/collect_week1_data.py
python src/data_collection/validate_week1_data.py
```

## Week 2: Data Cleaning, Feature Engineering & EDA

Week 2 performs:

- time-series cleaning and alignment,
- RSI,
- MACD,
- momentum,
- rolling returns,
- rolling volatility,
- exploratory analysis,
- sector correlation analysis,
- optional macroeconomic regime tagging.

Run the full pipeline:

```bash
python run_week2.py
```

Or run each step:

```bash
python src/data_cleaning/clean_data.py
python src/feature_engineering/build_features.py
python src/eda/run_eda.py
python src/validation/validate_week2.py
```

Expected outputs:

```text
data/processed/
├── sector_prices.csv
├── macro_aligned_daily.csv
├── cleaning_summary.csv
├── sector_features.csv
└── feature_dictionary.csv

results/figures/
├── normalized_sector_prices.png
├── sector_return_correlation.png
├── rolling_return_20d.png
├── rolling_volatility_20d.png
├── momentum_20d.png
├── rsi_14.png
└── macd_histogram.png

results/tables/
├── sector_return_summary.csv
├── sector_correlation.csv
├── latest_feature_snapshot.csv
└── macro_regime_summary.csv

docs/
├── data_sources.md
├── week2_methodology.md
└── early_insights_report.md
```

Important:
Macroeconomic observations are aligned with approximate publication lags.
Production-grade backtests should use exact point-in-time release/vintage data.
