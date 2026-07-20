# Week 2 Methodology

## Objective

Week 2 implements the required tasks:

1. Clean and align time-series data.
2. Engineer RSI, MACD, and momentum.
3. Engineer rolling returns and rolling volatility.
4. Conduct exploratory data analysis.
5. Conduct a simple sector correlation study.
6. Optionally add macroeconomic regime tags.

## Data Cleaning

Week 1 raw CSV files are treated as immutable source data.

XLF, XLK, and XLE are:
- parsed into a consistent date format,
- deduplicated,
- sorted chronologically,
- converted to numeric fields,
- aligned on common trading dates.

Output:

`data/processed/sector_prices.csv`

## Macroeconomic Alignment

The pipeline uses:
- Real GDP growth,
- CPI,
- CPI-derived year-over-year inflation,
- Federal Funds Effective Rate.

To reduce obvious look-ahead bias, approximate availability lags are used:
- CPI: +1 month
- Federal Funds Rate: +1 month
- GDP growth: +3 months

The lagged observations are backward-aligned to daily ETF trading dates.

Important limitation:
These are approximations. Production-grade backtesting should use exact historical
release dates and vintage data.

Output:

`data/processed/macro_aligned_daily.csv`

## Feature Engineering

For each ETF:

- Daily return
- 5-day rolling return
- 20-day rolling return
- 60-day rolling return
- 20-day momentum
- 20-day annualized rolling volatility
- 60-day annualized rolling volatility
- RSI(14)
- MACD(12,26)
- MACD signal(9)
- MACD histogram

Output:

`data/processed/sector_features.csv`

## Exploratory Data Analysis

Figures:
- Normalized sector prices
- Daily-return correlation matrix
- 20-day rolling return
- 20-day rolling volatility
- 20-day momentum
- RSI(14)
- MACD histogram

Tables:
- Sector return/risk summary
- Sector correlation matrix
- Latest feature snapshot
- Macro regime summary

Report:
- `docs/early_insights_report.md`

The report is generated from the user's actual data after the pipeline runs.

## Optional Macro Regime Tags

Growth:
- Expansion: GDP growth > 0
- Contraction: GDP growth <= 0

Inflation:
- Low: YoY inflation < 2%
- Moderate: 2% <= YoY inflation < 3%
- High: YoY inflation >= 3%

Rates:
- Rising: approximate 63-trading-day rate change > 0.25 percentage points
- Falling: change < -0.25 percentage points
- Stable: otherwise

These are heuristic research labels, not official economic-cycle classifications.

## Leakage Control

Week 2 technical features use only current and historical prices.
Macroeconomic inputs are approximately lagged before alignment.

Future modeling should use time-aware train/validation/test splits rather than
random splitting.


## Output Validation

The final validation stage checks:

- all required Week 2 files exist and are non-empty,
- aligned ETF dates are unique and sorted,
- adjusted-close prices are positive and non-missing,
- all required technical feature columns exist,
- RSI values remain within [0, 100],
- rolling volatility is non-negative,
- key engineered features contain usable observations,
- the sector correlation matrix is numeric and bounded within [-1, 1].

Run:

`python src/validation/validate_week2.py`

The full `python run_week2.py` pipeline runs this validation automatically as its final step.
