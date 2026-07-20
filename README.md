# Machine Learning-Based Sector Rotation Strategy

Innovation AI Internship Project

## Week 1: Data Infrastructure & Acquisition

The Week 1 baseline establishes the data layer for a machine learning-based sector rotation strategy.

### Sector ETF market data

Initial sector ETFs:
- XLF — Financials
- XLK — Technology
- XLE — Energy

Daily fields:
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

### Macroeconomic data

FRED series:
- `A191RL1Q225SBEA` — Real GDP growth, quarterly percent change from the preceding period, seasonally adjusted annual rate
- `CPIAUCSL` — Consumer Price Index for All Urban Consumers: All Items, monthly, seasonally adjusted
- `FEDFUNDS` — Federal Funds Effective Rate, monthly

`CPIAUCSL` is a price index rather than an inflation-rate series. Year-over-year inflation will be derived from the CPI series during Week 2 feature engineering.

### Data storage

Raw observations are stored as CSV files under `data/raw/`.

```text
data/raw/
├── XLF.csv
├── XLK.csv
├── XLE.csv
├── GDP_GROWTH.csv
├── CPI.csv
├── FEDFUNDS.csv
└── data_inventory.csv
```

Time-series cleaning, frequency alignment, technical indicators, rolling returns, and volatility calculations are deferred to Week 2.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Week 1 data collection

```bash
python src/data_collection/collect_week1_data.py
```

## Validate the downloaded data

```bash
python src/data_collection/validate_week1_data.py
```

A successful validation ends with:

```text
[DONE] All Week 1 raw datasets passed validation.
```

## Optional fundamental data

Sector-level P/E and EPS data are not included in the baseline because they are optional in the project specification and consistent historical sector-level datasets may require additional licensed or paid data sources.
