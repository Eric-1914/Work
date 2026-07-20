# Week 1 Data Sources

## 1. Sector ETF market data

Historical ETF data are retrieved with the Python `yfinance` package from Yahoo Finance.

Initial tickers:
- XLF
- XLK
- XLE

Requested fields:
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

Collection period:
- Start: 2015-01-01
- End: latest available observation when the script is run

Important usage note:
`yfinance` is an open-source research tool and is not affiliated with or endorsed by Yahoo. Before redistributing or using downloaded Yahoo Finance data in a production/commercial setting, confirm the applicable data-use terms with the project host.

## 2. Real GDP Growth

FRED series ID: `A191RL1Q225SBEA`

Definition:
Real Gross Domestic Product, Percent Change from Preceding Period, Seasonally Adjusted Annual Rate.

Frequency:
Quarterly.

## 3. Consumer Price Index

FRED series ID: `CPIAUCSL`

Definition:
Consumer Price Index for All Urban Consumers: All Items in U.S. City Average.

Frequency:
Monthly.

Note:
This is a CPI index. Percent changes in CPI measure inflation. The Week 2 pipeline will derive an inflation-rate feature, such as year-over-year CPI percent change.

## 4. Federal Funds Effective Rate

FRED series ID: `FEDFUNDS`

Definition:
Federal Funds Effective Rate.

Frequency:
Monthly.

Units:
Percent.

## Storage policy

Week 1 stores source observations in `data/raw/`.

No forward filling, resampling, or time-frequency alignment is applied during Week 1. Those transformations belong to Week 2.
