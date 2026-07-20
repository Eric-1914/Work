from __future__ import annotations
from itertools import combinations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT/"data"/"processed"
FIGURE_DIR = PROJECT_ROOT/"results"/"figures"
TABLE_DIR = PROJECT_ROOT/"results"/"tables"
DOCS_DIR = PROJECT_ROOT/"docs"
TICKERS = ["XLF","XLK","XLE"]

def ensure_dirs():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

def max_drawdown(close):
    return float((close/close.cummax()-1).min())

def build_summary(df):
    rows = []
    for t in TICKERS:
        close = df[f"{t}_Adj_Close"].dropna()
        ret = df.loc[close.index, f"{t}_Daily_Return"].dropna()
        cumulative = close.iloc[-1]/close.iloc[0]-1
        years = max((df.loc[close.index[-1],"Date"]-df.loc[close.index[0],"Date"]).days/365.25, 1e-9)
        annualized_return = (1+cumulative)**(1/years)-1 if cumulative > -1 else np.nan
        rows.append({
            "ticker":t,
            "start_date":df.loc[close.index[0],"Date"].date(),
            "end_date":df.loc[close.index[-1],"Date"].date(),
            "cumulative_return":cumulative,
            "annualized_return":annualized_return,
            "annualized_volatility":ret.std()*np.sqrt(252),
            "max_drawdown":max_drawdown(close),
        })
    return pd.DataFrame(rows)

def save_figures(df):
    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        s = df[f"{t}_Adj_Close"].dropna()
        ax.plot(df.loc[s.index,"Date"], s/s.iloc[0]*100, label=t)
    ax.set_title("Normalized Sector ETF Prices (Start = 100)")
    ax.set_xlabel("Date"); ax.set_ylabel("Normalized Value"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"normalized_sector_prices.png", dpi=160); plt.close(fig)

    cols = [f"{t}_Daily_Return" for t in TICKERS]
    corr = df[cols].corr(); corr.index=TICKERS; corr.columns=TICKERS
    corr.to_csv(TABLE_DIR/"sector_correlation.csv")
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(3), labels=TICKERS); ax.set_yticks(range(3), labels=TICKERS)
    ax.set_title("Daily Return Correlation")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(FIGURE_DIR/"sector_return_correlation.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        ax.plot(df["Date"], df[f"{t}_Rolling_Return_20D"], label=t)
    ax.set_title("20-Day Rolling Returns")
    ax.set_xlabel("Date"); ax.set_ylabel("20-Day Return"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"rolling_return_20d.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        ax.plot(df["Date"], df[f"{t}_Rolling_Volatility_20D"], label=t)
    ax.set_title("20-Day Annualized Rolling Volatility")
    ax.set_xlabel("Date"); ax.set_ylabel("Annualized Volatility"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"rolling_volatility_20d.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        ax.plot(df["Date"], df[f"{t}_Momentum_20D"], label=t)
    ax.axhline(0, linestyle="--")
    ax.set_title("20-Day Momentum")
    ax.set_xlabel("Date"); ax.set_ylabel("Momentum"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"momentum_20d.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        ax.plot(df["Date"], df[f"{t}_RSI_14"], label=t)
    ax.axhline(70, linestyle="--"); ax.axhline(30, linestyle="--")
    ax.set_title("14-Day RSI"); ax.set_xlabel("Date"); ax.set_ylabel("RSI"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"rsi_14.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6))
    for t in TICKERS:
        ax.plot(df["Date"], df[f"{t}_MACD_Hist"], label=t)
    ax.axhline(0, linestyle="--")
    ax.set_title("MACD Histogram")
    ax.set_xlabel("Date"); ax.set_ylabel("MACD Histogram"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURE_DIR/"macd_histogram.png", dpi=160); plt.close(fig)

    return corr

def latest_snapshot(df):
    row = df.iloc[-1]
    out = pd.DataFrame([{
        "date":row["Date"].date(),"ticker":t,"adj_close":row[f"{t}_Adj_Close"],
        "rsi_14":row[f"{t}_RSI_14"],"macd":row[f"{t}_MACD"],
        "macd_signal":row[f"{t}_MACD_Signal"],"momentum_20d":row[f"{t}_Momentum_20D"],
        "rolling_return_20d":row[f"{t}_Rolling_Return_20D"],
        "rolling_volatility_20d":row[f"{t}_Rolling_Volatility_20D"],
    } for t in TICKERS])
    out.to_csv(TABLE_DIR/"latest_feature_snapshot.csv", index=False)
    return out

def rsi_label(v):
    if pd.isna(v): return "unavailable"
    if v >= 70: return "overbought-range"
    if v <= 30: return "oversold-range"
    return "neutral-range"

def strongest_pair(corr):
    pairs=[(a,b,float(corr.loc[a,b])) for a,b in combinations(TICKERS,2)]
    return max(pairs, key=lambda x:abs(x[2]))

def write_report(df, summary, corr, snapshot):
    best = summary.loc[summary["cumulative_return"].idxmax()]
    volatile = summary.loc[summary["annualized_volatility"].idxmax()]
    a,b,cval = strongest_pair(corr)
    macro = df["Macro_Regime"].dropna().iloc[-1] if "Macro_Regime" in df and df["Macro_Regime"].notna().any() else "Unavailable"
    lines = []
    for _,r in snapshot.iterrows():
        lines.append(f"- {r['ticker']}: RSI(14) = {r['rsi_14']:.2f} ({rsi_label(r['rsi_14'])}), 20-day momentum = {r['momentum_20d']:.2%}.")
    report = f"""# Week 2 Early Insights Report

## Scope
This report summarizes preliminary descriptive findings from Week 2 data cleaning,
feature engineering, and exploratory data analysis for XLF, XLK, and XLE. It does
not make investment recommendations or train a predictive model.

## 1. Data Cleaning and Alignment
The three ETF series were aligned on common trading dates. Macroeconomic variables
were aligned only after approximate publication lags were applied. CPI year-over-year
inflation was derived from the CPI index.

## 2. Sector Performance and Risk
Across the available sample, **{best['ticker']}** had the highest cumulative return
at **{best['cumulative_return']:.2%}**. **{volatile['ticker']}** had the highest
annualized volatility at **{volatile['annualized_volatility']:.2%}**.

## 3. Sector Correlation
The strongest absolute daily-return correlation was between **{a}** and **{b}**,
with a correlation of **{cval:.3f}**.

## 4. Technical Feature Snapshot
Latest feature date: **{snapshot['date'].iloc[0]}**

{chr(10).join(lines)}

RSI, MACD, momentum, rolling returns, and rolling volatility are candidate model
features only; no single indicator is treated as a trading rule.

## 5. Optional Macroeconomic Regime Tag
Latest heuristic macro regime:

**{macro}**

These regime labels are research tags, not official economic-cycle classifications.

## 6. Overall Early Insight
The Week 2 pipeline establishes a common daily timeline, captures trend/momentum/risk
features, and quantifies cross-sector correlation. The next stage should define a
prediction target and use time-aware train/validation/test splits to avoid look-ahead bias.

## Methodology Limitation
The macro availability lags used here are approximate. Production-grade backtesting
should use exact point-in-time release/vintage data.
"""
    (DOCS_DIR/"early_insights_report.md").write_text(report, encoding="utf-8")

def main():
    ensure_dirs()
    try:
        path = PROCESSED_DIR/"sector_features.csv"
        if not path.exists():
            raise FileNotFoundError("Run cleaning and feature engineering first.")
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        summary = build_summary(df)
        summary.to_csv(TABLE_DIR/"sector_return_summary.csv", index=False)
        corr = save_figures(df)
        snapshot = latest_snapshot(df)
        if "Macro_Regime" in df:
            df["Macro_Regime"].fillna("Unknown").value_counts().rename_axis("Macro_Regime").reset_index(name="count").to_csv(TABLE_DIR/"macro_regime_summary.csv", index=False)
        write_report(df, summary, corr, snapshot)
        print("[PASS] Saved Week 2 figures, tables, and docs/early_insights_report.md")
        print("[DONE] Week 2 EDA and correlation analysis completed.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 2 EDA failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
