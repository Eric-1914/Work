"""Week 7 - Backtesting & Performance Evaluation.

Backtests the final Week 6 optimized sector portfolio with realistic monthly
execution, transaction costs, slippage, benchmark comparison, rolling
validation, and trading-friction sensitivity analysis.

The design intentionally preserves the Week 1-6 pipeline:
- Week 4 creates model probabilities/ranks.
- Week 5 converts them into portfolio targets.
- Week 6 applies risk management and optimization.
- Week 7 evaluates the saved Week 6 targets; it does not retrain or re-optimize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEEK6_PATH = PROJECT_ROOT / "results" / "tables" / "week6_optimized_portfolio.csv"
SECTOR_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sector_features.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "data" / "backtesting"
CONFIG_PATH = CONFIG_DIR / "week7_backtest_config.json"

SECTORS = ("XLF", "XLK", "XLE")
CASH = "CASH"
ASSETS = (*SECTORS, CASH)
ANNUALIZATION = 252
TOL = 1e-10

# Transparent, configurable implementation assumptions. The assignment requires
# realistic costs/slippage but does not prescribe exact values.
TRANSACTION_COST_BPS = 5.0
SLIPPAGE_BPS = 5.0
ROLLING_WINDOW_DAYS = 126
ROLLING_STEP_DAYS = 21
MIN_ROLLING_DAYS = 63
COST_SENSITIVITY_BPS = (0.0, 5.0, 10.0, 20.0, 40.0)

STRATEGY = "Week7_Strategy_Net"
SPY_BENCHMARK = "SPY_Benchmark"
EQUAL_WEIGHT_BENCHMARK = "Equal_Weighted_Sectors"

WEEK6_REQUIRED = {"Signal_Date", "Asset", "Target_Weight"}


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    rebalances: pd.DataFrame
    schedule: pd.DataFrame


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_week6_targets(path: Path = WEEK6_PATH) -> pd.DataFrame:
    """Load the final Week 6 optimized monthly target portfolio."""
    if not path.exists():
        raise FileNotFoundError(
            "results/tables/week6_optimized_portfolio.csv is missing. Complete Week 6 first."
        )
    data = pd.read_csv(path, parse_dates=["Signal_Date"])
    _require_columns(data, WEEK6_REQUIRED, "Week 6 optimized portfolio")
    if data.empty or data["Signal_Date"].isna().any():
        raise ValueError("Week 6 optimized portfolio is empty or has invalid Signal_Date values.")

    data["Asset"] = data["Asset"].astype(str).str.strip()
    if not set(data["Asset"].unique()).issubset(set(ASSETS)):
        raise ValueError("Week 6 optimized portfolio contains an unexpected asset.")
    data["Target_Weight"] = pd.to_numeric(data["Target_Weight"], errors="coerce")
    if data["Target_Weight"].isna().any() or not np.isfinite(data["Target_Weight"]).all():
        raise ValueError("Week 6 Target_Weight must be finite numeric values.")
    if (data["Target_Weight"] < -TOL).any():
        raise ValueError("Week 6 Target_Weight cannot be negative.")

    for date, block in data.groupby("Signal_Date", sort=True):
        if set(block["Asset"]) != set(ASSETS) or len(block) != len(ASSETS):
            raise ValueError(f"Signal date {date.date()} must contain XLF, XLK, XLE, and CASH exactly once.")
        total = float(block["Target_Weight"].sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Week 6 weights on {date.date()} do not sum to 1.0.")
        if float(block.loc[block["Asset"].isin(SECTORS), "Target_Weight"].max()) > 1.0 + TOL:
            raise ValueError("Invalid sector target weight greater than 100%.")

    if data["Signal_Date"].nunique() < 2:
        raise ValueError("Week 7 requires at least two Week 6 signal dates.")
    return data.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)


def load_sector_returns(path: Path = SECTOR_FEATURES_PATH) -> pd.DataFrame:
    """Load complete daily returns for XLF, XLK and XLE."""
    if not path.exists():
        raise FileNotFoundError(
            "data/processed/sector_features.csv is missing. Week 7 needs daily sector returns."
        )
    required = {"Date", *(f"{sector}_Daily_Return" for sector in SECTORS)}
    raw = pd.read_csv(path, parse_dates=["Date"])
    _require_columns(raw, required, "sector_features.csv")
    if raw.empty or raw["Date"].isna().any():
        raise ValueError("sector_features.csv is empty or contains invalid dates.")

    out = pd.DataFrame({"Date": raw["Date"]})
    for sector in SECTORS:
        out[sector] = pd.to_numeric(raw[f"{sector}_Daily_Return"], errors="coerce")
    out = out.sort_values("Date").drop_duplicates("Date", keep="last")
    out = out.dropna(subset=list(SECTORS)).reset_index(drop=True)
    finite = np.isfinite(out[list(SECTORS)].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[finite].reset_index(drop=True)
    if out.empty:
        raise ValueError("No complete daily sector return observations are available.")
    if (out[list(SECTORS)] <= -1.0).any().any():
        raise ValueError("Daily sector returns must be greater than -100%.")
    return out


def _return_from_price(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.pct_change(fill_method=None)


def _extract_spy_from_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Extract SPY daily returns from common wide or long CSV layouts."""
    cols = {str(c).strip().lower(): c for c in frame.columns}
    date_col = next((cols[k] for k in ("date", "datetime", "timestamp") if k in cols), None)
    if date_col is None:
        return None

    parsed_dates = pd.to_datetime(frame[date_col], errors="coerce")
    direct_return_names = (
        "spy_daily_return", "spy_return", "spy_returns", "spy_ret", "spy_pct_return"
    )
    for key in direct_return_names:
        if key in cols:
            ret = pd.to_numeric(frame[cols[key]], errors="coerce")
            out = pd.DataFrame({"Date": parsed_dates, "SPY": ret}).dropna()
            return out if not out.empty else None

    price_names = ("spy_adj_close", "spy_adjusted_close", "spy_close", "spy")
    for key in price_names:
        if key in cols:
            price = pd.to_numeric(frame[cols[key]], errors="coerce")
            temp = pd.DataFrame({"Date": parsed_dates, "Price": price}).dropna().sort_values("Date")
            if len(temp) >= 2:
                temp["SPY"] = _return_from_price(temp["Price"])
                out = temp[["Date", "SPY"]].dropna()
                return out if not out.empty else None

    ticker_col = next((cols[k] for k in ("ticker", "symbol", "asset", "etf") if k in cols), None)
    if ticker_col is not None:
        spy = frame[frame[ticker_col].astype(str).str.upper().str.strip().eq("SPY")].copy()
        if not spy.empty:
            spy_dates = pd.to_datetime(spy[date_col], errors="coerce")
            return_col = next((cols[k] for k in ("daily_return", "return", "returns", "ret") if k in cols), None)
            if return_col is not None:
                out = pd.DataFrame({"Date": spy_dates, "SPY": pd.to_numeric(spy[return_col], errors="coerce")}).dropna()
                return out if not out.empty else None
            price_col = next((cols[k] for k in ("adj close", "adj_close", "adjusted_close", "close") if k in cols), None)
            if price_col is not None:
                temp = pd.DataFrame({"Date": spy_dates, "Price": pd.to_numeric(spy[price_col], errors="coerce")}).dropna().sort_values("Date")
                if len(temp) >= 2:
                    temp["SPY"] = _return_from_price(temp["Price"])
                    out = temp[["Date", "SPY"]].dropna()
                    return out if not out.empty else None
    return None


def load_spy_returns(
    required_start: pd.Timestamp | None = None,
    required_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    """Find a real SPY series with adequate coverage for the backtest period."""
    candidates: list[Path] = [SECTOR_FEATURES_PATH]
    data_root = PROJECT_ROOT / "data"
    if data_root.exists():
        candidates.extend(sorted(p for p in data_root.rglob("*.csv") if p != SECTOR_FEATURES_PATH))

    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        extracted = _extract_spy_from_frame(frame)
        if extracted is None:
            continue
        extracted = extracted.sort_values("Date").drop_duplicates("Date", keep="last").dropna()
        finite = np.isfinite(extracted["SPY"].to_numpy(dtype=float))
        extracted = extracted.loc[finite].reset_index(drop=True)
        if not extracted.empty and (extracted["SPY"] <= -1.0).any():
            raise ValueError(f"SPY returns discovered in {path} contain values <= -100%.")
        if len(extracted) >= 20:
            if required_start is not None and required_end is not None:
                first = pd.Timestamp(extracted["Date"].min())
                last = pd.Timestamp(extracted["Date"].max())
                # Reject stale/partial local files rather than silently shortening
                # the comparison period. A few calendar days of tolerance covers
                # weekends and exchange holidays around the endpoints.
                if first > pd.Timestamp(required_start) + pd.Timedelta(days=7):
                    continue
                if last < pd.Timestamp(required_end) - pd.Timedelta(days=7):
                    continue
            try:
                source = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                source = str(path)
            return extracted, source

    # Week 1 already uses yfinance in this project. If SPY was not saved in an
    # earlier local data file, fetch it once and cache a normalized return file
    # for reproducibility. A real SPY series is always used; no synthetic proxy
    # is substituted.
    try:
        import yfinance as yf

        sector_dates = pd.read_csv(SECTOR_FEATURES_PATH, usecols=["Date"])
        parsed = pd.to_datetime(sector_dates["Date"], errors="coerce").dropna()
        if parsed.empty:
            raise ValueError("Cannot determine SPY download dates from sector_features.csv.")
        start = (parsed.min() - pd.Timedelta(days=10)).date().isoformat()
        end = (parsed.max() + pd.Timedelta(days=3)).date().isoformat()
        history = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=False)
        if history.empty:
            raise RuntimeError("yfinance returned no SPY history.")
        price_col = "Adj Close" if "Adj Close" in history.columns else "Close"
        temp = history.reset_index()
        date_col = "Date" if "Date" in temp.columns else temp.columns[0]
        temp["Date"] = pd.to_datetime(temp[date_col], errors="coerce").dt.tz_localize(None)
        temp["SPY"] = pd.to_numeric(temp[price_col], errors="coerce").pct_change(fill_method=None)
        extracted = temp[["Date", "SPY"]].dropna().sort_values("Date").drop_duplicates("Date", keep="last")
        if len(extracted) < 20:
            raise RuntimeError("Downloaded SPY history is too short.")
        cache = PROJECT_ROOT / "data" / "processed" / "spy_benchmark.csv"
        cache.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(columns={"SPY": "SPY_Daily_Return"}).to_csv(cache, index=False, date_format="%Y-%m-%d")
        return extracted.reset_index(drop=True), str(cache.relative_to(PROJECT_ROOT)) + " (downloaded via yfinance)"
    except Exception as exc:
        raise FileNotFoundError(
            "Week 7 requires a real SPY benchmark. No local SPY series was found and the yfinance fallback failed. "
            f"Check internet access/yfinance, or add SPY daily data under data/. Original error: {exc}"
        ) from exc


def _target_lookup(targets: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    lookup: dict[pd.Timestamp, dict[str, float]] = {}
    for date, block in targets.groupby("Signal_Date", sort=True):
        weights = block.set_index("Asset")["Target_Weight"].astype(float).to_dict()
        lookup[pd.Timestamp(date)] = {asset: float(weights.get(asset, 0.0)) for asset in ASSETS}
    return lookup


def _execution_schedule(targets: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates).sort_values().unique())
    rows = []
    for signal_date in sorted(pd.Timestamp(v) for v in targets["Signal_Date"].unique()):
        later = dates[dates > signal_date]
        if len(later) == 0:
            raise ValueError(
                f"No trading session exists after Week 6 signal date {signal_date.date()}. Refresh daily data before running Week 7."
            )
        execution_date = pd.Timestamp(later[0])
        effective_candidates = dates[dates > execution_date]
        if len(effective_candidates) == 0:
            raise ValueError(
                f"Week 7 needs at least two trading observations after signal date {signal_date.date()} "
                "(one to execute after the close and one to measure the first post-trade return). "
                "Refresh daily data before running Week 7."
            )
        effective_date = pd.Timestamp(effective_candidates[0])
        rows.append({
            "Signal_Date": signal_date,
            "Execution_Date": execution_date,
            "Effective_From_Date": effective_date,
            "Execution_Assumption": "Trade after execution-date close; new weights earn returns from the following trading observation",
        })
    return pd.DataFrame(rows)


def _drift_weights(weights: dict[str, float], row: pd.Series, portfolio_return: float) -> dict[str, float]:
    denominator = 1.0 + portfolio_return
    if denominator <= TOL:
        raise ValueError("Portfolio value became non-positive during weight drift.")
    drifted = {}
    for sector in SECTORS:
        drifted[sector] = float(weights[sector]) * (1.0 + float(row[sector])) / denominator
    drifted[CASH] = float(weights[CASH]) / denominator
    total = sum(drifted.values())
    return {asset: float(drifted[asset] / total) for asset in ASSETS}


def _one_way_turnover(old: dict[str, float], new: dict[str, float]) -> float:
    return 0.5 * float(sum(abs(float(new[a]) - float(old[a])) for a in ASSETS))


def _risky_trade_notional(old: dict[str, float], new: dict[str, float]) -> float:
    """Sum of absolute risky-asset trades; cash itself does not incur trading cost."""
    return float(sum(abs(float(new[s]) - float(old[s])) for s in SECTORS))


def run_strategy_backtest(
    targets: pd.DataFrame,
    sector_returns: pd.DataFrame,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
) -> BacktestResult:
    """Run a conservative close-to-close monthly backtest with drifting weights.

    A signal produced at the end of Signal_Date is executed after the close of
    the first later trading date. Therefore the new weights become effective for
    the following daily return observation. This avoids claiming an execution-
    day close-to-close return that could not have been fully held without open
    or intraday execution data.
    """
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Transaction-cost and slippage assumptions cannot be negative.")

    schedule = _execution_schedule(targets, sector_returns["Date"])
    target_by_signal = _target_lookup(targets)
    execution_to_signal = {
        pd.Timestamp(row.Execution_Date): pd.Timestamp(row.Signal_Date)
        for row in schedule.itertuples(index=False)
    }

    first_execution = pd.Timestamp(schedule["Execution_Date"].min())
    data = sector_returns[sector_returns["Date"] >= first_execution].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No sector returns are available for the Week 7 backtest period.")

    weights = {**{s: 0.0 for s in SECTORS}, CASH: 1.0}
    equity = 1.0
    peak = 1.0
    daily_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []

    for _, row in data.iterrows():
        date = pd.Timestamp(row["Date"])
        equity_start = equity
        gross_return = float(sum(weights[s] * float(row[s]) for s in SECTORS))
        equity_after_market = equity_start * (1.0 + gross_return)
        weights_after_market = _drift_weights(weights, row, gross_return)

        rebalanced = False
        signal_date: pd.Timestamp | pd.NaT = pd.NaT
        one_way_turnover = 0.0
        risky_trade_notional = 0.0
        transaction_cost_fraction = 0.0
        slippage_fraction = 0.0
        total_cost_fraction = 0.0

        if date in execution_to_signal:
            signal_date = execution_to_signal[date]
            new_weights = target_by_signal[signal_date]
            one_way_turnover = _one_way_turnover(weights_after_market, new_weights)
            risky_trade_notional = _risky_trade_notional(weights_after_market, new_weights)
            transaction_cost_fraction = risky_trade_notional * transaction_cost_bps / 10000.0
            slippage_fraction = risky_trade_notional * slippage_bps / 10000.0
            total_cost_fraction = transaction_cost_fraction + slippage_fraction
            if total_cost_fraction >= 1.0:
                raise ValueError("Trading-friction assumptions consume 100% or more of portfolio value.")
            equity = equity_after_market * (1.0 - total_cost_fraction)
            weights = new_weights.copy()
            rebalanced = True
            rebalance_rows.append({
                "Signal_Date": signal_date,
                "Execution_Date": date,
                "One_Way_Turnover": one_way_turnover,
                "Risky_Trade_Notional": risky_trade_notional,
                "Transaction_Cost_Bps": transaction_cost_bps,
                "Slippage_Bps": slippage_bps,
                "Transaction_Cost_Fraction": transaction_cost_fraction,
                "Slippage_Fraction": slippage_fraction,
                "Total_Trading_Cost_Fraction": total_cost_fraction,
                "Gross_Exposure_After_Rebalance": float(sum(new_weights[s] for s in SECTORS)),
                **{f"PreTrade_{asset}": float(weights_after_market[asset]) for asset in ASSETS},
                **{f"Target_{asset}": float(new_weights[asset]) for asset in ASSETS},
            })
        else:
            equity = equity_after_market
            weights = weights_after_market

        if equity <= 0:
            raise ValueError("Backtest equity became non-positive.")
        peak = max(peak, equity)
        net_return = equity / equity_start - 1.0
        daily_rows.append({
            "Date": date,
            "Strategy": STRATEGY,
            "Gross_Return": gross_return,
            "Net_Return": float(net_return),
            "Trading_Cost_Drag": float(gross_return - net_return),
            "Equity": float(equity),
            "Drawdown": float(equity / peak - 1.0),
            "Gross_Exposure": float(sum(weights[s] for s in SECTORS)),
            "Rebalanced": rebalanced,
            "Signal_Date": signal_date,
            "One_Way_Turnover": one_way_turnover,
            "Risky_Trade_Notional": risky_trade_notional,
            "Transaction_Cost_Fraction": transaction_cost_fraction,
            "Slippage_Fraction": slippage_fraction,
            "Total_Trading_Cost_Fraction": total_cost_fraction,
        })

    return BacktestResult(
        daily=pd.DataFrame(daily_rows),
        rebalances=pd.DataFrame(rebalance_rows),
        schedule=schedule,
    )


def _compound_equity(returns: pd.Series) -> pd.Series:
    values = pd.to_numeric(returns, errors="raise").astype(float)
    if (values <= -1.0).any():
        raise ValueError("A return <= -100% would make equity non-positive.")
    return (1.0 + values).cumprod()


def _drawdown_from_equity(equity: pd.Series) -> pd.Series:
    """Compute drawdown relative to the initial $1 capital and subsequent peaks."""
    eq = pd.to_numeric(equity, errors="raise").astype(float)
    peaks = eq.cummax().clip(lower=1.0)
    return eq / peaks - 1.0


def _run_equal_weight_benchmark(
    sector_returns: pd.DataFrame,
    execution_dates: set[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Frictionless monthly equal-weight benchmark using the strategy's execution dates.

    The benchmark starts from cash on the first execution date, rebalances to
    one-third per sector after that close, and therefore earns its first sector
    return on the following trading observation. Subsequent monthly rebalances
    use the same after-close convention. This avoids giving the benchmark an
    execution-timing advantage over the strategy.
    """
    data = sector_returns[
        (sector_returns["Date"] >= start_date) & (sector_returns["Date"] <= end_date)
    ].copy().sort_values("Date").reset_index(drop=True)
    if data.empty:
        raise ValueError("No sector returns are available for the equal-weight benchmark period.")

    weights = {s: 0.0 for s in SECTORS}
    weights[CASH] = 1.0
    equity = 1.0
    rows: list[dict[str, object]] = []
    for _, row in data.iterrows():
        date = pd.Timestamp(row["Date"])
        ret = float(sum(weights[s] * float(row[s]) for s in SECTORS))
        equity *= 1.0 + ret
        weights = _drift_weights(weights, row, ret)
        rebalanced = date in execution_dates
        if rebalanced:
            weights = {s: 1.0 / len(SECTORS) for s in SECTORS}
            weights[CASH] = 0.0
        rows.append({
            "Date": date,
            "Equal_Weighted_Sector_Return": ret,
            "Equal_Weighted_Sectors_Equity": equity,
            "Equal_Weighted_Rebalanced": rebalanced,
        })
    out = pd.DataFrame(rows)
    out["Equal_Weighted_Sectors_Drawdown"] = _drawdown_from_equity(out["Equal_Weighted_Sectors_Equity"])
    return out


def build_benchmarks(
    sector_returns: pd.DataFrame,
    spy_returns: pd.DataFrame,
    backtest: BacktestResult,
) -> pd.DataFrame:
    """Build execution-aligned SPY and monthly equal-weight-sector benchmarks."""
    start_date = pd.Timestamp(backtest.daily["Date"].min())
    end_date = pd.Timestamp(backtest.daily["Date"].max())
    strategy_dates = pd.DataFrame({"Date": pd.to_datetime(backtest.daily["Date"])})

    spy = spy_returns[
        (spy_returns["Date"] >= start_date) & (spy_returns["Date"] <= end_date)
    ][["Date", "SPY"]].copy()
    merged_spy = strategy_dates.merge(spy, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    if len(merged_spy) < 20:
        raise ValueError("Too few overlapping SPY observations for Week 7 evaluation.")

    # Fair initial timing: both strategy and benchmarks are funded before the
    # first execution, trade after that close, and earn market returns from the
    # next observation. SPY is buy-and-hold after that initial close.
    merged_spy.loc[0, "SPY"] = 0.0
    merged_spy["SPY_Equity"] = _compound_equity(merged_spy["SPY"])
    merged_spy["SPY_Drawdown"] = _drawdown_from_equity(merged_spy["SPY_Equity"])

    execution_dates = set(pd.to_datetime(backtest.schedule["Execution_Date"]))
    ew = _run_equal_weight_benchmark(sector_returns, execution_dates, start_date, end_date)
    out = merged_spy.merge(ew, on="Date", how="inner")
    if len(out) < 20:
        raise ValueError("Too few common strategy/SPY/equal-weight benchmark observations.")
    return out[[
        "Date", "SPY", "SPY_Equity", "SPY_Drawdown",
        "Equal_Weighted_Sector_Return", "Equal_Weighted_Sectors_Equity",
        "Equal_Weighted_Sectors_Drawdown", "Equal_Weighted_Rebalanced",
    ]]

def _performance_metrics(
    name: str,
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    turnover_values: pd.Series | None = None,
    total_cost_fraction: float = 0.0,
) -> dict[str, object]:
    r = pd.to_numeric(returns, errors="raise").astype(float)
    eq = pd.to_numeric(equity, errors="raise").astype(float)
    dd = pd.to_numeric(drawdown, errors="raise").astype(float)
    n = len(r)
    if n == 0:
        raise ValueError(f"No returns available for {name}.")

    total_return = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (ANNUALIZATION / n) - 1.0) if eq.iloc[-1] > 0 else -1.0
    daily_std = float(r.std(ddof=1)) if n > 1 else 0.0
    ann_vol = daily_std * math.sqrt(ANNUALIZATION)
    sharpe = float(r.mean() / daily_std * math.sqrt(ANNUALIZATION)) if daily_std > TOL else np.nan
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    downside_daily = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(r.mean() / downside_daily * math.sqrt(ANNUALIZATION)) if downside_daily > TOL else np.nan
    win_rate = float((r > 0).mean())
    max_drawdown = float(dd.min())

    if turnover_values is not None and len(turnover_values) > 0:
        turnover = pd.to_numeric(turnover_values, errors="raise").astype(float)
        total_turnover = float(turnover.sum())
        average_turnover = float(turnover.mean())
        years = n / ANNUALIZATION
        annualized_turnover = float(total_turnover / years) if years > 0 else np.nan
        rebalance_count = int(len(turnover))
    else:
        total_turnover = 0.0
        average_turnover = 0.0
        annualized_turnover = 0.0
        rebalance_count = 0

    return {
        "Strategy": name,
        "Daily_Observations": n,
        "Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Volatility": ann_vol,
        "Sharpe_0RF": sharpe,
        "Sortino_0MAR": sortino,
        "Max_Drawdown": max_drawdown,
        "Daily_Win_Rate": win_rate,
        "Rebalance_Count": rebalance_count,
        "Total_One_Way_Turnover": total_turnover,
        "Average_One_Way_Turnover": average_turnover,
        "Annualized_One_Way_Turnover": annualized_turnover,
        "Total_Trading_Cost_Fraction": float(total_cost_fraction),
    }


def build_performance_summary(
    backtest: BacktestResult,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    strategy_dates = backtest.daily[["Date"]].copy()
    aligned = strategy_dates.merge(benchmarks, on="Date", how="inner")
    strategy = backtest.daily[backtest.daily["Date"].isin(aligned["Date"])].copy().sort_values("Date")
    aligned = aligned.sort_values("Date").reset_index(drop=True)
    strategy = strategy.reset_index(drop=True)
    if len(strategy) != len(aligned):
        raise RuntimeError("Strategy/benchmark date alignment failed.")

    # Rebase strategy equity to the aligned benchmark start so every metric uses
    # exactly the same date range.
    net_returns = strategy["Net_Return"].astype(float)
    strategy_equity = _compound_equity(net_returns)
    strategy_dd = _drawdown_from_equity(strategy_equity)
    rebal = backtest.rebalances[backtest.rebalances["Execution_Date"].isin(aligned["Date"])]
    total_cost = float(rebal["Total_Trading_Cost_Fraction"].sum()) if not rebal.empty else 0.0

    rows = [
        _performance_metrics(
            STRATEGY, net_returns, strategy_equity, strategy_dd,
            rebal["One_Way_Turnover"] if not rebal.empty else None,
            total_cost,
        ),
        _performance_metrics(
            SPY_BENCHMARK, aligned["SPY"], aligned["SPY_Equity"], aligned["SPY_Drawdown"]
        ),
        _performance_metrics(
            EQUAL_WEIGHT_BENCHMARK,
            aligned["Equal_Weighted_Sector_Return"],
            aligned["Equal_Weighted_Sectors_Equity"],
            aligned["Equal_Weighted_Sectors_Drawdown"],
        ),
    ]
    return pd.DataFrame(rows)


def build_rolling_validation(backtest: BacktestResult) -> pd.DataFrame:
    """Evaluate realized strategy robustness over trailing six-month windows."""
    data = backtest.daily.sort_values("Date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    if len(data) < MIN_ROLLING_DAYS:
        return pd.DataFrame(columns=[
            "Window_Start", "Window_End", "Observations", "Total_Return", "CAGR",
            "Sharpe_0RF", "Sortino_0MAR", "Max_Drawdown", "Daily_Win_Rate"
        ])

    endpoints = list(range(MIN_ROLLING_DAYS - 1, len(data), ROLLING_STEP_DAYS))
    if endpoints[-1] != len(data) - 1:
        endpoints.append(len(data) - 1)
    for end_idx in endpoints:
        start_idx = max(0, end_idx - ROLLING_WINDOW_DAYS + 1)
        window = data.iloc[start_idx:end_idx + 1].copy()
        if len(window) < MIN_ROLLING_DAYS:
            continue
        r = window["Net_Return"].astype(float).reset_index(drop=True)
        eq = _compound_equity(r)
        dd = _drawdown_from_equity(eq)
        m = _performance_metrics("Rolling", r, eq, dd)
        rows.append({
            "Window_Start": pd.Timestamp(window["Date"].iloc[0]),
            "Window_End": pd.Timestamp(window["Date"].iloc[-1]),
            "Observations": len(window),
            "Total_Return": m["Total_Return"],
            "CAGR": m["CAGR"],
            "Sharpe_0RF": m["Sharpe_0RF"],
            "Sortino_0MAR": m["Sortino_0MAR"],
            "Max_Drawdown": m["Max_Drawdown"],
            "Daily_Win_Rate": m["Daily_Win_Rate"],
        })
    return pd.DataFrame(rows)


def build_cost_sensitivity(targets: pd.DataFrame, sector_returns: pd.DataFrame) -> pd.DataFrame:
    """Stress-test the strategy across total trading-friction assumptions."""
    rows = []
    for total_bps in COST_SENSITIVITY_BPS:
        half = total_bps / 2.0
        result = run_strategy_backtest(targets, sector_returns, half, half)
        r = result.daily["Net_Return"].astype(float)
        eq = _compound_equity(r)
        dd = _drawdown_from_equity(eq)
        m = _performance_metrics(
            f"Cost_{total_bps:g}bps", r, eq, dd,
            result.rebalances["One_Way_Turnover"] if not result.rebalances.empty else None,
            float(result.rebalances["Total_Trading_Cost_Fraction"].sum()) if not result.rebalances.empty else 0.0,
        )
        rows.append({
            "Total_Friction_Bps_Per_Risky_Dollar_Traded": total_bps,
            "Transaction_Cost_Bps": half,
            "Slippage_Bps": half,
            "Total_Return": m["Total_Return"],
            "CAGR": m["CAGR"],
            "Sharpe_0RF": m["Sharpe_0RF"],
            "Sortino_0MAR": m["Sortino_0MAR"],
            "Max_Drawdown": m["Max_Drawdown"],
            "Total_Trading_Cost_Fraction": m["Total_Trading_Cost_Fraction"],
        })
    return pd.DataFrame(rows)


def save_figures(
    backtest: BacktestResult,
    benchmarks: pd.DataFrame,
    rolling: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    merged = backtest.daily[["Date", "Net_Return"]].merge(benchmarks, on="Date", how="inner").sort_values("Date")
    merged["Strategy_Equity"] = _compound_equity(merged["Net_Return"])
    merged["Strategy_Drawdown"] = _drawdown_from_equity(merged["Strategy_Equity"])

    ax = merged.set_index("Date")[["Strategy_Equity", "SPY_Equity", "Equal_Weighted_Sectors_Equity"]].plot(figsize=(10, 5))
    ax.set_title("Week 7 Backtest: Strategy vs Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week7_equity_curve.png", dpi=160)
    plt.close()

    ax = merged.set_index("Date")[["Strategy_Drawdown", "SPY_Drawdown", "Equal_Weighted_Sectors_Drawdown"]].plot(figsize=(10, 5))
    ax.set_title("Week 7 Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week7_drawdown_comparison.png", dpi=160)
    plt.close()

    if not rolling.empty:
        ax = rolling.plot(x="Window_End", y="Sharpe_0RF", marker="o", figsize=(10, 4))
        ax.axhline(0.0, linewidth=1)
        ax.set_title("Week 7 Rolling Sharpe Validation")
        ax.set_xlabel("Window End")
        ax.set_ylabel("Rolling Sharpe")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "week7_rolling_sharpe.png", dpi=160)
        plt.close()

    ax = sensitivity.plot(
        x="Total_Friction_Bps_Per_Risky_Dollar_Traded", y="CAGR", marker="o", figsize=(8, 4)
    )
    ax.set_title("Week 7 Trading-Cost Sensitivity")
    ax.set_xlabel("Total Friction (bps per risky dollar traded)")
    ax.set_ylabel("CAGR")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week7_cost_sensitivity.png", dpi=160)
    plt.close()


def save_config(spy_source: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "week6_target_input": str(WEEK6_PATH.relative_to(PROJECT_ROOT)),
        "sector_return_input": str(SECTOR_FEATURES_PATH.relative_to(PROJECT_ROOT)),
        "spy_source": spy_source,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "total_friction_bps_per_risky_dollar_traded": TRANSACTION_COST_BPS + SLIPPAGE_BPS,
        "execution_rule": "Signal after close; trade after next trading-session close; new target effective from following daily return observation",
        "benchmark_treatment": "SPY is buy-and-hold after the common initial execution close; equal-weight sectors rebalance monthly on the same execution dates; both benchmarks are frictionless",
        "rolling_window_days": ROLLING_WINDOW_DAYS,
        "rolling_step_days": ROLLING_STEP_DAYS,
        "cost_sensitivity_total_bps": list(COST_SENSITIVITY_BPS),
        "annualization_days": ANNUALIZATION,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * value:.2f}%"


def _fmt_num(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def save_report(summary: pd.DataFrame, rolling: pd.DataFrame, sensitivity: pd.DataFrame, spy_source: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lookup = summary.set_index("Strategy")
    s = lookup.loc[STRATEGY]
    spy = lookup.loc[SPY_BENCHMARK]
    ew = lookup.loc[EQUAL_WEIGHT_BENCHMARK]
    if not rolling.empty:
        positive_rolling = float((rolling["Sharpe_0RF"] > 0).mean())
        rolling_text = f"{positive_rolling:.1%} of reported rolling windows had a positive Sharpe ratio."
    else:
        rolling_text = "The available history was too short for the configured rolling validation window."
    worst_cost = sensitivity.sort_values("Total_Friction_Bps_Per_Risky_Dollar_Traded").iloc[-1]

    lines = [
        "# Week 7 Backtesting & Performance Evaluation Report",
        "",
        "## Backtest design",
        "",
        f"The final Week 6 optimized portfolio was backtested with monthly execution, {TRANSACTION_COST_BPS:.1f} bps transaction cost and {SLIPPAGE_BPS:.1f} bps slippage per risky dollar traded.",
        "Signals are executed conservatively after the close of the next available trading session, and the new weights become effective on the following daily return observation.",
        f"SPY benchmark data source: `{spy_source}`.",
        "SPY is treated as a frictionless buy-and-hold benchmark after the common initial execution close. The equal-weight sector benchmark is frictionless and rebalanced monthly on the same execution dates as the strategy.",
        "",
        "## Main results",
        "",
        f"- Week 7 strategy CAGR: {_fmt_pct(float(s['CAGR']))}",
        f"- Week 7 strategy Sharpe: {_fmt_num(float(s['Sharpe_0RF']))}",
        f"- Week 7 strategy Sortino: {_fmt_num(float(s['Sortino_0MAR']))}",
        f"- Week 7 strategy maximum drawdown: {_fmt_pct(float(s['Max_Drawdown']))}",
        f"- Week 7 strategy daily win rate: {_fmt_pct(float(s['Daily_Win_Rate']))}",
        f"- Average one-way turnover per rebalance: {_fmt_pct(float(s['Average_One_Way_Turnover']))}",
        f"- SPY CAGR: {_fmt_pct(float(spy['CAGR']))}; Sharpe: {_fmt_num(float(spy['Sharpe_0RF']))}; max drawdown: {_fmt_pct(float(spy['Max_Drawdown']))}",
        f"- Equal-weight sectors CAGR: {_fmt_pct(float(ew['CAGR']))}; Sharpe: {_fmt_num(float(ew['Sharpe_0RF']))}; max drawdown: {_fmt_pct(float(ew['Max_Drawdown']))}",
        "",
        "## Robustness checks",
        "",
        f"{rolling_text}",
        f"At the highest tested trading-friction assumption ({float(worst_cost['Total_Friction_Bps_Per_Risky_Dollar_Traded']):.0f} bps), strategy CAGR was {_fmt_pct(float(worst_cost['CAGR']))}.",
        "",
        "## Interpretation",
        "",
        "Week 7 is an evaluation layer. It does not change Week 6 portfolio targets or tune parameters using future returns. Results should therefore be interpreted as historical evidence, not as a guarantee of future performance.",
    ]
    (DOCS_DIR / "week7_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    backtest: BacktestResult,
    benchmarks: pd.DataFrame,
    summary: pd.DataFrame,
    rolling: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    backtest.daily.to_csv(TABLE_DIR / "week7_daily_backtest.csv", index=False, date_format="%Y-%m-%d")
    backtest.rebalances.to_csv(TABLE_DIR / "week7_rebalance_log.csv", index=False, date_format="%Y-%m-%d")
    backtest.schedule.to_csv(TABLE_DIR / "week7_execution_schedule.csv", index=False, date_format="%Y-%m-%d")
    benchmarks.to_csv(TABLE_DIR / "week7_benchmark_daily.csv", index=False, date_format="%Y-%m-%d")
    summary.to_csv(TABLE_DIR / "week7_performance_summary.csv", index=False)
    rolling.to_csv(TABLE_DIR / "week7_rolling_validation.csv", index=False, date_format="%Y-%m-%d")
    sensitivity.to_csv(TABLE_DIR / "week7_cost_sensitivity.csv", index=False)


def main() -> int:
    try:
        targets = load_week6_targets()
        sector_returns = load_sector_returns()
        backtest = run_strategy_backtest(targets, sector_returns)
        required_start = pd.Timestamp(backtest.daily["Date"].min())
        required_end = pd.Timestamp(backtest.daily["Date"].max())
        spy_returns, spy_source = load_spy_returns(required_start, required_end)
        benchmarks = build_benchmarks(sector_returns, spy_returns, backtest)
        summary = build_performance_summary(backtest, benchmarks)
        rolling = build_rolling_validation(backtest)
        sensitivity = build_cost_sensitivity(targets, sector_returns)

        save_outputs(backtest, benchmarks, summary, rolling, sensitivity)
        save_figures(backtest, benchmarks, rolling, sensitivity)
        save_config(spy_source)
        save_report(summary, rolling, sensitivity, spy_source)

        print(f"[PASS] Loaded {targets['Signal_Date'].nunique()} monthly Week 6 target portfolios.")
        print(f"[PASS] Used real local SPY benchmark data from {spy_source}.")
        print("[PASS] Applied conservative monthly execution with transaction costs and slippage.")
        print("[PASS] Accounted for between-rebalance weight drift before computing actual turnover.")
        print("[PASS] Evaluated CAGR, Sharpe, Sortino, drawdown, win rate, turnover, and trading costs.")
        print("[PASS] Compared the strategy with execution-aligned SPY and monthly equal-weight sector benchmarks.")
        print("[PASS] Completed rolling-window validation and trading-cost sensitivity analysis.")
        print("[PASS] Saved Week 7 tables, figures, config, and report.")
        print("[DONE] Week 7 backtesting and performance evaluation completed successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 7 backtest failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
