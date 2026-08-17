"""Week 6 - Risk Management & Optimization.

This module extends the Week 5 combined sector portfolio with explicit risk
constraints and optimization. It uses only information available on or before
each signal date.

Implemented Week 6 components
-----------------------------
1. Portfolio constraints: sector caps, one-way turnover limits, and a drawdown
   defensive overlay.
2. Optimization: constrained mean-variance optimization (MVO) and inverse-
   volatility inverse volatility with volatility targeting.
3. Rule-based defensive positioning: reduce risky exposure when the live
   strategy drawdown breaches a configured threshold.
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

WEEK5_PATH = PROJECT_ROOT / "results" / "tables" / "week5_default_portfolio.csv"
DAILY_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sector_features.csv"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "data" / "strategy"
CONFIG_PATH = CONFIG_DIR / "week6_risk_config.json"

SECTORS = ("XLF", "XLK", "XLE")
CASH = "CASH"
ASSETS = (*SECTORS, CASH)

LOOKBACK_DAYS = 60
MIN_LOOKBACK_DAYS = 40
ANNUALIZATION = 252
MAX_SECTOR_WEIGHT = 0.60
MAX_ONE_WAY_TURNOVER = 0.50
TARGET_VOLATILITY = 0.12
DRAWDOWN_TRIGGER = -0.10
DEFENSIVE_GROSS_EXPOSURE = 0.50
MVO_RISK_AVERSION = 2.0
MVO_GRID_STEP = 0.01
COVARIANCE_SHRINKAGE = 0.20
MVO_BLEND_WEIGHT = 0.50
INVERSE_VOL_BLEND_WEIGHT = 0.50
TOL = 1e-10

MVO_STRATEGY = "MVO_Constrained"
INV_VOL_STRATEGY = "Inverse_Volatility_Targeted"
OPT_STRATEGY = "Week6_Optimized"
BASELINE_STRATEGY = "Week5_Baseline"

WEEK5_REQUIRED_COLUMNS = {
    "Signal_Date",
    "Strategy",
    "Asset",
    "Target_Weight",
    "Selected",
    "Outperformance_Probability",
    "Sector_Rank",
    "Rank_Score_0_1",
    "Macro_Regime",
    "model",
}


@dataclass(frozen=True)
class RiskEstimate:
    mean_annual: pd.Series
    covariance_annual: pd.DataFrame
    volatility_annual: pd.Series
    observations: int
    lookback_start: pd.Timestamp
    lookback_end: pd.Timestamp


def _require_columns(data: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_week5_portfolio(path: Path = WEEK5_PATH) -> pd.DataFrame:
    """Load and validate the Week 5 default portfolio handoff."""
    if not path.exists():
        raise FileNotFoundError(
            "results/tables/week5_default_portfolio.csv is missing. Complete Week 5 first."
        )

    data = pd.read_csv(path, parse_dates=["Signal_Date"])
    _require_columns(data, WEEK5_REQUIRED_COLUMNS, "Week 5 default portfolio")
    if data.empty:
        raise ValueError("week5_default_portfolio.csv is empty.")
    if data["Signal_Date"].isna().any():
        raise ValueError("Week 5 portfolio contains an invalid Signal_Date.")

    data["Asset"] = data["Asset"].astype(str).str.strip()
    if not set(data["Asset"].unique()).issubset(set(ASSETS)):
        raise ValueError("Week 5 default portfolio contains an unexpected asset.")

    weights = pd.to_numeric(data["Target_Weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Week 5 Target_Weight must contain finite numeric values.")
    if (weights < -TOL).any():
        raise ValueError("Week 5 Target_Weight cannot be negative.")
    data["Target_Weight"] = weights.astype(float)

    sector_rows = data[data["Asset"].isin(SECTORS)].copy()
    probabilities = pd.to_numeric(sector_rows["Outperformance_Probability"], errors="coerce")
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError("Week 5 sector probabilities must be numeric and inside [0, 1].")

    for date, group in data.groupby("Signal_Date", sort=True):
        if set(group["Asset"]) != set(ASSETS) or len(group) != len(ASSETS):
            raise ValueError(f"Signal date {date.date()} must contain XLF, XLK, XLE, and CASH.")
        total = float(group["Target_Weight"].sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Week 5 weights on {date.date()} do not sum to 1.0.")
        if group["Strategy"].nunique() != 1 or str(group["Strategy"].iloc[0]) != "Combined_Default":
            raise ValueError("Week 6 expects Week 5 Combined_Default portfolio rows.")
        if group["Macro_Regime"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent Macro_Regime values on {date.date()}.")
        if group["model"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent model labels on {date.date()}.")

    if data["Signal_Date"].nunique() < 2:
        raise ValueError("Week 6 requires at least two Week 5 signal dates.")

    return data.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)


def load_daily_returns(path: Path = DAILY_FEATURES_PATH) -> pd.DataFrame:
    """Load daily sector returns created in Week 2."""
    if not path.exists():
        raise FileNotFoundError(
            "data/processed/sector_features.csv is missing. Week 6 needs Week 2 daily returns."
        )

    required = {"Date", *(f"{sector}_Daily_Return" for sector in SECTORS)}
    data = pd.read_csv(path, parse_dates=["Date"])
    _require_columns(data, required, "sector_features.csv")
    if data.empty:
        raise ValueError("sector_features.csv is empty.")
    if data["Date"].isna().any():
        raise ValueError("sector_features.csv contains an invalid Date.")

    data = data.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    returns = pd.DataFrame({"Date": data["Date"]})
    for sector in SECTORS:
        series = pd.to_numeric(data[f"{sector}_Daily_Return"], errors="coerce")
        returns[sector] = series

    returns = returns.dropna(subset=list(SECTORS)).copy()
    finite = np.isfinite(returns[list(SECTORS)].to_numpy(dtype=float)).all(axis=1)
    returns = returns.loc[finite].reset_index(drop=True)
    if (returns[list(SECTORS)] <= -1.0).any().any():
        raise ValueError("Daily returns must be greater than -100%.")
    if len(returns) < MIN_LOOKBACK_DAYS:
        raise ValueError(
            f"Week 6 needs at least {MIN_LOOKBACK_DAYS} complete daily return observations."
        )
    return returns


def estimate_risk(daily_returns: pd.DataFrame, signal_date: pd.Timestamp) -> RiskEstimate:
    """Estimate trailing mean returns and a shrunk covariance matrix without look-ahead."""
    history = daily_returns[daily_returns["Date"] <= signal_date].tail(LOOKBACK_DAYS).copy()
    if len(history) < MIN_LOOKBACK_DAYS:
        raise ValueError(
            f"Only {len(history)} daily observations are available through {signal_date.date()}; "
            f"at least {MIN_LOOKBACK_DAYS} are required."
        )

    matrix = history[list(SECTORS)].astype(float)
    mean_annual = matrix.mean() * ANNUALIZATION
    sample_cov = matrix.cov() * ANNUALIZATION
    diagonal = pd.DataFrame(
        np.diag(np.diag(sample_cov.to_numpy(dtype=float))),
        index=SECTORS,
        columns=SECTORS,
    )
    covariance = (1.0 - COVARIANCE_SHRINKAGE) * sample_cov + COVARIANCE_SHRINKAGE * diagonal
    covariance = covariance.reindex(index=SECTORS, columns=SECTORS).astype(float)
    vol = pd.Series(
        np.sqrt(np.clip(np.diag(covariance.to_numpy(dtype=float)), 0.0, None)),
        index=SECTORS,
        dtype=float,
    )
    return RiskEstimate(
        mean_annual=mean_annual.reindex(SECTORS).astype(float),
        covariance_annual=covariance,
        volatility_annual=vol,
        observations=len(history),
        lookback_start=pd.Timestamp(history["Date"].min()),
        lookback_end=pd.Timestamp(history["Date"].max()),
    )


def _empty_weights() -> dict[str, float]:
    return {**{sector: 0.0 for sector in SECTORS}, CASH: 1.0}


def _normalize_dict(weights: dict[str, float]) -> dict[str, float]:
    result = {asset: max(0.0, float(weights.get(asset, 0.0))) for asset in ASSETS}
    total = sum(result.values())
    if total <= TOL:
        return _empty_weights()
    return {asset: value / total for asset, value in result.items()}


def _sector_exposure(weights: dict[str, float]) -> float:
    return float(sum(float(weights.get(sector, 0.0)) for sector in SECTORS))


def _forecast_volatility(weights: dict[str, float], covariance: pd.DataFrame) -> float:
    vector = np.array([float(weights.get(sector, 0.0)) for sector in SECTORS], dtype=float)
    variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
    return math.sqrt(max(0.0, variance))


def _eligible_sectors(month: pd.DataFrame) -> list[str]:
    sector = month[month["Asset"].isin(SECTORS)].copy()
    eligible = sector.loc[sector["Target_Weight"] > TOL, "Asset"].astype(str).tolist()
    return sorted(eligible)


def _week5_risky_budget(month: pd.DataFrame) -> float:
    """Return the gross sector exposure already authorized by Week 5."""
    sector = month[month["Asset"].isin(SECTORS)]
    gross = float(pd.to_numeric(sector["Target_Weight"], errors="raise").sum())
    return min(1.0, max(0.0, gross))


def _candidate_grid(eligible: list[str], risky_budget: float) -> list[dict[str, float]]:
    """Enumerate MVO allocations while preserving the Week 5 risky budget.

    Week 6 optimizes sizing inside the Week 5 signal layer. It does not create
    extra gross exposure before the volatility and drawdown overlays are applied.
    """
    risky_budget = min(float(risky_budget), 1.0, len(eligible) * MAX_SECTOR_WEIGHT)
    if not eligible or risky_budget <= TOL:
        return [_empty_weights()]

    step = MVO_GRID_STEP
    budget_units = int(round(risky_budget / step))
    cap_units = int(round(MAX_SECTOR_WEIGHT / step))
    candidates: list[dict[str, float]] = []

    if len(eligible) == 1:
        sector = eligible[0]
        weight = min(risky_budget, MAX_SECTOR_WEIGHT)
        return [{**{s: (weight if s == sector else 0.0) for s in SECTORS}, CASH: 1.0 - weight}]

    if len(eligible) == 2:
        a, b = eligible
        for ua in range(cap_units + 1):
            ub = budget_units - ua
            if 0 <= ub <= cap_units:
                row = {sector: 0.0 for sector in SECTORS}
                row[a], row[b] = ua * step, ub * step
                row[CASH] = max(0.0, 1.0 - row[a] - row[b])
                candidates.append(row)
    else:
        a, b, c = eligible[:3]
        for ua in range(cap_units + 1):
            for ub in range(cap_units + 1):
                uc = budget_units - ua - ub
                if 0 <= uc <= cap_units:
                    row = {sector: 0.0 for sector in SECTORS}
                    row[a], row[b], row[c] = ua * step, ub * step, uc * step
                    row[CASH] = max(0.0, 1.0 - row[a] - row[b] - row[c])
                    candidates.append(row)

    if not candidates:
        raise RuntimeError("No feasible MVO candidates were generated.")
    return candidates

def mean_variance_weights(month: pd.DataFrame, risk: RiskEstimate) -> tuple[dict[str, float], float]:
    """Grid-search a constrained MVO objective over the small three-sector universe."""
    eligible = _eligible_sectors(month)
    risky_budget = _week5_risky_budget(month)
    if not eligible or risky_budget <= TOL:
        return _empty_weights(), 0.0

    best_weights: dict[str, float] | None = None
    best_objective = -np.inf
    mu = risk.mean_annual.to_numpy(dtype=float)
    cov = risk.covariance_annual.to_numpy(dtype=float)

    for candidate in _candidate_grid(eligible, risky_budget):
        vector = np.array([candidate[sector] for sector in SECTORS], dtype=float)
        expected_return = float(vector @ mu)
        variance = float(vector @ cov @ vector)
        objective = expected_return - MVO_RISK_AVERSION * variance
        if objective > best_objective + 1e-14:
            best_objective = objective
            best_weights = candidate

    assert best_weights is not None
    return best_weights, float(best_objective)


def _cap_and_fill(
    scores: dict[str, float],
    eligible: list[str],
    risky_budget: float,
) -> dict[str, float]:
    """Allocate a risky budget by positive scores subject to the sector cap."""
    if not eligible or risky_budget <= TOL:
        return _empty_weights()

    risky_budget = min(float(risky_budget), 1.0, len(eligible) * MAX_SECTOR_WEIGHT)
    clean = {sector: max(float(scores.get(sector, 0.0)), 0.0) for sector in eligible}
    if sum(clean.values()) <= TOL:
        clean = {sector: 1.0 for sector in eligible}

    result = {sector: 0.0 for sector in SECTORS}
    remaining = set(eligible)
    remaining_budget = risky_budget

    while remaining and remaining_budget > TOL:
        denominator = sum(clean[s] for s in remaining)
        if denominator <= TOL:
            proposed = {s: remaining_budget / len(remaining) for s in remaining}
        else:
            proposed = {s: clean[s] / denominator * remaining_budget for s in remaining}
        offenders = [s for s, w in proposed.items() if w > MAX_SECTOR_WEIGHT + TOL]
        if not offenders:
            for s, w in proposed.items():
                result[s] = w
            remaining_budget = 0.0
            break
        for s in offenders:
            result[s] = MAX_SECTOR_WEIGHT
            remaining.remove(s)
            remaining_budget -= MAX_SECTOR_WEIGHT

    invested = float(sum(result.values()))
    cash = max(0.0, 1.0 - invested)
    if cash < TOL:
        cash = 0.0
    return {**result, CASH: cash}


def inverse_volatility_target_weights(month: pd.DataFrame, risk: RiskEstimate) -> tuple[dict[str, float], float, float]:
    """Build inverse-volatility weights and apply a volatility target."""
    eligible = _eligible_sectors(month)
    if not eligible:
        return _empty_weights(), 0.0, 1.0

    risky_budget = _week5_risky_budget(month)
    inverse_vol = {
        sector: (1.0 / max(float(risk.volatility_annual[sector]), 1e-8))
        for sector in eligible
    }
    full = _cap_and_fill(inverse_vol, eligible, risky_budget)
    pre_vol = _forecast_volatility(full, risk.covariance_annual)
    scale = 1.0 if pre_vol <= TOL else min(1.0, TARGET_VOLATILITY / pre_vol)

    scaled = {sector: full[sector] * scale for sector in SECTORS}
    scaled[CASH] = max(0.0, 1.0 - sum(scaled.values()))
    return scaled, pre_vol, scale


def blend_portfolios(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    blended = {
        sector: MVO_BLEND_WEIGHT * first[sector] + INVERSE_VOL_BLEND_WEIGHT * second[sector]
        for sector in SECTORS
    }
    for sector in SECTORS:
        blended[sector] = min(MAX_SECTOR_WEIGHT, max(0.0, blended[sector]))
    blended[CASH] = max(0.0, 1.0 - sum(blended[sector] for sector in SECTORS))
    return blended


def apply_volatility_target(weights: dict[str, float], covariance: pd.DataFrame) -> tuple[dict[str, float], float, float]:
    pre_vol = _forecast_volatility(weights, covariance)
    scale = 1.0 if pre_vol <= TOL else min(1.0, TARGET_VOLATILITY / pre_vol)
    result = {sector: weights[sector] * scale for sector in SECTORS}
    result[CASH] = max(0.0, 1.0 - sum(result.values()))
    return result, pre_vol, scale


def one_way_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return 0.5 * sum(abs(float(current[a]) - float(previous[a])) for a in ASSETS)


def apply_turnover_limit(
    previous: dict[str, float],
    proposed: dict[str, float],
    eligible: list[str],
    *,
    initial_deployment: bool = False,
) -> tuple[dict[str, float], float, float, bool, str]:
    """Apply a one-way turnover cap without trapping forced-sale proceeds in cash.

    The initial portfolio establishment is not treated as a rebalance. For later
    dates, sectors removed by Week 5 must be exited. If those exits alone exceed
    the normal cap, the minimum required turnover becomes an explicit exception.
    We then move as far as possible toward the proposed target while staying
    within that allowed turnover level.
    """
    raw_turnover = one_way_turnover(previous, proposed)
    if initial_deployment:
        return proposed.copy(), raw_turnover, 1.0, False, "INITIAL_DEPLOYMENT"

    eligible_set = set(eligible)
    mandatory = previous.copy()
    forced_exit_weight = 0.0
    for sector in SECTORS:
        if sector not in eligible_set and mandatory[sector] > TOL:
            forced_exit_weight += mandatory[sector]
            mandatory[sector] = 0.0
    mandatory[CASH] += forced_exit_weight
    mandatory = _normalize_dict(mandatory)

    mandatory_turnover = one_way_turnover(previous, mandatory)
    allowed_turnover = max(MAX_ONE_WAY_TURNOVER, mandatory_turnover)
    signal_exit_exception = mandatory_turnover > MAX_ONE_WAY_TURNOVER + TOL

    if raw_turnover <= allowed_turnover + TOL:
        reason = "MANDATORY_SIGNAL_EXIT" if signal_exit_exception else "NONE"
        return proposed.copy(), raw_turnover, 1.0, signal_exit_exception, reason

    # Search the straight line from the mandatory-exit portfolio toward the
    # optimized proposal. Measuring turnover against the original portfolio
    # correctly recognizes that sale proceeds can be reallocated without
    # unnecessarily increasing one-way turnover.
    best = mandatory.copy()
    best_alpha = 0.0
    for alpha in np.linspace(0.0, 1.0, 1001):
        candidate = {
            asset: float(mandatory[asset]) + float(alpha) * (float(proposed[asset]) - float(mandatory[asset]))
            for asset in ASSETS
        }
        candidate = _normalize_dict(candidate)
        if one_way_turnover(previous, candidate) <= allowed_turnover + 1e-12:
            best = candidate
            best_alpha = float(alpha)

    reason = "MANDATORY_SIGNAL_EXIT" if signal_exit_exception else "NONE"
    return best, raw_turnover, best_alpha, signal_exit_exception, reason

def apply_drawdown_overlay(
    weights: dict[str, float],
    current_drawdown: float,
) -> tuple[dict[str, float], bool]:
    """Cut total sector exposure to the defensive cap after a drawdown breach."""
    if current_drawdown > DRAWDOWN_TRIGGER + TOL:
        return weights.copy(), False

    gross = _sector_exposure(weights)
    if gross <= DEFENSIVE_GROSS_EXPOSURE + TOL:
        return weights.copy(), True

    scale = DEFENSIVE_GROSS_EXPOSURE / gross
    defensive = {sector: weights[sector] * scale for sector in SECTORS}
    defensive[CASH] = max(0.0, 1.0 - sum(defensive.values()))
    return defensive, True


def _week5_month_lookup(week5: pd.DataFrame, signal_date: pd.Timestamp) -> pd.DataFrame:
    month = week5[week5["Signal_Date"] == signal_date].copy()
    if month.empty:
        raise RuntimeError(f"Missing Week 5 target for {signal_date.date()}.")
    return month


def _weights_to_rows(
    signal_date: pd.Timestamp,
    strategy: str,
    weights: dict[str, float],
    month: pd.DataFrame,
) -> list[dict[str, object]]:
    lookup = month.set_index("Asset")
    rows: list[dict[str, object]] = []
    for asset in ASSETS:
        if asset == CASH:
            probability = np.nan
            rank = np.nan
        else:
            probability = float(lookup.loc[asset, "Outperformance_Probability"])
            rank_value = lookup.loc[asset, "Sector_Rank"]
            rank = int(rank_value) if pd.notna(rank_value) else np.nan
        rows.append(
            {
                "Signal_Date": signal_date,
                "Strategy": strategy,
                "Asset": asset,
                "Target_Weight": float(weights[asset]),
                "Selected": bool(asset != CASH and weights[asset] > TOL),
                "Outperformance_Probability": probability,
                "Sector_Rank": rank,
                "Macro_Regime": str(month["Macro_Regime"].iloc[0]),
                "model": str(month["model"].iloc[0]),
            }
        )
    return rows


def _apply_interval_returns(
    daily_returns: pd.DataFrame,
    start_exclusive: pd.Timestamp | None,
    end_inclusive: pd.Timestamp,
    weights: dict[str, float],
    equity: float,
    peak: float,
) -> tuple[float, float]:
    """Update live equity/drawdown state between two signal dates."""
    if start_exclusive is None:
        return equity, peak

    mask = (daily_returns["Date"] > start_exclusive) & (daily_returns["Date"] <= end_inclusive)
    interval = daily_returns.loc[mask]
    for _, row in interval.iterrows():
        portfolio_return = sum(weights[sector] * float(row[sector]) for sector in SECTORS)
        equity *= 1.0 + portfolio_return
        peak = max(peak, equity)
    return equity, peak

def build_week6_weights(
    week5: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build MVO, inverse-volatility and final Week 6 optimized targets sequentially."""
    signal_dates = sorted(pd.Timestamp(value) for value in week5["Signal_Date"].unique())

    mvo_rows: list[dict[str, object]] = []
    rp_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    previous_final = _empty_weights()
    last_signal: pd.Timestamp | None = None
    equity = 1.0
    peak = 1.0

    for signal_date in signal_dates:
        equity, peak = _apply_interval_returns(
            daily_returns, last_signal, signal_date, previous_final, equity, peak
        )
        current_drawdown = equity / peak - 1.0 if peak > 0 else 0.0

        month = _week5_month_lookup(week5, signal_date)
        risk = estimate_risk(daily_returns, signal_date)
        eligible = _eligible_sectors(month)

        mvo, mvo_objective = mean_variance_weights(month, risk)
        rp, rp_pre_vol, rp_scale = inverse_volatility_target_weights(month, risk)
        blended = blend_portfolios(mvo, rp)
        vol_targeted, blend_pre_vol, blend_vol_scale = apply_volatility_target(
            blended, risk.covariance_annual
        )
        turnover_limited, raw_turnover, turnover_scale, signal_exit_exception, turnover_reason = apply_turnover_limit(
            previous_final,
            vol_targeted,
            eligible,
            initial_deployment=(last_signal is None),
        )
        pre_risk_turnover = one_way_turnover(previous_final, turnover_limited)

        # Re-apply the volatility target after turnover control. This makes the
        # volatility target a hard risk-reduction rule rather than allowing a
        # turnover constraint to keep an overly risky legacy allocation.
        post_turnover_risk, post_turnover_vol, post_turnover_vol_scale = apply_volatility_target(
            turnover_limited, risk.covariance_annual
        )
        final, defensive_active = apply_drawdown_overlay(post_turnover_risk, current_drawdown)
        final_turnover = one_way_turnover(previous_final, final)

        volatility_risk_reduction = post_turnover_vol_scale < 1.0 - 1e-12
        risk_reduction_exception = bool(
            final_turnover > MAX_ONE_WAY_TURNOVER + 1e-9
            and final_turnover > pre_risk_turnover + 1e-9
            and (volatility_risk_reduction or defensive_active)
        )
        turnover_exception = bool(signal_exit_exception or risk_reduction_exception)
        if signal_exit_exception and risk_reduction_exception:
            exception_reason = "SIGNAL_EXIT_AND_RISK_REDUCTION"
        elif risk_reduction_exception:
            exception_reason = "RISK_REDUCTION"
        else:
            exception_reason = turnover_reason

        final_forecast_vol = _forecast_volatility(final, risk.covariance_annual)
        mvo_rows.extend(_weights_to_rows(signal_date, MVO_STRATEGY, mvo, month))
        rp_rows.extend(_weights_to_rows(signal_date, INV_VOL_STRATEGY, rp, month))
        final_rows.extend(_weights_to_rows(signal_date, OPT_STRATEGY, final, month))

        diagnostics.append(
            {
                "Signal_Date": signal_date,
                "Eligible_Sectors": ",".join(eligible),
                "Eligible_Count": len(eligible),
                "Lookback_Observations": risk.observations,
                "Lookback_Start": risk.lookback_start,
                "Lookback_End": risk.lookback_end,
                "MVO_Objective": mvo_objective,
                "MVO_Forecast_Vol": _forecast_volatility(mvo, risk.covariance_annual),
                "Inverse_Vol_PreTarget_Vol": rp_pre_vol,
                "Inverse_Vol_Scale": rp_scale,
                "Blended_PreTarget_Vol": blend_pre_vol,
                "Blended_Vol_Scale": blend_vol_scale,
                "PostTurnover_PreRisk_Vol": post_turnover_vol,
                "PostTurnover_Vol_Scale": post_turnover_vol_scale,
                "Pre_Rebalance_Drawdown": current_drawdown,
                "Defensive_Overlay_Active": defensive_active,
                "Raw_One_Way_Turnover": raw_turnover,
                "Turnover_Limit_Applied": last_signal is not None,
                "Turnover_Scale": turnover_scale,
                "Final_One_Way_Turnover": final_turnover,
                "Turnover_Limit_Exception": turnover_exception,
                "Turnover_Exception_Reason": exception_reason,
                "Final_Gross_Exposure": _sector_exposure(final),
                "Final_Forecast_Vol": final_forecast_vol,
                "Macro_Regime": str(month["Macro_Regime"].iloc[0]),
            }
        )

        previous_final = final
        last_signal = signal_date

    # Extend the live optimized performance to the final available daily observation.
    if last_signal is not None:
        last_date = pd.Timestamp(daily_returns["Date"].max())
        if last_date > last_signal:
            equity, peak = _apply_interval_returns(
                daily_returns, last_signal, last_date, previous_final, equity, peak
            )

    return (
        pd.DataFrame(mvo_rows),
        pd.DataFrame(rp_rows),
        pd.DataFrame(final_rows),
        pd.DataFrame(diagnostics),
    )


def build_rebalance_trades(final_portfolio: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    previous = _empty_weights()
    rows: list[dict[str, object]] = []
    diag = diagnostics.set_index("Signal_Date")

    for date, month in final_portfolio.groupby("Signal_Date", sort=True):
        current = month.set_index("Asset")["Target_Weight"].astype(float).to_dict()
        for asset in ASSETS:
            trade = float(current[asset]) - float(previous[asset])
            rows.append(
                {
                    "Signal_Date": pd.Timestamp(date),
                    "Asset": asset,
                    "Previous_Target_Weight": float(previous[asset]),
                    "New_Target_Weight": float(current[asset]),
                    "Trade_Weight": trade,
                    "Trade_Direction": "BUY" if trade > TOL else ("SELL" if trade < -TOL else "HOLD"),
                    "Turnover_Contribution": 0.5 * abs(trade),
                    "Defensive_Overlay_Active": bool(diag.loc[date, "Defensive_Overlay_Active"]),
                    "Turnover_Limit_Exception": bool(diag.loc[date, "Turnover_Limit_Exception"]),
                    "Turnover_Exception_Reason": str(diag.loc[date, "Turnover_Exception_Reason"]),
                    "Execution_Rule": "Next trading session after signal date",
                }
            )
        previous = current
    return pd.DataFrame(rows)


def build_constraint_audit(final_portfolio: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    diag = diagnostics.set_index("Signal_Date")
    for date, month in final_portfolio.groupby("Signal_Date", sort=True):
        sector = month[month["Asset"].isin(SECTORS)]
        d = diag.loc[date]
        total = float(month["Target_Weight"].sum())
        max_sector = float(sector["Target_Weight"].max())
        nonnegative = bool((month["Target_Weight"] >= -1e-12).all())
        sector_cap_ok = max_sector <= MAX_SECTOR_WEIGHT + 1e-9
        turnover_ok = (
            not bool(d["Turnover_Limit_Applied"])
            or float(d["Final_One_Way_Turnover"]) <= MAX_ONE_WAY_TURNOVER + 1e-9
            or bool(d["Turnover_Limit_Exception"])
        )
        volatility_ok = float(d["Final_Forecast_Vol"]) <= TARGET_VOLATILITY + 1e-9
        drawdown_ok = (
            float(d["Pre_Rebalance_Drawdown"]) > DRAWDOWN_TRIGGER + 1e-9
            or float(d["Final_Gross_Exposure"]) <= DEFENSIVE_GROSS_EXPOSURE + 1e-9
        )
        rows.append(
            {
                "Signal_Date": pd.Timestamp(date),
                "Weights_Sum_To_One": abs(total - 1.0) <= 1e-9,
                "Long_Only": nonnegative,
                "Sector_Cap_OK": sector_cap_ok,
                "Turnover_Limit_OK": turnover_ok,
                "Volatility_Target_OK": volatility_ok,
                "Drawdown_Control_OK": drawdown_ok,
                "Maximum_Sector_Weight": max_sector,
                "One_Way_Turnover": float(d["Final_One_Way_Turnover"]),
                "Turnover_Limit_Exception": bool(d["Turnover_Limit_Exception"]),
                "Turnover_Exception_Reason": str(d["Turnover_Exception_Reason"]),
                "Pre_Rebalance_Drawdown": float(d["Pre_Rebalance_Drawdown"]),
                "Final_Gross_Exposure": float(d["Final_Gross_Exposure"]),
                "All_Constraints_OK": bool(
                    abs(total - 1.0) <= 1e-9
                    and nonnegative
                    and sector_cap_ok
                    and turnover_ok
                    and volatility_ok
                    and drawdown_ok
                ),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_daily_performance(
    name: str,
    weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Backtest target weights using the next trading session after each signal date."""
    dates = sorted(pd.Timestamp(value) for value in weights["Signal_Date"].unique())
    equity = 1.0
    peak = 1.0
    rows: list[dict[str, object]] = []

    for index, signal_date in enumerate(dates):
        next_signal = dates[index + 1] if index + 1 < len(dates) else pd.Timestamp(daily_returns["Date"].max())
        month = weights[weights["Signal_Date"] == signal_date]
        current = month.set_index("Asset")["Target_Weight"].astype(float).to_dict()
        interval = daily_returns[
            (daily_returns["Date"] > signal_date) & (daily_returns["Date"] <= next_signal)
        ].copy()
        for _, row in interval.iterrows():
            ret = sum(float(current.get(sector, 0.0)) * float(row[sector]) for sector in SECTORS)
            equity *= 1.0 + ret
            peak = max(peak, equity)
            rows.append(
                {
                    "Date": pd.Timestamp(row["Date"]),
                    "Strategy": name,
                    "Daily_Return": float(ret),
                    "Equity": float(equity),
                    "Drawdown": float(equity / peak - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _week5_baseline_table(week5: pd.DataFrame) -> pd.DataFrame:
    columns = ["Signal_Date", "Asset", "Target_Weight"]
    out = week5[columns].copy()
    out["Strategy"] = BASELINE_STRATEGY
    return out[["Signal_Date", "Strategy", "Asset", "Target_Weight"]]


def build_daily_performance(
    week5: pd.DataFrame,
    mvo: pd.DataFrame,
    rp: pd.DataFrame,
    final: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    parts = [
        _portfolio_daily_performance(BASELINE_STRATEGY, _week5_baseline_table(week5), daily_returns),
        _portfolio_daily_performance(MVO_STRATEGY, mvo, daily_returns),
        _portfolio_daily_performance(INV_VOL_STRATEGY, rp, daily_returns),
        _portfolio_daily_performance(OPT_STRATEGY, final, daily_returns),
    ]
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        raise RuntimeError("Week 6 could not produce any daily performance observations.")
    return pd.concat(nonempty, ignore_index=True).sort_values(["Date", "Strategy"]).reset_index(drop=True)


def _turnover_for_table(weights: pd.DataFrame) -> float:
    """Average one-way turnover across true rebalances, excluding initial deployment."""
    previous = _empty_weights()
    values: list[float] = []
    for index, (_, month) in enumerate(weights.groupby("Signal_Date", sort=True)):
        current = month.set_index("Asset")["Target_Weight"].astype(float).to_dict()
        if index > 0:
            values.append(one_way_turnover(previous, current))
        previous = current
    return float(np.mean(values)) if values else 0.0

def build_risk_metrics(
    performance: pd.DataFrame,
    week5: pd.DataFrame,
    mvo: pd.DataFrame,
    rp: pd.DataFrame,
    final: pd.DataFrame,
) -> pd.DataFrame:
    weight_lookup = {
        BASELINE_STRATEGY: _week5_baseline_table(week5),
        MVO_STRATEGY: mvo,
        INV_VOL_STRATEGY: rp,
        OPT_STRATEGY: final,
    }
    rows = []
    for strategy, block in performance.groupby("Strategy", sort=False):
        returns = block["Daily_Return"].astype(float)
        n = len(returns)
        if n == 0:
            continue
        total_return = float(block["Equity"].iloc[-1] - 1.0)
        annualized_return = (1.0 + total_return) ** (ANNUALIZATION / n) - 1.0 if total_return > -1 else -1.0
        annualized_vol = float(returns.std(ddof=1) * math.sqrt(ANNUALIZATION)) if n > 1 else 0.0
        daily_std = float(returns.std(ddof=1)) if n > 1 else 0.0
        sharpe = (float(returns.mean()) / daily_std * math.sqrt(ANNUALIZATION)) if daily_std > TOL else np.nan
        max_drawdown = float(block["Drawdown"].min())
        weights = weight_lookup[strategy]
        gross = (
            weights[weights["Asset"].isin(SECTORS)]
            .groupby("Signal_Date")["Target_Weight"]
            .sum()
        )
        rows.append(
            {
                "Strategy": strategy,
                "Daily_Observations": n,
                "Total_Return": total_return,
                "Annualized_Return": annualized_return,
                "Annualized_Volatility": annualized_vol,
                "Sharpe_0RF": sharpe,
                "Max_Drawdown": max_drawdown,
                "Average_Gross_Exposure": float(gross.mean()) if not gross.empty else 0.0,
                "Average_One_Way_Turnover": _turnover_for_table(weights),
            }
        )
    return pd.DataFrame(rows)


def save_figures(performance: pd.DataFrame, final: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    equity = performance.pivot(index="Date", columns="Strategy", values="Equity").sort_index()
    ax = equity.plot(figsize=(10, 5))
    ax.set_title("Week 6 Risk-Managed Strategy Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week6_equity_curve.png", dpi=160)
    plt.close()

    drawdown = performance.pivot(index="Date", columns="Strategy", values="Drawdown").sort_index()
    ax = drawdown.plot(figsize=(10, 5))
    ax.set_title("Week 6 Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week6_drawdown_comparison.png", dpi=160)
    plt.close()

    pivot = final.pivot(index="Signal_Date", columns="Asset", values="Target_Weight")
    pivot = pivot.reindex(columns=list(ASSETS)).fillna(0.0)
    ax = pivot.plot(kind="area", stacked=True, figsize=(10, 5))
    ax.set_title("Week 6 Optimized Target Weights")
    ax.set_xlabel("Signal Date")
    ax.set_ylabel("Target Weight")
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week6_optimized_weights.png", dpi=160)
    plt.close()

    ax = diagnostics.plot(
        x="Signal_Date",
        y=["Raw_One_Way_Turnover", "Final_One_Way_Turnover"],
        marker="o",
        figsize=(10, 5),
    )
    ax.axhline(MAX_ONE_WAY_TURNOVER, linestyle="--", linewidth=1.2)
    ax.set_title("Week 6 Turnover Before and After Risk Controls")
    ax.set_xlabel("Signal Date")
    ax.set_ylabel("One-Way Turnover")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "week6_turnover_control.png", dpi=160)
    plt.close()


def save_config() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "week": 6,
        "objective": "Risk Management & Optimization",
        "inputs": {
            "week5_portfolio": "results/tables/week5_default_portfolio.csv",
            "daily_returns": "data/processed/sector_features.csv",
        },
        "sectors": list(SECTORS),
        "cash_asset": CASH,
        "risk_estimation": {
            "lookback_trading_days": LOOKBACK_DAYS,
            "minimum_observations": MIN_LOOKBACK_DAYS,
            "annualization_factor": ANNUALIZATION,
            "covariance_shrinkage_to_diagonal": COVARIANCE_SHRINKAGE,
            "uses_only_data_on_or_before_signal_date": True,
        },
        "constraints": {
            "long_only": True,
            "leverage": False,
            "max_sector_weight": MAX_SECTOR_WEIGHT,
            "max_one_way_turnover_per_rebalance": MAX_ONE_WAY_TURNOVER,
            "initial_deployment_counts_as_rebalance": False,
            "turnover_limit_exceptions": [
                "mandatory exit from a sector removed by the Week 5 signal",
                "drawdown-driven risk reduction",
            ],
            "drawdown_trigger": DRAWDOWN_TRIGGER,
            "defensive_max_gross_sector_exposure": DEFENSIVE_GROSS_EXPOSURE,
        },
        "optimization": {
            "mean_variance": {
                "risk_aversion": MVO_RISK_AVERSION,
                "grid_step": MVO_GRID_STEP,
            },
            "inverse_volatility": "inverse trailing annualized volatility with volatility targeting",
            "volatility_target": TARGET_VOLATILITY,
            "final_blend": {
                MVO_STRATEGY: MVO_BLEND_WEIGHT,
                INV_VOL_STRATEGY: INVERSE_VOL_BLEND_WEIGHT,
            },
        },
        "defensive_overlay": (
            "If pre-rebalance live drawdown is <= -10%, cap sector exposure at 50% and hold the rest in cash. "
            "Risk-reducing trades may exceed the normal turnover limit."
        ),
        "execution_rule": "Next trading session after signal date",
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _format_table(data: pd.DataFrame, float_digits: int = 4) -> str:
    display = data.copy()
    for column in display.select_dtypes(include=["float", "float64", "float32"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
        )
    return "```text\n" + display.to_string(index=False) + "\n```"


def save_report(
    metrics: pd.DataFrame,
    final: pd.DataFrame,
    diagnostics: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    latest_date = pd.Timestamp(final["Signal_Date"].max())
    latest = final[final["Signal_Date"] == latest_date][
        ["Asset", "Target_Weight", "Outperformance_Probability", "Sector_Rank"]
    ].copy()
    latest["Sector_Rank"] = latest["Sector_Rank"].map(
        lambda value: "" if pd.isna(value) else str(int(value))
    )
    defensive_count = int(diagnostics["Defensive_Overlay_Active"].sum())
    all_constraints = bool(audit["All_Constraints_OK"].all())

    lines = [
        "# Week 6 Risk Management & Optimization Report",
        "",
        "## Scope",
        "",
        "Week 6 extends the Week 5 combined portfolio with explicit sector caps, turnover control, drawdown protection, constrained mean-variance optimization, inverse-volatility allocation, volatility targeting, and a rule-based defensive overlay.",
        "",
        "## Core Risk Rules",
        "",
        f"- Maximum sector weight: {MAX_SECTOR_WEIGHT:.0%}",
        f"- Normal maximum one-way turnover per rebalance: {MAX_ONE_WAY_TURNOVER:.0%}",
        f"- Volatility target: {TARGET_VOLATILITY:.0%} annualized",
        f"- Drawdown trigger: {DRAWDOWN_TRIGGER:.0%}",
        f"- Defensive maximum sector exposure: {DEFENSIVE_GROSS_EXPOSURE:.0%}",
        "- Initial portfolio establishment is not treated as a rebalance for turnover control",
        "- Risk-reduction exception: defensive deleveraging may exceed the normal turnover limit",
        "- Long-only, no leverage, residual weight held in cash",
        "",
        "## Optimization",
        "",
        "MVO uses trailing 60-trading-day annualized mean returns and a shrunk covariance matrix. The second allocation uses inverse trailing volatility and a 12% volatility target. Both are restricted to sectors selected by the Week 5 combined signal and preserve the Week 5 risky budget before risk scaling.",
        "",
        "The final Week 6 target blends MVO and inverse-volatility allocations 50/50, applies the 12% volatility target, limits normal rebalancing turnover, and then applies the drawdown defensive overlay when required.",
        "",
        "## Risk Metrics",
        "",
        _format_table(metrics),
        "",
        "## Latest Week 6 Target",
        "",
        f"Signal date: {latest_date.date()}",
        "",
        _format_table(latest),
        "",
        "## Constraint Audit",
        "",
        f"All saved Week 6 targets passed the configured constraint audit: {all_constraints}.",
        f"Defensive overlay activations: {defensive_count}.",
        "",
        "## Look-Ahead Control",
        "",
        "Every optimization window ends on or before its signal date. Target weights are intended for the next trading session, and realized returns are used only after the corresponding signal date when computing portfolio performance and drawdown.",
    ]
    (DOCS_DIR / "week6_risk_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    mvo: pd.DataFrame,
    rp: pd.DataFrame,
    final: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trades: pd.DataFrame,
    audit: pd.DataFrame,
    performance: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mvo.to_csv(TABLE_DIR / "week6_mvo_weights.csv", index=False, date_format="%Y-%m-%d")
    rp.to_csv(TABLE_DIR / "week6_inverse_vol_weights.csv", index=False, date_format="%Y-%m-%d")
    final.to_csv(TABLE_DIR / "week6_optimized_portfolio.csv", index=False, date_format="%Y-%m-%d")
    diagnostics.to_csv(TABLE_DIR / "week6_optimization_diagnostics.csv", index=False, date_format="%Y-%m-%d")
    trades.to_csv(TABLE_DIR / "week6_rebalance_trades.csv", index=False, date_format="%Y-%m-%d")
    audit.to_csv(TABLE_DIR / "week6_constraint_audit.csv", index=False, date_format="%Y-%m-%d")
    performance.to_csv(TABLE_DIR / "week6_daily_performance.csv", index=False, date_format="%Y-%m-%d")
    metrics.to_csv(TABLE_DIR / "week6_risk_metrics.csv", index=False)


def main() -> int:
    try:
        week5 = load_week5_portfolio()
        daily_returns = load_daily_returns()

        first_signal = pd.Timestamp(week5["Signal_Date"].min())
        last_signal = pd.Timestamp(week5["Signal_Date"].max())
        last_return_date = pd.Timestamp(daily_returns["Date"].max())
        if last_signal > last_return_date:
            raise ValueError(
                f"The last Week 5 signal ({last_signal.date()}) is later than the available daily return history "
                f"({last_return_date.date()}). Refresh the Week 2 daily data before running Week 6."
            )
        available_before = int((daily_returns["Date"] <= first_signal).sum())
        if available_before < MIN_LOOKBACK_DAYS:
            raise ValueError(
                f"The first Week 5 signal has only {available_before} prior daily return observations; "
                f"Week 6 requires at least {MIN_LOOKBACK_DAYS}."
            )

        mvo, rp, final, diagnostics = build_week6_weights(week5, daily_returns)
        trades = build_rebalance_trades(final, diagnostics)
        audit = build_constraint_audit(final, diagnostics)
        performance = build_daily_performance(week5, mvo, rp, final, daily_returns)
        metrics = build_risk_metrics(performance, week5, mvo, rp, final)

        if not bool(audit["All_Constraints_OK"].all()):
            raise RuntimeError("At least one Week 6 optimized target failed the constraint audit.")

        save_outputs(mvo, rp, final, diagnostics, trades, audit, performance, metrics)
        save_figures(performance, final, diagnostics)
        save_config()
        save_report(metrics, final, diagnostics, audit)

        print(f"[PASS] Loaded {week5['Signal_Date'].nunique()} Week 5 monthly portfolio targets.")
        print(f"[PASS] Estimated trailing risk using up to {LOOKBACK_DAYS} daily observations per signal date.")
        print("[PASS] Built constrained mean-variance and inverse-volatility/volatility-target portfolios.")
        print("[PASS] Enforced long-only sector caps and normal monthly turnover limits.")
        print("[PASS] Applied drawdown-based defensive positioning with explicit cash allocation.")
        print("[PASS] Saved Week 6 targets, trades, risk metrics, diagnostics, figures, config, and report.")
        print("[DONE] Week 6 risk management and optimization completed successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 6 optimization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
