"""
Trade Statistics
================

One place that turns a trade log into the numbers a strategy decision is
actually made on, so every model reports the same things the same way.

Two deliberate choices:

- **R is the primary unit, not dollars.** Dollars depend on the notional chosen
  per instrument; R does not, so a basket of EURUSD and the Nasdaq CFD can be
  summed honestly only in R.
- **Win rate is reported but is never the headline.** A 70%-win model that risks
  3R to make 1R is worse than a 30%-win model at 3:1. Expectancy and
  cost-adjusted profit factor are the metrics that survive that trade-off;
  win rate is included because it is asked for, not because it decides anything.

Every function here is deterministic and computed only from the rows given.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from infrastructure.execution import (TP, SL, EOD_PROFIT, EOD_LOSS,
                                      BREAKEVEN, OTHER, EXIT_REASONS)

# Trading periods per year, used to annualise Sharpe/Sortino from per-trade
# returns. This model takes at most one trade per instrument per session, so a
# year of sessions is the natural scaling unit.
TRADES_PER_YEAR = 252.0

BREAKEVEN_EPS = 1e-9

REQUIRED_COLUMNS = (
    'entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price',
    'stop', 'target', 'pnl_pts', 'pnl_r', 'exit_reason', 'holding_minutes',
    'mae_pts', 'mfe_pts',
)


def _streak(flags: pd.Series) -> int:
    """Longest run of True in order."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def _drawdown(equity: pd.Series) -> tuple:
    """Max drawdown and the longest time spent below a prior peak.

    Duration is measured in trades, not calendar days: a strategy that takes two
    trades a month and one that takes twenty are not comparable in trade counts,
    so the calendar version is reported separately by the caller when timestamps
    are available.
    """
    if len(equity) == 0:
        return 0.0, 0
    peak = equity.cummax()
    dd = peak - equity
    max_dd = float(dd.max())
    under = (dd > BREAKEVEN_EPS).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return max_dd, longest


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe of per-trade R returns (risk-free = 0)."""
    if len(returns) < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return 0.0
    return float(returns.mean() / sd * math.sqrt(TRADES_PER_YEAR))


def _sortino(returns: pd.Series) -> float:
    """Annualised Sortino — downside deviation only, target return 0."""
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float('inf') if float(returns.mean()) > 0 else 0.0
    dd = float(np.sqrt((downside ** 2).mean()))
    if dd == 0 or not math.isfinite(dd):
        return 0.0
    return float(returns.mean() / dd * math.sqrt(TRADES_PER_YEAR))


def compute_stats(trades: pd.DataFrame, cost_model=None) -> dict:
    """Full statistics for a trade log. Empty log returns a zeroed dict."""
    n = len(trades)
    empty = {k: 0.0 for k in (
        'total_trades', 'wins', 'losses', 'breakeven', 'win_pct',
        'target_hit_pct', 'avg_r', 'median_r', 'expectancy_r', 'total_r',
        'gross_profit_r', 'gross_loss_r', 'profit_factor', 'max_dd_r',
        'dd_duration_trades', 'sharpe', 'sortino', 'max_win_streak',
        'max_loss_streak', 'avg_holding_minutes', 'median_holding_minutes',
        'total_pnl_usd', 'spread_cost_usd', 'slippage_cost_usd',
        'commission_usd', 'exec_cost_usd', 'max_dd_usd')}
    empty['total_trades'] = 0
    if n == 0:
        empty['cost_model'] = cost_model.describe() if cost_model else 'n/a'
        empty['commission_is_zero'] = True
        return empty

    r = trades['pnl_r'].astype(float)
    wins = trades[r > BREAKEVEN_EPS]
    losses = trades[r < -BREAKEVEN_EPS]
    flat = trades[r.abs() <= BREAKEVEN_EPS]

    gross_profit = float(wins['pnl_r'].sum()) if len(wins) else 0.0
    gross_loss = float(-losses['pnl_r'].sum()) if len(losses) else 0.0
    eq_r = r.cumsum()
    max_dd_r, dd_len = _drawdown(eq_r)

    hold = trades['holding_minutes'].astype(float) if 'holding_minutes' in trades else pd.Series(dtype=float)

    out = {
        'total_trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'breakeven': len(flat),
        'win_pct': round(len(wins) / n * 100, 2),
        'target_hit_pct': round(float((trades['exit_reason'] == TP).sum()) / n * 100, 2),
        'avg_r': round(float(r.mean()), 4),
        'median_r': round(float(r.median()), 4),
        # Expectancy is the mean R per trade taken - the same number as avg_r,
        # named separately because it is the metric decisions are made on.
        'expectancy_r': round(float(r.mean()), 4),
        'total_r': round(float(r.sum()), 3),
        'gross_profit_r': round(gross_profit, 3),
        'gross_loss_r': round(gross_loss, 3),
        'profit_factor': (round(gross_profit / gross_loss, 3) if gross_loss > 0
                          else (float('inf') if gross_profit > 0 else 0.0)),
        'max_dd_r': round(max_dd_r, 3),
        'dd_duration_trades': dd_len,
        'sharpe': round(_sharpe(r), 3),
        'sortino': round(_sortino(r), 3),
        'max_win_streak': _streak(r > BREAKEVEN_EPS),
        'max_loss_streak': _streak(r < -BREAKEVEN_EPS),
        'avg_holding_minutes': round(float(hold.mean()), 1) if len(hold) else 0.0,
        'median_holding_minutes': round(float(hold.median()), 1) if len(hold) else 0.0,
    }

    if 'exit_time' in trades and 'entry_time' in trades:
        ts = pd.to_datetime(trades['exit_time'], utc=True, errors='coerce')
        under = (eq_r.cummax() - eq_r) > BREAKEVEN_EPS
        if bool(under.any()) and ts.notna().all():
            spans, start = [], None
            for t, u in zip(ts, under):
                if u and start is None:
                    start = t
                elif not u and start is not None:
                    spans.append((t - start).total_seconds() / 86400.0)
                    start = None
            if start is not None:
                spans.append((ts.iloc[-1] - start).total_seconds() / 86400.0)
            out['dd_duration_days'] = round(max(spans), 1) if spans else 0.0
        else:
            out['dd_duration_days'] = 0.0

    for col, key in (('pnl_usd', 'total_pnl_usd'),
                     ('spread_cost_usd', 'spread_cost_usd'),
                     ('slippage_cost_usd', 'slippage_cost_usd'),
                     ('commission_usd', 'commission_usd'),
                     ('exec_cost_usd', 'exec_cost_usd')):
        out[key] = round(float(trades[col].sum()), 2) if col in trades else 0.0
    if 'pnl_usd' in trades:
        eq_usd = trades['pnl_usd'].astype(float).cumsum()
        out['max_dd_usd'] = round(_drawdown(eq_usd)[0], 2)
    else:
        out['max_dd_usd'] = 0.0

    if 'exec_cost_pts' in trades:
        out['exec_cost_r'] = round(
            float((trades['exec_cost_pts'] / trades['risk_pts']).sum()), 3)
        out['gross_total_r'] = round(out['total_r'] + out['exec_cost_r'], 3)

    out['cost_model'] = cost_model.describe() if cost_model else 'n/a'
    out['commission_is_zero'] = (cost_model.commission_round_turn == 0.0
                                 if cost_model else True)
    return out


def breakdown(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-group statistics. `by` is a column name or 'month' / 'year'."""
    if len(trades) == 0:
        return pd.DataFrame()
    df = trades.copy()
    if by in ('month', 'year'):
        ts = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
        df[by] = (ts.dt.to_period('M').astype(str) if by == 'month'
                  else ts.dt.year.astype('Int64').astype(str))
    if by not in df.columns:
        return pd.DataFrame()

    rows = []
    for key, grp in df.groupby(by, dropna=False):
        r = grp['pnl_r'].astype(float)
        wins = int((r > BREAKEVEN_EPS).sum())
        gp = float(r[r > 0].sum())
        gl = float(-r[r < 0].sum())
        rows.append({
            by: key,
            'n': len(grp),
            'win%': round(wins / len(grp) * 100, 1),
            'TP%': round(float((grp['exit_reason'] == TP).sum()) / len(grp) * 100, 1),
            'total_R': round(float(r.sum()), 2),
            'exp_R': round(float(r.mean()), 3),
            'PF': (round(gp / gl, 2) if gl > 0 else (float('inf') if gp > 0 else 0.0)),
        })
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def exit_reason_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Counts and R contribution per exit reason, including reasons never seen.

    Reasons with zero trades are kept so that "no EOD losses occurred" is visible
    as a fact rather than as a missing row.
    """
    rows = []
    n = max(len(trades), 1)
    for reason in EXIT_REASONS:
        grp = trades[trades['exit_reason'] == reason] if len(trades) else trades
        r = grp['pnl_r'].astype(float) if len(grp) else pd.Series(dtype=float)
        rows.append({
            'exit_reason': reason,
            'n': len(grp),
            'share%': round(len(grp) / n * 100, 1),
            'total_R': round(float(r.sum()), 2) if len(grp) else 0.0,
            'exp_R': round(float(r.mean()), 3) if len(grp) else 0.0,
        })
    return pd.DataFrame(rows)


def print_stats(stats: dict, label: str = 'Results') -> None:
    """Human-readable summary. Prints nothing it did not compute."""
    line = '=' * 78
    print(f"\n{line}\n  {label}\n{line}")
    print(f"  Cost model      : {stats.get('cost_model', 'n/a')}")
    if stats.get('commission_is_zero'):
        print("  Commission      : 0.00 (NOT modelled - baseline preserved)")
    if stats['total_trades'] == 0:
        print("  No trades.")
        return
    print(f"  Trades          : {stats['total_trades']}  "
          f"({stats['wins']}W / {stats['losses']}L / {stats['breakeven']}BE)")
    print(f"  Profitable      : {stats['win_pct']}%   "
          f"target-hit {stats['target_hit_pct']}%")
    print(f"  Expectancy      : {stats['expectancy_r']:+.3f} R/trade   "
          f"(median {stats['median_r']:+.3f} R)")
    print(f"  Total R         : {stats['total_r']:+.2f}   "
          f"gross +{stats['gross_profit_r']:.2f} / -{stats['gross_loss_r']:.2f}")
    print(f"  Profit factor   : {stats['profit_factor']}")
    if 'gross_total_r' in stats:
        print(f"  Pre-cost R      : {stats['gross_total_r']:+.2f}   "
              f"(execution cost {stats['exec_cost_r']:.2f} R)")
    print(f"  Max drawdown    : {stats['max_dd_r']:.2f} R over "
          f"{stats['dd_duration_trades']} trades"
          + (f" / {stats['dd_duration_days']:.0f} days"
             if 'dd_duration_days' in stats else ''))
    print(f"  Sharpe / Sortino: {stats['sharpe']} / {stats['sortino']}")
    print(f"  Streaks         : {stats['max_win_streak']}W max, "
          f"{stats['max_loss_streak']}L max")
    print(f"  Holding time    : mean {stats['avg_holding_minutes']:.0f} min, "
          f"median {stats['median_holding_minutes']:.0f} min")
    if stats.get('total_pnl_usd'):
        print(f"  P&L (USD)       : ${stats['total_pnl_usd']:,.2f}   "
              f"max DD ${stats['max_dd_usd']:,.2f}")
        print(f"  Execution cost  : spread ${stats['spread_cost_usd']:,.2f} + "
              f"slippage ${stats['slippage_cost_usd']:,.2f} + "
              f"commission ${stats['commission_usd']:,.2f} = "
              f"${stats['exec_cost_usd']:,.2f}")


def validate_trade_log(trades: pd.DataFrame) -> None:
    """Raise if the log is missing a column the statistics contract requires."""
    if len(trades) == 0:
        return
    missing = [c for c in REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"Trade log is missing required columns: {missing}")
    bad = set(trades['exit_reason'].unique()) - set(EXIT_REASONS)
    if bad:
        raise ValueError(f"Trade log has exit reasons outside the enum: {sorted(bad)}")
