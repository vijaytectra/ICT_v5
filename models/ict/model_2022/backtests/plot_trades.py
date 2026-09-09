#!/usr/bin/env python3
"""
Backtest Utilities
==================

Shared helpers used by all model runner scripts in backtests/.

Reusable pieces
---------------
  Timezone      : to_ny, eod_utc, localize_to
  Simulation    : simulate_fill, simulate_exit, calc_pnl_pts
  Stats         : compute_stats, print_stats
  Static charts : plot_equity_curve
  Matplotlib    : draw_candles, ts_to_x, plot_trade

CLI (standalone)
----------------
  python backtests/plot_trades.py                   # first 12 trades
  python backtests/plot_trades.py --date 2025-03-14
  python backtests/plot_trades.py --n 20
  python backtests/plot_trades.py --all
"""

from __future__ import annotations

import os
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
import sys
import argparse

import numpy as np
from dataclasses import dataclass
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ─── Timezone helpers ────────────────────────────────────────────────────────

def to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a tz-naive UTC timestamp to America/New_York."""
    return ts.tz_localize('UTC').tz_convert('America/New_York')


def eod_utc(session_date: pd.Timestamp, eod_hour: int = 16,
            tz_aware: bool = False) -> pd.Timestamp:
    """Return EOD (default 16:00 NY) as UTC. tz_aware controls return tz."""
    date_str = str(session_date.date())
    eod_ny = pd.Timestamp(f'{date_str} {eod_hour:02d}:00:00', tz='America/New_York')
    eod = eod_ny.tz_convert('UTC')
    return eod if tz_aware else eod.replace(tzinfo=None)


def localize_to(ts: pd.Timestamp, ref_index: pd.DatetimeIndex) -> pd.Timestamp:
    """Make ts tz-aware/naive to match ref_index tz."""
    if ref_index.tz is not None and ts.tz is None:
        return ts.tz_localize(ref_index.tz)
    if ref_index.tz is None and ts.tz is not None:
        return ts.replace(tzinfo=None)
    return ts


# ─── Fill / Exit simulation ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CostModel:
    """Execution costs charged against a mid-price backtest.

    The cleaned CSVs record a per-bar `spread` (the ask-bid distance at that
    bar) alongside mid OHLC. Rather than invent a constant, half of that
    recorded spread is applied around the mid on both entry and exit, in the
    direction that costs the trade:

        long  - buy the ask on entry (mid + h), sell the bid on exit (mid - h)
        short - sell the bid on entry (mid - h), buy the ask on exit (mid + h)

    `slippage_points` is adverse-only and applies to stop fills, where the
    trigger is a market order. Limit fills (entry, target) get no slippage, but
    note the fill test is a bare touch of the level: this model assumes **zero
    queue latency and full fill on touch**, which real limit orders do not get.
    Entry results are therefore optimistic even with spread charged.

    All-zero defaults reproduce the pre-cost engine exactly.
    """
    use_spread: bool = False
    spread_multiplier: float = 1.0
    commission_points: float = 0.0     # round turn, in price points
    slippage_points: float = 0.0       # adverse, stop fills only

    @property
    def enabled(self) -> bool:
        return bool(self.use_spread or self.commission_points or self.slippage_points)

    def half_spread(self, row) -> float:
        """Half the bar's recorded spread, or 0.0 when not charging spread."""
        if not self.use_spread:
            return 0.0
        try:
            sp = float(row['spread'])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if sp != sp or sp <= 0:        # NaN or non-positive
            return 0.0
        return sp * 0.5 * self.spread_multiplier


NO_COSTS = CostModel()


def _entry_exec(price: float, direction: str, hs: float) -> float:
    """Execution price for opening: long lifts the ask, short hits the bid."""
    return price + hs if direction == 'long' else price - hs


def _exit_exec(price: float, direction: str, hs: float) -> float:
    """Execution price for closing: long hits the bid, short lifts the ask."""
    return price - hs if direction == 'long' else price + hs


def _stop_exec(row, stop: float, direction: str, hs: float, slip: float) -> float:
    """Stop fill, honouring gaps and adverse slippage.

    The trigger test is unchanged. What changes is the price: if the bar opened
    beyond the stop the fill is taken at that open, because the level was gone
    before the order could work. Slippage is then charged on top, always
    against the position.
    """
    o = float(row['open'])
    if direction == 'long':
        px = min(stop, o)              # gapped below -> filled at the open
        return _exit_exec(px - slip, direction, hs)
    px = max(stop, o)                  # gapped above
    return _exit_exec(px + slip, direction, hs)


def simulate_fill(
    df: pd.DataFrame,
    signal: dict,
    session_date: pd.Timestamp,
    eod_hour: int = 16,
    costs: CostModel = NO_COSTS,
    expiry: pd.Timestamp | None = None,
    cancel_on_target: bool = False,
) -> dict | None:
    """
    Scan bars after signal['entry_time'] for a limit fill.

    entry-sequence.md: "the limit isn't filled before the killzone closes / the
    target pool is hit first (no chasing)". `expiry` is that killzone close and
    `cancel_on_target` that no-chase rule; without them the order would rest
    until the session flat, and a setup that only fills hours after its killzone
    has closed is no longer the setup that was signalled.

    Short fills when high >= entry; long fills when low <= entry.
    Returns {'fill_time', 'fill_price', ...} or None.
    """
    entry_time  = localize_to(signal['entry_time'], df.index)
    entry_price = signal['entry']
    direction   = signal['direction']
    eod         = eod_utc(session_date, eod_hour, tz_aware=(df.index.tz is not None))

    if expiry is not None:
        eod = min(eod, localize_to(expiry, df.index))

    target = signal.get('target')
    window = df[(df.index > entry_time) & (df.index <= eod)]
    for ts, row in window.iterrows():
        if cancel_on_target and target is not None:
            reached = (float(row['low']) <= target if direction == 'short'
                       else float(row['high']) >= target)
            if reached:
                return None            # target ran without us - do not chase
        # Touch test on the mid is unchanged; only the executed price moves.
        touched = (row['high'] >= entry_price if direction == 'short'
                   else row['low'] <= entry_price)
        if touched:
            hs = costs.half_spread(row)
            return {'fill_time': ts,
                    'fill_price': _entry_exec(entry_price, direction, hs),
                    'mid_fill_price': entry_price,
                    'entry_spread_cost': hs}
    return None


def simulate_exit(
    df: pd.DataFrame,
    signal: dict,
    fill: dict,
    session_date: pd.Timestamp,
    eod_hour: int = 16,
    be_trigger_r: float = 0.0,
    partials: list | None = None,
    partial_be: bool = True,
    costs: CostModel = NO_COSTS,
) -> dict:
    """
    Scan bars from fill_time to EOD for the first target or stop hit.
    Falls back to the EOD close.

    be_trigger_r > 0 arms a breakeven stop: once the trade has run that many R
    in favour, the stop moves to the entry price. Within a bar the original stop
    and target are checked *before* the trigger is re-evaluated, so a bar that
    both reaches the trigger and takes out the stop is scored as a loss - the
    conservative reading, since bar data cannot say which came first.

    `partials` is a schedule of (fraction, R_multiple) legs closed on the way to
    the target, e.g. [(0.5, 1.0), (0.25, 2.0)] leaves a 25% runner. ICT teaches
    these as the primary trade-management tool: bank realised R early, then move
    the stop to breakeven on the remainder (`partial_be`). That is a different
    proposition from a bare breakeven stop, which risks the whole position with
    nothing banked. When a schedule is given the returned 'pnl_pts' is the
    fraction-weighted total across every leg, and the caller should use it
    instead of recomputing from a single exit price.

    Returns {'exit_time', 'exit_price', 'exit_reason', 'mae_pts', 'mfe_pts',
    'bars_held'}. MAE is how far price ran against the position before the exit
    and MFE how far it ran in favour - together they say whether a stopped trade
    was ever really working, which a bare win/loss column cannot.
    """
    direction = signal['direction']
    target    = signal['target']
    stop      = signal['stop']
    fill_time = localize_to(fill['fill_time'], df.index)
    eod       = eod_utc(session_date, eod_hour, tz_aware=(df.index.tz is not None))

    entry = float(fill['fill_price'])
    window = df[(df.index > fill_time) & (df.index <= eod)]

    risk = abs(float(signal['stop']) - float(signal['entry']))
    be_distance = be_trigger_r * risk if (be_trigger_r and risk > 0) else None
    be_armed = False

    sign = -1.0 if direction == 'short' else 1.0
    legs = sorted(partials or [], key=lambda x: x[1])
    pending = [(float(f), entry + sign * float(r) * risk) for f, r in legs] if risk > 0 else []
    realized = 0.0          # fraction-weighted points already banked
    remaining = 1.0

    slip = costs.slippage_points
    # Entry spread was charged on the full position inside fill_price; exit
    # spread accrues per leg as each fraction closes.
    spread_cost = float(fill.get('entry_spread_cost', 0.0))
    slip_cost = 0.0

    def _finish(ts, price, reason, exc, exit_hs=0.0, exit_slip=0.0):
        pnl = realized + remaining * sign * (price - entry)
        return {'exit_time': ts, 'exit_price': price, 'exit_reason': reason,
                'pnl_pts': pnl, 'partials_filled': len(legs) - len(pending),
                'spread_cost_pts': spread_cost + remaining * exit_hs,
                'slippage_cost_pts': slip_cost + remaining * exit_slip,
                **exc}

    mae = mfe = 0.0
    for n, (ts, row) in enumerate(window.iterrows(), 1):
        hi, lo = float(row['high']), float(row['low'])
        if direction == 'short':
            mae = max(mae, hi - entry)
            mfe = max(mfe, entry - lo)
        else:
            mae = max(mae, entry - lo)
            mfe = max(mfe, hi - entry)
        excursion = {'mae_pts': mae, 'mfe_pts': mfe, 'bars_held': n}
        stop_reason = 'breakeven' if be_armed else 'stop'

        # Stop and target are resolved before the partial legs, so a bar that
        # both reaches a leg and takes out the stop is scored against us.
        hs = costs.half_spread(row)
        hit_target = (lo <= target) if direction == 'short' else (hi >= target)
        hit_stop = (hi >= stop) if direction == 'short' else (lo <= stop)
        if hit_target:
            return _finish(ts, _exit_exec(target, direction, hs), 'target',
                           excursion, exit_hs=hs)
        if hit_stop:
            # Market exit: gap-aware price, adverse slippage, then the spread.
            return _finish(ts, _stop_exec(row, stop, direction, hs, slip),
                           stop_reason, excursion, exit_hs=hs, exit_slip=slip)

        while pending:
            frac, level = pending[0]
            hit = lo <= level if direction == 'short' else hi >= level
            if not hit:
                break
            pending.pop(0)
            take = min(frac, remaining)
            realized += take * sign * (_exit_exec(level, direction, hs) - entry)
            remaining -= take
            spread_cost += take * hs
            if partial_be and not be_armed:
                be_armed = True
                stop = entry
            if remaining <= 1e-9:
                return _finish(ts, _exit_exec(level, direction, hs), 'partial', excursion)

        if be_distance is not None and not be_armed and mfe >= be_distance:
            be_armed = True
            stop = entry

    if len(window) > 0:
        last = window.iloc[-1]
        hs = costs.half_spread(last)
        return _finish(window.index[-1], _exit_exec(float(last['close']), direction, hs),
                       'eod', {'mae_pts': mae, 'mfe_pts': mfe, 'bars_held': len(window)},
                       exit_hs=hs)
    return _finish(fill_time, entry, 'eod',
                   {'mae_pts': 0.0, 'mfe_pts': 0.0, 'bars_held': 0})


def calc_pnl_pts(direction: str, entry: float, exit_price: float) -> float:
    return (entry - exit_price) if direction == 'short' else (exit_price - entry)


# ─── Statistics ──────────────────────────────────────────────────────────────

def compute_stats(trades: pd.DataFrame, price_dp: int = 2) -> dict:
    """Aggregate trade stats.

    price_dp controls rounding of the point-denominated fields. Index futures
    quote in points so 2dp is plenty; FX prices move in ~0.0001 steps, where
    2dp would round every points column to zero (pass 6, or 4 for JPY pairs).
    """
    if len(trades) == 0:
        return {}

    pnl_pts = trades['pnl_pts']
    pnl_usd = trades['pnl_usd']
    wins    = trades[trades['exit_reason'] == 'target']
    losses  = trades[trades['exit_reason'] == 'stop']
    eods    = trades[trades['exit_reason'].str.startswith('eod')]

    # Profit factor over EVERY trade, by realised P&L sign. Keying it off
    # exit_reason instead silently drops eod exits, and drops the 'partial' and
    # 'breakeven' exits entirely once a partial-take schedule is running - which
    # produced the contradiction of a run showing PF 0.87 on +$1,402.
    gross_wins   = float(pnl_usd[pnl_usd > 0].sum())
    gross_losses = abs(float(pnl_usd[pnl_usd < 0].sum())) or 1e-9

    equity = pnl_usd.cumsum()
    peak   = equity.cummax()
    max_dd = float((equity - peak).min())

    daily_ret = trades.groupby('session_date')['pnl_usd'].sum()
    sharpe = (float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
              if len(daily_ret) > 1 else 0.0)

    avg_rr = (float(trades['reward_pts'].mean() /
                    trades['risk_pts'].replace(0, np.nan).mean())
              if len(trades) else 0.0)

    return {
        'total_trades':   len(trades),
        'total_pnl_usd':  round(float(pnl_usd.sum()), 2),
        'total_pnl_pts':  round(float(pnl_pts.sum()), price_dp),
        'win_rate':        round(len(wins) / len(trades) * 100, 1),
        # Share of trades that actually finished in profit, whatever the exit
        # route - the meaningful figure when partials or EOD exits are in play.
        'profitable_pct':  round(float((pnl_usd > 0).mean()) * 100, 1),
        'profitable':      int((pnl_usd > 0).sum()),
        'unprofitable':    int((pnl_usd < 0).sum()),
        'wins':            len(wins),
        'losses':          len(losses),
        'eods':            len(eods),
        'avg_win_pts':     round(float(wins['pnl_pts'].mean()), price_dp) if len(wins) else 0,
        'avg_loss_pts':    round(float(losses['pnl_pts'].mean()), price_dp) if len(losses) else 0,
        'avg_win_usd':     round(float(wins['pnl_usd'].mean()), 2) if len(wins) else 0,
        'avg_loss_usd':    round(float(losses['pnl_usd'].mean()), 2) if len(losses) else 0,
        'profit_factor':   round(gross_wins / gross_losses, 2),
        'avg_rr':          round(avg_rr, 2),
        'max_dd_usd':      round(max_dd, 2),
        'sharpe_ratio':    round(sharpe, 2),
        **_enriched(trades, pnl_usd, equity, daily_ret),
    }

def _streak(flags) -> int:
    """Longest run of True in an iterable of booleans."""
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def _enriched(trades: pd.DataFrame, pnl_usd, equity, daily_ret) -> dict:
    """Distribution and path statistics the headline figures hide.

    Sortino divides by downside deviation only, so it does not penalise the
    large upside days a 3R-target model depends on the way Sharpe does.
    Drawdown duration is counted in trades rather than calendar time, because
    the model takes at most one setup per session.
    """
    out = {}
    if 'R' in trades.columns and len(trades):
        r = trades['R']
        out['avg_r'] = round(float(r.mean()), 3)
        out['median_r'] = round(float(r.median()), 3)
        out['total_r'] = round(float(r.sum()), 2)
        out['expectancy_r'] = round(float(r.mean()), 3)
    out['gross_profit_usd'] = round(float(pnl_usd[pnl_usd > 0].sum()), 2)
    out['gross_loss_usd'] = round(float(pnl_usd[pnl_usd < 0].sum()), 2)
    out['expectancy_usd'] = round(float(pnl_usd.mean()), 2)

    downside = daily_ret[daily_ret < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    out['sortino_ratio'] = (round(float(daily_ret.mean() / dstd * np.sqrt(252)), 2)
                            if dstd > 0 else 0.0)

    peak = equity.cummax()
    out['max_dd_duration_trades'] = _streak((equity < peak).tolist())
    out['max_win_streak'] = _streak((pnl_usd > 0).tolist())
    out['max_loss_streak'] = _streak((pnl_usd < 0).tolist())

    if 'holding_minutes' in trades.columns:
        h = trades['holding_minutes']
        out['avg_holding_min'] = round(float(h.mean()), 1)
        out['median_holding_min'] = round(float(h.median()), 1)
    for col in ('spread_cost_pts', 'commission_pts', 'slippage_cost_pts', 'total_cost_pts'):
        if col in trades.columns:
            out['total_' + col] = round(float(trades[col].sum()), 4)
    if 'mae_pts' in trades.columns:
        out['avg_mae_pts'] = round(float(trades['mae_pts'].mean()), 4)
        out['avg_mfe_pts'] = round(float(trades['mfe_pts'].mean()), 4)
    return out



def print_stats(stats: dict, label: str = 'Results', price_dp: int = 2):
    w = 52
    print(f"\n{'='*w}")
    print(f"  {label}")
    print(f"{'='*w}")
    if not stats:
        print("  No trades.")
        return
    print(f"  Total trades:    {stats['total_trades']}")
    print(f"  Target hit:      {stats['win_rate']}%  "
          f"({stats['wins']}W / {stats['losses']}L / {stats['eods']} EOD)")
    if 'profitable_pct' in stats:
        # The figure that matters when trades also exit at EOD or on a partial:
        # a target-only rate reads as 7.5% on a book that is 40% profitable.
        print(f"  Profitable:      {stats['profitable_pct']}%  "
              f"({stats.get('profitable')} up / {stats.get('unprofitable')} down)")
    print(f"  Total P&L:       ${stats['total_pnl_usd']:>10,.2f}  ({stats['total_pnl_pts']:.{price_dp}f} pts)")
    print(f"  Avg win:         {stats['avg_win_pts']:>10.{price_dp}f} pts  (${stats['avg_win_usd']:,.2f})")
    print(f"  Avg loss:        {stats['avg_loss_pts']:>10.{price_dp}f} pts  (${stats['avg_loss_usd']:,.2f})")
    print(f"  Avg R:R (setup): {stats['avg_rr']:.2f}")
    print(f"  Profit factor:   {stats['profit_factor']:.2f}")
    print(f"  Max drawdown:    ${stats['max_dd_usd']:>10,.2f}")
    print(f"  Sharpe ratio:    {stats['sharpe_ratio']:.2f}")
    print()


# ─── Static equity curve ─────────────────────────────────────────────────────

def plot_equity_curve(trades: pd.DataFrame, path: str, title: str = 'Equity Curve'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    equity = trades['pnl_usd'].cumsum()
    xs = range(len(equity))

    ax1.plot(equity.values, color='royalblue', linewidth=1.5, zorder=3)
    ax1.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax1.fill_between(xs, equity.values, 0,
                     where=(equity.values >= 0), alpha=0.12, color='green', zorder=2)
    ax1.fill_between(xs, equity.values, 0,
                     where=(equity.values < 0), alpha=0.12, color='red', zorder=2)
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel('Cumulative P&L (USD)')
    ax1.grid(True, alpha=0.25)

    colors = ['green' if r == 'target' else ('red' if r == 'stop' else 'darkorange')
              for r in trades['exit_reason']]
    ax2.bar(xs, trades['pnl_usd'], color=colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Trade P&L (USD)')
    ax2.set_xlabel('Trade #')
    ax2.grid(True, alpha=0.25)

    legend_elements = [
        mpatches.Patch(facecolor='green',      alpha=0.8, label='Target hit'),
        mpatches.Patch(facecolor='red',        alpha=0.8, label='Stop hit'),
        mpatches.Patch(facecolor='darkorange', alpha=0.8, label='EOD exit'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Equity curve → {path}")


# ─── Matplotlib candlestick primitives ───────────────────────────────────────

def draw_candles(ax, df: pd.DataFrame):
    """Draw OHLC candlesticks on ax using integer x positions."""
    width = 0.6
    for i, (_, row) in enumerate(df.iterrows()):
        bull  = row['close'] >= row['open']
        color = '#26a69a' if bull else '#ef5350'
        body_lo = min(row['open'], row['close'])
        body_hi = max(row['open'], row['close'])
        ax.add_line(Line2D([i, i], [row['low'], row['high']],
                           color=color, linewidth=0.8, zorder=2))
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - width / 2, body_lo), width, max(body_hi - body_lo, 0.01),
            boxstyle='square,pad=0', facecolor=color,
            edgecolor=color, linewidth=0.5, zorder=3,
        ))

    ax.set_xlim(-1, len(df))
    ax.set_ylim(df['low'].min() - 10, df['high'].max() + 10)
    ax.autoscale_view()

    tick_step = max(1, len(df) // 10)
    tick_positions = list(range(0, len(df), tick_step))
    tick_labels = []
    for pos in tick_positions:
        ts = df.index[pos]
        ny = ts.tz_convert('America/New_York') if ts.tzinfo else ts
        tick_labels.append(ny.strftime('%H:%M'))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7)


def ts_to_x(df: pd.DataFrame, ts: pd.Timestamp, tolerance_mins: int = 4) -> float | None:
    """Convert a timestamp to an integer x position in df, or None if out of range."""
    idx = df.index
    if idx.tz is not None and ts.tz is None:
        ts = ts.tz_localize(idx.tz)
    elif idx.tz is None and ts.tz is not None:
        ts = ts.replace(tzinfo=None)
    diffs = abs(idx - ts)
    pos   = diffs.argmin()
    if diffs[pos] <= pd.Timedelta(minutes=tolerance_mins):
        return float(pos)
    return None


# ─── Per-trade matplotlib plot ────────────────────────────────────────────────

CHART_START_NY = '07:00'
CHART_END_NY   = '14:00'

C_LONG  = '#26a69a'
C_SHORT = '#ef5350'


def plot_trade(df_1m: pd.DataFrame, session_date: pd.Timestamp,
               signal: dict, fill: dict | None, exit_info: dict | None,
               pnl_usd: float | None, out_path: str,
               resample: str = '3min'):
    """
    Render a single trade to a PNG using matplotlib.

    Args:
        df_1m:        Full 1m dataset (used to slice the chart window).
        session_date: Calendar date of the trade.
        signal:       Model signal dict (entry, stop, target, fvg_*, sweep_*, bms_*).
        fill:         {'fill_time', 'fill_price'} or None.
        exit_info:    {'exit_time', 'exit_price', 'exit_reason'} or None.
        pnl_usd:      Dollar P&L to display in the corner.
        out_path:     Where to save the PNG.
        resample:     Bar size for the chart (default '3min').
    """
    date_str  = str(session_date.date())
    start_ny  = pd.Timestamp(f'{date_str} {CHART_START_NY}', tz='America/New_York')
    end_ny    = pd.Timestamp(f'{date_str} {CHART_END_NY}',   tz='America/New_York')
    start_utc = start_ny.tz_convert('UTC')
    end_utc   = end_ny.tz_convert('UTC')

    s = start_utc if df_1m.index.tz is not None else start_utc.replace(tzinfo=None)
    e = end_utc   if df_1m.index.tz is not None else end_utc.replace(tzinfo=None)
    raw = df_1m.loc[s:e]
    if len(raw) < 5:
        print(f"  Skipping {date_str} — not enough chart bars")
        return

    chart_df = (raw.resample(resample)
                .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
                .dropna(subset=['close']))
    if len(chart_df) < 5:
        print(f"  Skipping {date_str} — chart df too short after resample")
        return

    direction = signal['direction']
    dir_color = C_SHORT if direction == 'short' else C_LONG

    fig, ax = plt.subplots(figsize=(16, 8))
    draw_candles(ax, chart_df)
    tick = (chart_df['high'].max() - chart_df['low'].min()) * 0.01

    def hline(price, color, ls, lw, label, y_offset=0, x_pct=(0.0, 1.0), fontsize=8):
        x0 = x_pct[0] * len(chart_df)
        x1 = x_pct[1] * len(chart_df)
        ax.hlines(price, x0, x1, colors=color, linestyles=ls, linewidth=lw, zorder=4)
        ax.text(x1 + 0.3, price + y_offset, label,
                va='center', ha='left', fontsize=fontsize,
                color=color, fontweight='bold')

    def vline(ts, color, ls, lw, label, y_frac=0.98):
        x = ts_to_x(chart_df, ts)
        if x is None:
            return
        y_min, y_max = ax.get_ylim()
        ax.axvline(x, color=color, linestyle=ls, linewidth=lw, zorder=5, alpha=0.7)
        ax.text(x, y_min + (y_max - y_min) * y_frac, label,
                va='top', ha='center', fontsize=7, color=color,
                rotation=90, fontweight='bold')

    def highlight_candle(x_pos, bar, border_color, label, label_above=True):
        if x_pos is None:
            return
        x_pos   = int(round(x_pos))
        if x_pos < 0 or x_pos >= len(chart_df):
            return
        body_lo = min(bar['open'], bar['close'])
        body_hi = max(bar['open'], bar['close'])
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_pos - 0.35, body_lo), 0.7, max(body_hi - body_lo, 0.5),
            boxstyle='square,pad=0', facecolor='none', edgecolor=border_color,
            linewidth=2.5, zorder=7,
        ))
        y_anchor = body_hi if label_above else body_lo
        dy = tick * 2.5 if label_above else -tick * 2.5
        ax.annotate(label, xy=(x_pos, y_anchor), xytext=(x_pos, y_anchor + dy),
                    arrowprops=dict(arrowstyle='->', color=border_color, lw=1.2),
                    fontsize=7.5, color=border_color, fontweight='bold',
                    ha='center', va='bottom' if label_above else 'top', zorder=8)

    # ── Pool level ─────────────────────────────────────────────────────────
    pool_price = signal.get('sweep_level')
    if pool_price:
        ax.axhline(pool_price, color='#9c27b0', linestyle='--', linewidth=1.2,
                   zorder=4, alpha=0.85)
        ax.text(0.01, pool_price + tick, '① Pool level',
                va='bottom', fontsize=7.5, color='#9c27b0', fontweight='bold',
                transform=ax.get_yaxis_transform())

    # ── Stop hunt ──────────────────────────────────────────────────────────
    sweep_ts = signal.get('sweep_time')
    sweep_x  = ts_to_x(chart_df, sweep_ts) if sweep_ts else None
    if sweep_x is not None:
        sweep_bar = chart_df.iloc[int(sweep_x)]
        arrow_y   = sweep_bar['high'] if direction == 'short' else sweep_bar['low']
        dy        = tick * 3 if direction == 'short' else -tick * 3
        ax.annotate('② Stop\n   hunt', xy=(sweep_x, arrow_y),
                    xytext=(sweep_x, arrow_y + dy),
                    arrowprops=dict(arrowstyle='->', color='#ff9800', lw=1.5),
                    fontsize=7.5, color='#ff9800', fontweight='bold',
                    ha='center', va='bottom' if direction == 'short' else 'top')

    # ── MSS ────────────────────────────────────────────────────────────────
    bms_ts    = signal.get('bms_time')
    bms_level = signal.get('bms_swing_level')
    bms_x     = ts_to_x(chart_df, bms_ts) if bms_ts else None
    if bms_x is not None and bms_level:
        ax.scatter([bms_x], [bms_level], marker='D', s=60, color='#2196f3', zorder=6)
        ax.text(bms_x + 0.5, bms_level + tick * 1.5, '③ MSS',
                va='bottom', ha='left', fontsize=7.5, color='#2196f3', fontweight='bold')

    # ── FVG zone ───────────────────────────────────────────────────────────
    fvg_top = signal.get('fvg_top')
    fvg_bot = signal.get('fvg_bottom')
    if fvg_top and fvg_bot:
        fvg_x = ts_to_x(chart_df, signal['entry_time']) if signal.get('entry_time') else None
        if fvg_x is not None:
            z0 = max(0, int(fvg_x) - 1)
            z1 = min(len(chart_df), int(fvg_x) + 4)
            ax.fill_between(range(z0, z1), fvg_bot, fvg_top,
                            alpha=0.25, color='#ffeb3b', zorder=2)
            ax.text(z0 + 0.3, (fvg_top + fvg_bot) / 2, '④ FVG',
                    va='center', ha='left', fontsize=7.5,
                    color='#f57f17', fontweight='bold')

    # ── Entry / Stop / Target ──────────────────────────────────────────────
    hline(signal['entry'], dir_color, '-',   1.5, f"⑤ Entry  {signal['entry']:.2f}",
          y_offset=tick, x_pct=(0.4, 0.98))
    hline(signal['stop'],  C_SHORT,   '--',  1.2, f"⑥ Stop   {signal['stop']:.2f}",
          y_offset=tick, x_pct=(0.4, 0.98))
    hline(signal['target'], C_LONG,   '--',  1.2, f"⑦ Target {signal['target']:.2f}",
          y_offset=tick, x_pct=(0.4, 0.98))

    # Session midpoint
    mid = signal.get('session_midpoint')
    if mid:
        ax.axhline(mid, color='gray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)
        ax.text(0.02, mid + tick * 0.5, 'Session 50%', va='bottom', fontsize=6.5,
                color='gray', transform=ax.get_yaxis_transform())

    # ── Fill / Exit candle highlights ──────────────────────────────────────
    if fill:
        fill_ts = fill['fill_time'] if not isinstance(fill['fill_time'], str) \
                  else pd.Timestamp(fill['fill_time'])
        vline(fill_ts, dir_color, ':', 1.5, 'FILLED', y_frac=0.02)
        fx = ts_to_x(chart_df, fill_ts)
        if fx is not None:
            highlight_candle(fx, chart_df.iloc[int(round(fx))],
                             dir_color, 'ENTRY', label_above=(direction == 'long'))

    if exit_info:
        exit_ts = exit_info['exit_time'] if not isinstance(exit_info['exit_time'], str) \
                  else pd.Timestamp(exit_info['exit_time'])
        reason   = exit_info['exit_reason'].upper()
        ex_color = C_LONG if reason == 'TARGET' else (C_SHORT if reason == 'STOP' else 'darkorange')
        vline(exit_ts, ex_color, '--', 1.5, f'EXIT ({reason})', y_frac=0.15)
        ex = ts_to_x(chart_df, exit_ts)
        if ex is not None:
            label_above = (direction == 'short' and reason == 'STOP') or \
                          (direction == 'long'  and reason == 'TARGET')
            highlight_candle(ex, chart_df.iloc[int(round(ex))],
                             ex_color, f'EXIT\n{reason}', label_above=label_above)

        if pnl_usd is not None:
            sign = '+' if pnl_usd >= 0 else ''
            ax.text(0.98, 0.97, f'{sign}${pnl_usd:,.0f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=14,
                    fontweight='bold',
                    color=C_LONG if pnl_usd >= 0 else C_SHORT)

    # ── Title & legend ─────────────────────────────────────────────────────
    risk_pts = abs(signal['stop'] - signal['entry'])
    rew_pts  = abs(signal['target'] - signal['entry'])
    rr       = rew_pts / risk_pts if risk_pts else 0
    ax.set_title(
        f"ICT 2022 — NQ  {date_str}  |  {direction.upper()}  "
        f"risk={risk_pts:.1f}pts  reward={rew_pts:.1f}pts  R:R={rr:.1f}",
        fontsize=11, fontweight='bold', pad=10,
    )
    ax.set_ylabel('Price', fontsize=9)
    ax.grid(True, alpha=0.2, zorder=0)

    legend_items = [
        mpatches.Patch(color='#9c27b0', label='① Pool level'),
        mpatches.Patch(color='#ff9800', label='② Stop hunt (sweep)'),
        mpatches.Patch(color='#2196f3', label='③ MSS'),
        mpatches.Patch(color='#ffeb3b', alpha=0.5, label='④ FVG zone'),
        mpatches.Patch(color=dir_color,  label=f'⑤ Entry ({direction})'),
        mpatches.Patch(color=C_SHORT,    label='⑥ Stop'),
        mpatches.Patch(color=C_LONG,     label='⑦ Target'),
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=7.5, framealpha=0.9)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.07)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot individual trades from a backtest CSV')
    parser.add_argument('--date', default=None, help='Plot one session (YYYY-MM-DD)')
    parser.add_argument('--n',    type=int, default=12, help='Number of trades (default 12)')
    parser.add_argument('--all',  action='store_true',  help='Plot all trades')
    args = parser.parse_args()

    # Import runner-specific helpers (lazy to avoid circular imports)
    sys.path.insert(0, os.path.dirname(__file__))
    from fvg_sweep_backtest import build_snapshot, DATA_DIR, OUT_DIR

    TRADES_CSV = os.path.join(OUT_DIR, 'fvg_sweep_nq_trades.csv')
    CHARTS_DIR = os.path.join(OUT_DIR, 'charts')
    os.makedirs(CHARTS_DIR, exist_ok=True)

    trades = pd.read_csv(TRADES_CSV)
    if args.date:
        trades = trades[trades['session_date'] == args.date]
        if len(trades) == 0:
            print(f"No trade found for {args.date}")
            return
    elif not args.all:
        trades = trades.head(args.n)

    print(f"Plotting {len(trades)} trade(s)...")
    from infrastructure import DataLoader
    from models.ict.model_2022 import Model2022
    loader = DataLoader(timeframe='1min', weeks=64, data_dir=DATA_DIR)
    df_1m  = loader.read_NQ()
    model  = Model2022()

    for _, row in trades.iterrows():
        session_ts = pd.Timestamp(row['session_date'])
        print(f"\n  {row['session_date']}  {row['direction'].upper()}")
        try:
            snap   = build_snapshot(df_1m, session_ts)
            signal = model.generate_signal(snap)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        if not signal['actionable']:
            print(f"  Signal no longer actionable — skipping")
            continue

        fill      = simulate_fill(snap.df_3m, signal, session_ts)
        exit_info = simulate_exit(snap.df_3m, signal, fill, session_ts) if fill else None
        pnl_usd   = float(row['pnl_usd']) if fill else None
        out_path  = os.path.join(CHARTS_DIR, f"{row['session_date']}_{row['direction']}.png")
        plot_trade(df_1m, session_ts, signal, fill, exit_info, pnl_usd, out_path)

    print(f"\nDone. Charts → {CHARTS_DIR}/")


if __name__ == '__main__':
    main()
