#!/usr/bin/env python3
"""
SMT Divergence Backtester
=========================

Walk-forward backtest of the SMT model. One evaluation per (session, window):
the AM comparison (05:00-09:30 NY) and the PM comparison (12:00-15:00 NY) are
separate setups with separate pools and separate forced-flat times, so they are
scored separately.

Unlike the other runners in this repo, the *group* is the unit of loading: SMT
needs every correlated instrument on one shared bar index, and the signal names
which instrument to trade. Fills and exits then run on that instrument's frame.

Usage:
  python smt_backtest.py --config configs/smt_indices.yaml
  python smt_backtest.py --config configs/smt_indices.yaml --quiet

Importable:
  from smt_backtest import run_backtest, load_group
  trades, funnel = run_backtest(cfg, data)
"""

from __future__ import annotations

import argparse
import collections
import pathlib as _pl
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_ROOT))

from infrastructure import DataLoader                                # noqa: E402
from infrastructure.execution import (ZERO_COST, cost_model_from_config,  # noqa: E402
                                      simulate_entry, simulate_exit,
                                      trade_costs)
from infrastructure import stats as stats_mod                        # noqa: E402
from infrastructure.time_utils import localize_like                  # noqa: E402
from models.ict.smt_model import SMTModel, SMTSnapshot, WINDOWS, WINDOW_EXIT  # noqa: E402

DATA_DIR = str(_ROOT / 'Data')

# USD per 1.00 price move, per unit traded.
DOLLARS_PER_POINT = {
    'USATECHIDXUSD': 1.0, 'USA500IDXUSD': 1.0, 'USA30IDXUSD': 1.0,
    'USATECHIDXUSD_ALL': 1.0, 'USA500IDXUSD_ALL': 1.0, 'USA30IDXUSD_ALL': 1.0,
    'EURUSD': 100000.0, 'GBPUSD': 100000.0, 'AUDUSD': 100000.0,
    'USDJPY': 640.0, 'USDCHF': 125000.0, 'USDCAD': 72000.0,
    'DOLLARIDXUSD': 1000.0,
    'NQ': 20.0, 'ES': 50.0, 'YM': 5.0,
}

# Keys the runner consumes; everything else is forwarded to SMTModel.
RUNNER_KEYS = {
    'name', 'description', 'group', 'start_date', 'end_date',
    'entry_timeframe', 'costs', 'be_trigger_r', 'partials', 'partial_be',
    'dollars_per_point', 'warmup_weeks',
}


# ─── data ────────────────────────────────────────────────────────────────────

def load_group(group: str, entry_timeframe: str = '5min',
               data_dir: str = DATA_DIR) -> dict:
    """Load one SMT group at the entry timeframe, plus daily and 4H frames.

    Returned once and reused across every arm of an experiment - reloading
    806k bars per arm is the single biggest cost in this study.
    """
    loader = DataLoader(timeframe=entry_timeframe, data_dir=data_dir)
    g = loader.read_smt_group(group)
    frames = g['frames']
    symbols = list(frames)

    # The daily and 4H frames are derived from the entry-timeframe bars rather
    # than re-read from disk. Re-reading meant parsing ~36M CSV rows (3 symbols
    # x 3 timeframes x 4M rows) to produce data already in memory, and it was
    # the slowest step in the whole suite.
    #
    # One caveat, stated rather than hidden: `frames` has been intersected to a
    # common index, so a bar one instrument was missing is absent from all
    # three. On this dataset the intersection is 100% of bars, so the derived
    # daily/4H frames are identical to the re-read ones; on a group with real
    # coverage gaps they would differ slightly at the edges.
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in next(iter(frames.values())).columns:
        agg['volume'] = 'sum'

    daily, four_h = {}, {}
    for s in symbols:
        base = frames[s]
        daily[s] = base.resample('1D').agg(agg).dropna(
            subset=['open', 'high', 'low', 'close'])
        four_h[s] = base.resample('4h').agg(agg).dropna(
            subset=['open', 'high', 'low', 'close'])
    return {'frames': frames, 'daily': daily, 'four_h': four_h,
            'inverse': tuple(g['inverse']), 'symbols': symbols}


def session_dates(frames: dict, start: Optional[str], end: Optional[str]) -> list:
    """NY calendar dates present in the data, bounded by the config."""
    ref = next(iter(frames.values()))
    idx = ref.index
    ny = idx.tz_convert('America/New_York') if idx.tz is not None else idx
    dates = pd.Index(sorted(set(pd.Index(ny.date))))
    if start:
        dates = dates[dates >= pd.Timestamp(start).date()]
    if end:
        dates = dates[dates <= pd.Timestamp(end).date()]
    return list(dates)


# ─── engine ──────────────────────────────────────────────────────────────────

def _exit_timestamp(df: pd.DataFrame, session_date, exit_ny: str) -> pd.Timestamp:
    """Forced-flat moment, built in NY local time and converted back."""
    h, m = map(int, exit_ny.split(':'))
    ts = (pd.Timestamp(session_date).tz_localize('America/New_York')
          + pd.Timedelta(hours=h, minutes=m))
    return ts.tz_convert('UTC') if df.index.tz is not None else ts.tz_localize(None)


def collect_signals(cfg: dict, data: dict) -> Tuple[list, dict]:
    """Generate every actionable signal for a configuration.

    Split out from simulation because signals are the expensive half (11.7 years
    of pivot scans) and are completely independent of execution cost, breakeven
    triggers and partial schedules. Those variants re-simulate the same signal
    list instead of rescanning the data.
    """
    model_cfg = {k: v for k, v in cfg.items() if k not in RUNNER_KEYS}
    model = SMTModel(model_cfg)
    windows = model.config['windows']
    one_per_day = model.config['one_trade_per_day']
    frames, daily, four_h = data['frames'], data['daily'], data['four_h']
    inverse = data['inverse']

    funnel = collections.Counter()
    signals = []
    for d in session_dates(frames, cfg.get('start_date'), cfg.get('end_date')):
        traded_today = False
        for w in windows:
            if one_per_day and traded_today:
                break
            snap = SMTSnapshot(frames=frames, daily=daily,
                               session_date=pd.Timestamp(d),
                               inverse=inverse, four_h=four_h)
            sig = model.generate_signal(snap, window_name=w)
            funnel[f"{w}:{sig.get('reject_stage') or 'ACTIONABLE'}"] += 1
            if sig['actionable']:
                _assert_sane(sig)
                sig['session_date'] = d
                sig['entry_fill_minutes'] = model.config['entry_fill_minutes']
                signals.append(sig)
                traded_today = True
    return signals, dict(funnel)


def _assert_sane(sig: dict) -> None:
    """Invariants every actionable signal must satisfy.

    Cheap, and it fails the run loudly instead of letting a geometrically
    impossible trade through. An inverted target (a long whose target sits
    below its entry) is filled on the very next bar and booked as a winning
    'TP' at a large loss, which quietly poisons every aggregate downstream -
    it is not detectable from expectancy alone, only from TP% exceeding win%.
    """
    e, s, t = float(sig['entry']), float(sig['stop']), float(sig['target'])
    d = sig['direction']
    if d == 'long':
        ok = t > e > s
    else:
        ok = t < e < s
    if not ok:
        raise AssertionError(
            f"impossible {d} geometry on {sig.get('session_date')} "
            f"{sig.get('instrument')}: entry={e:.5f} stop={s:.5f} target={t:.5f}. "
            f"Expected " + ("target > entry > stop" if d == 'long'
                            else "target < entry < stop"))


def simulate_signals(signals: list, cfg: dict, data: dict,
                     quiet: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Fill and exit a signal list under one execution / exit configuration."""
    cost = cost_model_from_config(cfg)
    frames = data['frames']
    dpp_cfg = cfg.get('dollars_per_point') or {}
    extra = collections.Counter()
    rows = []

    for sig in signals:
        w = sig['window']
        sym = sig['instrument']
        df = frames[sym]
        d = sig['session_date']
        eod = _exit_timestamp(df, d, sig['exit_ny'])
        deadline = min(
            localize_like(pd.Timestamp(sig['entry_time']), df.index)
            + pd.Timedelta(minutes=sig['entry_fill_minutes']),
            eod,
        )

        fill = simulate_entry(df, sig, deadline, cost)
        if fill is None:
            extra[f"{w}:no_fill"] += 1
            continue

        # An unfilled-then-negated setup is dropped before the fill; a *filled*
        # one is honoured. The premise dying after we are in the market is a
        # trade outcome, not a signal that never existed.
        ex = simulate_exit(df, sig, fill, eod, cost,
                           be_trigger_r=cfg.get('be_trigger_r', 0.0) or 0.0,
                           partials=cfg.get('partials'),
                           partial_be=cfg.get('partial_be', True))

        risk = abs(float(sig['entry']) - float(sig['stop']))
        dpp = dpp_cfg.get(sym, DOLLARS_PER_POINT.get(sym, 1.0))
        costs = trade_costs(ex, sig['direction'], fill['fill_mid'], cost, dpp)
        hold = (pd.Timestamp(ex['exit_time']) - pd.Timestamp(fill['fill_time'])
                ).total_seconds() / 60.0

        rows.append({
            'entry_time': fill['fill_time'], 'exit_time': ex['exit_time'],
            'instrument': sym, 'window': w, 'side': sig['side'],
            'direction': sig['direction'],
            'entry_price': fill['fill_price'], 'exit_price': ex['exit_price'],
            'stop': sig['stop'], 'target': sig['target'],
            'risk_pts': risk,
            'pnl_pts': ex['pnl_pts'],
            'pnl_r': ex['pnl_pts'] / risk if risk > 0 else 0.0,
            'pnl_usd': ex['pnl_pts'] * dpp,
            'exit_reason': ex['exit_reason'],
            'raw_exit_reason': ex['raw_exit_reason'],
            'holding_minutes': hold,
            'mae_pts': ex['mae_pts'], 'mfe_pts': ex['mfe_pts'],
            'mae_r': ex['mae_pts'] / risk if risk > 0 else 0.0,
            'mfe_r': ex['mfe_pts'] / risk if risk > 0 else 0.0,
            'bars_held': ex['bars_held'],
            'partials_filled': ex['partials_filled'],
            'bias': sig['bias'], 'smt_gate': sig['smt'],
            'leader': sig['leader'], 'diverging': sig['diverging'],
            'clarity_adr': sig['clarity_adr'],
            'sweep_depth_adr': sig['sweep_depth_adr'],
            'reversal_kind': sig['reversal_kind'],
            'entry_array': sig['entry_array'],
            'adr': sig['adr'],
            **{k: costs[k] for k in ('spread_cost_pts', 'slippage_cost_pts',
                                     'spread_cost_usd', 'slippage_cost_usd',
                                     'commission_usd', 'exec_cost_pts',
                                     'exec_cost_usd', 'pnl_gross_pts')},
        })
        if not quiet:
            r = rows[-1]
            print(f"  {r['entry_time']}  {sym:18s} {r['direction']:5s} "
                  f"{r['pnl_r']:+.2f}R  {r['exit_reason']}")

    trades = pd.DataFrame(rows)
    if len(trades):
        trades = trades.sort_values('entry_time').reset_index(drop=True)
        stats_mod.validate_trade_log(trades)
    return trades, dict(extra)


def run_backtest(cfg: dict, data: dict, quiet: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Signals + simulation in one call. Returns (trade log, funnel counts)."""
    signals, funnel = collect_signals(cfg, data)
    trades, extra = simulate_signals(signals, cfg, data, quiet=quiet)
    funnel = dict(collections.Counter(funnel) + collections.Counter(extra))
    return trades, funnel


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _load_yaml(path) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, type=_pl.Path)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--out', type=_pl.Path, default=None)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    print(f"loading group {cfg['group']!r} at {cfg.get('entry_timeframe', '5min')} ...",
          flush=True)
    data = load_group(cfg['group'], cfg.get('entry_timeframe', '5min'))
    ref = next(iter(data['frames'].values()))
    print(f"  {len(ref):,} aligned bars  {ref.index.min()} -> {ref.index.max()}")
    print(f"  instruments: {', '.join(data['symbols'])}"
          + (f"  (inverse: {', '.join(data['inverse'])})" if data['inverse'] else ""))

    trades, funnel = run_backtest(cfg, data, quiet=args.quiet)
    cost = cost_model_from_config(cfg)

    print("\nRejection funnel:")
    for k, v in sorted(funnel.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v:>6,}")

    st = stats_mod.compute_stats(trades, cost)
    stats_mod.print_stats(st, label=f"SMT — {cfg.get('name', args.config.stem)}")

    if len(trades):
        print("\nBy exit reason:")
        print(stats_mod.exit_reason_table(trades).to_string(index=False))
        for by in ('instrument', 'window', 'year'):
            b = stats_mod.breakdown(trades, by)
            if len(b):
                print(f"\nBy {by}:")
                print(b.to_string(index=False))

    out = args.out or (_HERE / 'results' / 'runs' / cfg.get('name', 'run'))
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / 'trades.csv', index=False)
    print(f"\nwrote {out / 'trades.csv'}  ({len(trades):,} trades)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
