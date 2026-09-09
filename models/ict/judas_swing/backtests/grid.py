#!/usr/bin/env python3
"""
Judas Swing parameter grid
==========================

Sweeps the model's *doctrine* choices — which bias read, which reversal
confirmation, how long the manipulation may take to reject — over a cached set
of snapshots, so a 60-cell grid costs one data load and one resample pass
instead of sixty.

Every axis here is a place where the source corpus offers more than one reading;
the grid measures which reading the market actually paid for, on this data.

Usage:
  python grid.py --config configs/judas_fx.yaml --grid bias
  python grid.py --config configs/judas_fx.yaml --grid full --split
  python grid.py --config configs/judas_fx.yaml --set bias_mode=week_open \
      --set close_back_within_bars=8
"""

import argparse
import itertools
import os
import pathlib as _pl
import sys

import pandas as pd

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'models' / 'ict' / 'model_2022' / 'backtests'))

from judas_backtest import (build_snapshot, precompute_frames,  # noqa: E402
                            execute_signal, apply_pip_scaling,
                            apply_pip_scaling_costs, BACKTEST_KEYS,
                            DATA_DIR, DOLLARS_PER_POINT)
from infrastructure.execution import cost_model_from_config  # noqa: E402
from infrastructure import stats as stats_mod                # noqa: E402
from infrastructure import DataLoader                       # noqa: E402
from models.ict.judas_swing import JudasSwingModel          # noqa: E402


# Each axis is a doctrine fork in the corpus, not a free parameter.
GRIDS = {
    # Which higher-timeframe read supplies the bias the Judas must oppose.
    'bias': {
        'bias_mode': ['strict', 'order_flow', 'draw', 'week_open'],
    },
    # How the manipulation is confirmed as manipulation.
    'confirm': {
        'confirmation': ['cisd', 'mss', 'cisd_or_mss'],
        'max_minutes_to_reversal': [45, 90],
        'close_back_within_bars': [3, 8, 0],
    },
    # What we do once we are in.
    'manage': {
        'target_min_rr': [2.0, 2.5, 3.0],
        'fvg_entry_depth': ['edge', 'ce'],
        'partials': [None, [[0.5, 1.0]], [[0.5, 1.0], [0.25, 2.0]]],
    },
    # The two decisions that dominate everything else: which chart the reversal
    # is read on, and whether we pay up at the confirmation or wait for the
    # retrace. The corpus names the 5-minute chart for both the scalping module
    # and the CISD confirmation; 2-minute is the 2022 model's habit, not this
    # model's doctrine.
    'core': {
        'entry_timeframe': ['2min', '5min', '15min'],
        'entry_style': ['limit', 'market'],
        'bias_mode': ['strict', 'week_open'],
    },
    'stopfloor': {
        'min_stop_atr_mult': [0.0, 0.5, 1.0, 1.5],
        'entry_style': ['limit', 'market'],
    },
    # Target + scale-out shapes. A list of explicit scenarios, not a cross
    # product: `partial_be` is meaningless when there are no partials, and a
    # cross product would spend a third of the run on those dead cells. The
    # no-partials rows exist to separate "the ladder helped" from "the lower
    # R:R floor simply admitted more trades".
    'exits': [
        {'_label': 'A  baseline: 2.5R floor, no partials',
         'target_min_rr': 2.5, 'partials': None},
        {'_label': 'B  1R floor, no partials',
         'target_min_rr': 1.0, 'partials': None},
        {'_label': 'C  1R floor + 50%@1R 25%@1.5R, BE after 1st',
         'target_min_rr': 1.0, 'partials': [[0.5, 1.0], [0.25, 1.5]],
         'partial_be': True},
        {'_label': 'D  1R floor + 50%@1R 25%@1.5R, no BE',
         'target_min_rr': 1.0, 'partials': [[0.5, 1.0], [0.25, 1.5]],
         'partial_be': False},
        # The missing cell: the ladder WITHOUT dropping the R:R floor. Without
        # this row, "the ladder cost expectancy" and "the lower floor cost
        # expectancy" are indistinguishable, because D changes both at once.
        {'_label': 'E  2.5R floor + 50%@1R 25%@1.5R, no BE',
         'target_min_rr': 2.5, 'partials': [[0.5, 1.0], [0.25, 1.5]],
         'partial_be': False},
    ],
    'full': {
        'bias_mode': ['strict', 'order_flow', 'week_open'],
        'confirmation': ['cisd', 'cisd_or_mss'],
        'max_minutes_to_reversal': [45, 90],
        'close_back_within_bars': [3, 8],
        'target_min_rr': [2.0, 3.0],
    },
}


def load_snapshots(instrument: str, start: str, end: str | None, entry_tf: str,
                   verbose: bool = True):
    """Build every session snapshot once; the grid then reuses them."""
    start_dt = pd.Timestamp(start)
    weeks = int((pd.Timestamp.now() - start_dt).days / 7) + 14
    if verbose:
        print(f"Loading {instrument} 1m ...")
    df_1m = DataLoader(timeframe='1min', weeks=weeks, data_dir=DATA_DIR).read(instrument)
    idx = df_1m.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    ny = idx.tz_convert('America/New_York')
    sel = ny >= start_dt.tz_localize('America/New_York')
    if end:
        sel &= ny <= pd.Timestamp(end).tz_localize('America/New_York')
    days = sorted(set(ny.date[sel]))

    frames = precompute_frames(df_1m, entry_tf)
    snaps = []
    for day in days:
        try:
            snap = build_snapshot(df_1m, pd.Timestamp(day), entry_tf, frames)
        except Exception:
            continue
        if len(snap.df_entry) < 50 or len(snap.df_daily) < 10:
            continue
        snaps.append((day, snap))
    if verbose:
        print(f"  {len(snaps)} session snapshots  {snaps[0][0]} -> {snaps[-1][0]}\n")
    return snaps


def evaluate(snaps, cfg: dict, usd_per_point: float,
             instrument: str = 'EURUSD') -> pd.DataFrame:
    """Run one config over the cached snapshots; return the trade log."""
    model_cfg = apply_pip_scaling(
        {k: v for k, v in cfg.items() if k not in BACKTEST_KEYS}, instrument)
    model = JudasSwingModel(config=model_cfg)
    sessions = model.config['sessions']
    directions = (None,) if cfg.get('use_daily_bias', True) else ('bullish', 'bearish')
    partials = cfg.get('partials') or None
    partial_be = bool(cfg.get('partial_be', True))
    be_trigger_r = float(cfg.get('be_trigger_r', 0.0))
    one_per_day = bool(cfg.get('one_trade_per_day', True))

    cost = cost_model_from_config(apply_pip_scaling_costs(cfg, instrument))
    eod_exit_hour = cfg.get('eod_exit_hour')
    if eod_exit_hour is not None:
        eod_exit_hour = int(eod_exit_hour)

    rows = []
    for day, snap in snaps:
        ts = pd.Timestamp(day)
        traded = False
        for session in sessions:
            if traded and one_per_day:
                break
            for forced in directions:
                try:
                    sig = model.generate_signal(snap, session=session,
                                                force_direction=forced)
                except Exception:
                    continue
                if not sig['actionable']:
                    continue
                row = execute_signal(snap, sig, ts, cost=cost,
                                     usd_per_point=usd_per_point,
                                     eod_exit_hour=eod_exit_hour,
                                     be_trigger_r=be_trigger_r,
                                     partials=partials, partial_be=partial_be)
                if row is None:
                    continue
                row.pop('_fill', None)
                row.pop('_exit', None)
                row['date'] = day
                row['reason'] = row['exit_reason']
                rows.append(row)
                traded = True
                break
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, label: str) -> dict:
    """Compact row for a grid table. Delegates the arithmetic to infrastructure.stats
    so a grid cell and a full backtest report can never disagree."""
    st = stats_mod.compute_stats(trades)
    return {
        'config': label,
        'n': st['total_trades'],
        'win%': st['win_pct'],
        'TP%': st['target_hit_pct'],
        'PF': st['profit_factor'],
        'R': st['total_r'],
        'exp_R': st['expectancy_r'],
        'maxDD_R': st['max_dd_r'],
        'usd': round(st['total_pnl_usd']),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--grid', default=None, choices=sorted(GRIDS))
    ap.add_argument('--set', action='append', default=[],
                    help='override a config key, e.g. --set bias_mode=week_open')
    ap.add_argument('--split', action='store_true',
                    help='also report an in-sample / out-of-sample halving')
    ap.add_argument('--min-trades', type=int, default=10)
    ap.add_argument('--sort', default='exp_R')
    args = ap.parse_args()

    import yaml
    path = args.config if os.path.isabs(args.config) else str(_HERE / args.config)
    base = yaml.safe_load(open(path)) or {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        base[k] = yaml.safe_load(v)

    instrument = base.get('instrument', 'EURUSD').upper()
    usd = float(base.get('dollars_per_point', DOLLARS_PER_POINT.get(instrument, 1.0)))
    axes = GRIDS.get(args.grid, {})
    keys = list(axes)
    combos = list(itertools.product(*(axes[k] for k in keys))) if keys else [()]

    # entry_timeframe changes the bars themselves, so snapshots are cached per
    # timeframe rather than once for the whole grid.
    snap_cache: dict = {}

    def snaps_for(tf: str):
        if tf not in snap_cache:
            snap_cache[tf] = load_snapshots(
                instrument, base.get('start_date', '2024-04-01'),
                base.get('end_date'), tf)
        return snap_cache[tf]

    results = []
    for i, combo in enumerate(combos, 1):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        label = ', '.join(f"{k}={v}" for k, v in zip(keys, combo)) or 'base'
        snaps = snaps_for(str(cfg.get('entry_timeframe', '2min')))
        trades = evaluate(snaps, cfg, usd, instrument)
        row = summarize(trades, label)
        if args.split and len(trades):
            mid = len(snaps) // 2
            cut = snaps[mid][0]
            a = trades[trades['date'] < cut]
            b = trades[trades['date'] >= cut]
            row['IS_expR'] = summarize(a, '')['exp_R'] if len(a) else 0.0
            row['OOS_expR'] = summarize(b, '')['exp_R'] if len(b) else 0.0
            row['OOS_n'] = len(b)
        results.append(row)
        print(f"  [{i}/{len(combos)}] {label:<62} n={row['n']:<4} "
              f"win={row['win%']:>5}%  PF={row['PF']:<5}  expR={row['exp_R']:+.3f}")

    df = pd.DataFrame(results)
    keep = df[df['n'] >= args.min_trades]
    if len(keep) == 0:
        keep = df
    keep = keep.sort_values(args.sort, ascending=False)
    print("\n" + "=" * 100)
    print(f"Ranked by {args.sort}  (min {args.min_trades} trades)")
    print("=" * 100)
    print(keep.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
