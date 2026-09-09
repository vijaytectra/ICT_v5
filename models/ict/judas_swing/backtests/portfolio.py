#!/usr/bin/env python3
"""
Judas Swing across a basket
===========================

Runs one doctrine config over several instruments and reports each one plus the
combined result, optionally sweeping a doctrine axis at basket level.

This is not a convenience wrapper — it is how the pattern is meant to be traded.
ICT is explicit that the Judas is "about one per session across a basket of four
or five pairs", not one per pair per day (ICT-2017-BREAD-BUTTER-BUY 04:33). A
single instrument produces too few setups to judge, and picking a config on one
pair is how you end up fitting its noise.

Snapshots are cached per (instrument, timeframe), so a sweep across six pairs
costs one data load each rather than one per config.

Usage:
  python portfolio.py --config configs/judas_fx.yaml
  python portfolio.py --config configs/judas_fx.yaml --split 2025-09-01
  python portfolio.py --config configs/judas_fx.yaml --grid core
  python portfolio.py --config configs/judas_fx.yaml --instruments EURUSD,GBPUSD
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

from grid import GRIDS, load_snapshots, evaluate, summarize  # noqa: E402
from judas_backtest import DOLLARS_PER_POINT                 # noqa: E402

FX = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
INDICES = ['USATECHIDXUSD', 'USA500IDXUSD', 'USA30IDXUSD']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--instruments', default=None,
                    help="comma list, or 'fx' / 'indices' / 'all'")
    ap.add_argument('--set', action='append', default=[],
                    help='override a config key, e.g. --set bias_mode=week_open')
    ap.add_argument('--grid', default=None, choices=sorted(GRIDS),
                    help='sweep a doctrine axis at basket level')
    ap.add_argument('--split', default=None,
                    help='date splitting in-sample from out-of-sample')
    ap.add_argument('--save', default=None, help='write the combined trade log here')
    args = ap.parse_args()

    import yaml
    path = args.config if os.path.isabs(args.config) else str(_HERE / args.config)
    base = yaml.safe_load(open(path)) or {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        base[k] = yaml.safe_load(v)

    sel = (args.instruments or base.get('instruments') or 'fx')
    if isinstance(sel, str):
        sel = {'fx': FX, 'indices': INDICES, 'all': FX + INDICES}.get(
            sel.lower(), [x.strip().upper() for x in sel.split(',') if x.strip()])

    entry_tf = str(base.get('entry_timeframe', '5min'))
    cache: dict = {}

    def snaps_for(inst: str, tf: str):
        key = (inst, tf)
        if key not in cache:
            print(f"  loading {inst} @ {tf} ...", flush=True)
            cache[key] = load_snapshots(inst, base.get('start_date', '2024-04-01'),
                                        base.get('end_date'), tf, verbose=False)
        return cache[key]

    spec = GRIDS.get(args.grid, {})
    if isinstance(spec, list):
        # Named scenarios: each entry is a complete override set.
        scenarios = [({k: v for k, v in d.items() if k != '_label'},
                      d.get('_label', 'scenario')) for d in spec]
    else:
        keys = list(spec)
        combos = list(itertools.product(*(spec[k] for k in keys))) if keys else [()]
        scenarios = [(dict(zip(keys, c)),
                      ', '.join(f"{k}={v}" for k, v in zip(keys, c)) or 'base')
                     for c in combos]
    keys = [] if len(scenarios) == 1 and not scenarios[0][0] else ['scenario']

    sweep_rows = []
    for overrides, label in scenarios:
        tf = str(overrides.get('entry_timeframe', entry_tf))

        rows, all_trades = [], []
        for inst in sel:
            cfg = dict(base, instrument=inst, **overrides)
            usd = float(cfg.get('dollars_per_point', DOLLARS_PER_POINT.get(inst, 1.0)))
            trades = evaluate(snaps_for(inst, tf), cfg, usd, inst)
            if len(trades):
                trades = trades.copy()
                trades['instrument'] = inst
                all_trades.append(trades)
            rows.append(summarize(trades, inst))

        combined = (pd.concat(all_trades).sort_values('date')
                    if all_trades else pd.DataFrame())
        tot = summarize(combined, label)
        if args.split and len(combined):
            cut = pd.Timestamp(args.split).date()
            a = combined[combined['date'] < cut]
            b = combined[combined['date'] >= cut]
            tot['IS_n'] = len(a)
            tot['IS_expR'] = summarize(a, '')['exp_R'] if len(a) else 0.0
            tot['OOS_n'] = len(b)
            tot['OOS_expR'] = summarize(b, '')['exp_R'] if len(b) else 0.0
        sweep_rows.append(tot)

        print(f"\n### {label}", flush=True)
        for r in rows:
            print(f"  {r['config']:<15} n={r['n']:<4} win={r['win%']:>5}%  "
                  f"PF={r['PF']:<6} expR={r['exp_R']:+.3f}")
        print(f"  {'BASKET':<15} n={tot['n']:<4} win={tot['win%']:>5}%  "
              f"PF={tot['PF']:<6} expR={tot['exp_R']:+.3f}  totalR={tot['R']}",
              flush=True)

        # A single config gets the full breakdown; a sweep gets the table below.
        if not keys and len(combined):
            print("\n" + "=" * 100)
            print("BASKET DETAIL (equal risk per trade)")
            print(f"  exits: {combined['reason'].value_counts().to_dict()}")
            for col in ('session', 'direction', 'instrument'):
                print(f"\n  by {col}:")
                for key, grp in combined.groupby(col):
                    w = (grp['pnl_pts'] > 0).sum()
                    print(f"    {str(key):<16} n={len(grp):<4} "
                          f"win={w / len(grp) * 100:>5.1f}%  "
                          f"R={grp['pnl_r'].sum():>7.2f}  expR={grp['pnl_r'].mean():+.3f}")
            if args.split:
                cut = pd.Timestamp(args.split).date()
                for lbl, part in (('IN-SAMPLE ', combined[combined['date'] < cut]),
                                  ('OUT-SAMPLE', combined[combined['date'] >= cut])):
                    if not len(part):
                        continue
                    t = summarize(part, lbl)
                    print(f"\n  {lbl}: n={t['n']}  win={t['win%']}%  PF={t['PF']}  "
                          f"expR={t['exp_R']:+.3f}")
        if args.save and len(combined):
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            combined.to_csv(args.save, index=False)
            print(f"\n  trade log -> {args.save}")

    if keys:
        print("\n" + "=" * 110)
        print(f"BASKET SWEEP: {args.grid}")
        print("=" * 110)
        print(pd.DataFrame(sweep_rows).sort_values('exp_R', ascending=False)
              .to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
