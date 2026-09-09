#!/usr/bin/env python3
"""
Judas Swing rejection funnel
============================

Runs the model over a date range and tallies *which* checklist item killed each
(day, session). A model that fires twice in three years is not selective, it is
broken — this says which gate is doing the killing before any parameter is
touched.

Usage:
  python funnel.py --config configs/judas_fx.yaml [--days 400]
"""

import argparse
import collections
import os
import pathlib as _pl
import re
import sys

import pandas as pd

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_HERE))

from judas_backtest import (build_snapshot, precompute_frames,  # noqa: E402
                            BACKTEST_KEYS, DATA_DIR)
from infrastructure import DataLoader                               # noqa: E402
from models.ict.judas_swing import JudasSwingModel                  # noqa: E402


def _bucket(checks: list) -> str:
    """Name the gate that stopped this evaluation."""
    for line in reversed(checks):
        if line.startswith(('Rule of exclusion', 'Stand aside', 'Judas rejected',
                            'Displacement rejected')):
            # Collapse the variable parts so buckets aggregate.
            return re.sub(r'[-+]?\d[\d.,:%]*', 'N', line)[:110]
    return 'signal'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--days', type=int, default=0, help='limit to the first N days')
    args = ap.parse_args()

    import yaml
    path = args.config if os.path.isabs(args.config) else str(_HERE / args.config)
    cfg = yaml.safe_load(open(path)) or {}

    instrument = cfg.get('instrument', 'EURUSD').upper()
    entry_tf = str(cfg.get('entry_timeframe', '2min'))
    use_bias = bool(cfg.get('use_daily_bias', True))
    model = JudasSwingModel(config={k: v for k, v in cfg.items()
                                    if k not in BACKTEST_KEYS})

    start_dt = pd.Timestamp(cfg.get('start_date', '2024-04-01'))
    weeks = int((pd.Timestamp.now() - start_dt).days / 7) + 14
    df_1m = DataLoader(timeframe='1min', weeks=weeks, data_dir=DATA_DIR).read(instrument)

    idx = df_1m.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    ny = idx.tz_convert('America/New_York')
    days = sorted(set(ny.date[ny >= start_dt.tz_localize('America/New_York')]))
    if args.days:
        days = days[:args.days]

    frames = precompute_frames(df_1m, entry_tf)
    tally = {s: collections.Counter() for s in model.config['sessions']}
    directions = (None,) if use_bias else ('bullish', 'bearish')

    for day in days:
        ts = pd.Timestamp(day)
        try:
            snap = build_snapshot(df_1m, ts, entry_tf, frames)
        except Exception:
            continue
        if len(snap.df_entry) < 50 or len(snap.df_daily) < 10:
            continue
        for session in model.config['sessions']:
            outcomes = []
            for forced in directions:
                sig = model.generate_signal(snap, session=session, force_direction=forced)
                outcomes.append('signal' if sig['actionable'] else _bucket(sig['checks']))
            tally[session]['signal' if 'signal' in outcomes else outcomes[0]] += 1

    print(f"\n{instrument}  {days[0]} -> {days[-1]}  ({len(days)} days)\n")
    for session, counter in tally.items():
        total = sum(counter.values())
        print(f"== {session} ({total} evaluations) " + "=" * 40)
        for reason, n in counter.most_common():
            print(f"  {n:>5}  {n / total * 100:>5.1f}%  {reason}")
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
