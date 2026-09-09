#!/usr/bin/env python3
"""
SMT Entry Study
===============

Answers one question: which entry technique makes 1.5R reachable?

The delayed FVG-retrace entry produced a median adverse excursion (1.03R)
larger than the median favourable one (0.76R), and only 29% of trades ever
touched 1.5R. ICT teaches two entries for this setup and neither is a delayed
retrace, so both are implemented and measured here against it.

Reported per technique:
  signals / fills / fill%   - a tight entry that never fills is not an edge
  reach1.5R%                - the fraction that touch 1.5R before the stop,
                              which is the ceiling on win rate under the
                              partial-at-1.5R exit
  expR / PF / win%          - at the configured target

Usage:
  python entry_study.py --group indices --start 2015-03-01 --end 2021-12-31
  python entry_study.py --group fx_dollar
"""

from __future__ import annotations

import argparse
import pathlib as _pl
import sys
import time

import numpy as np
import pandas as pd

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from smt_backtest import (load_group, collect_signals, simulate_signals,  # noqa: E402
                          _load_yaml)
from infrastructure import stats as S                              # noqa: E402
from infrastructure.execution import cost_model_from_config        # noqa: E402

# (label, config overrides). Everything else is held fixed.
#
# The detection MODE matters as much as the entry: 'pivot' compares two
# confirmed swing pivots and is late by construction, so it can only support a
# retrace entry. 'breach' treats the pool as history and the run through it as
# the live signal, which is both strictly causal and the only reading a
# turtle-soup entry can be filled on.
TECHNIQUES = [
    ('pivot + fvg retrace (baseline)',
     {'smt_mode': 'pivot', 'entry_style': 'fvg', 'require_reversal': True}),
    # turtle soup must be traded on the LEADER - the instrument that ran the
    # stops. A diverging instrument has no sweep zone to buy into.
    ('breach + turtle soup 0.5',
     {'smt_mode': 'breach', 'entry_style': 'turtle_soup',
      'turtle_entry_depth': 0.5, 'require_reversal': False,
      'instrument_selection': 'leader'}),
    ('breach + turtle soup 0.0 (at level)',
     {'smt_mode': 'breach', 'entry_style': 'turtle_soup',
      'turtle_entry_depth': 0.0, 'require_reversal': False,
      'instrument_selection': 'leader'}),
    ('breach + turtle soup + reversal',
     {'smt_mode': 'breach', 'entry_style': 'turtle_soup',
      'turtle_entry_depth': 0.5, 'require_reversal': True,
      'instrument_selection': 'leader'}),
    ('breach + buy stop',
     {'smt_mode': 'breach', 'entry_style': 'buy_stop', 'require_reversal': False,
      'instrument_selection': 'leader'}),
    ('breach + market',
     {'smt_mode': 'breach', 'entry_style': 'market', 'require_reversal': False}),
    ('breach + market + reversal',
     {'smt_mode': 'breach', 'entry_style': 'market', 'require_reversal': True}),
    ('breach + fvg retrace',
     {'smt_mode': 'breach', 'entry_style': 'fvg', 'require_reversal': True}),
]

# Gate stacks, so the frequency cost of the bias veto is visible.
GATES = [
    ('strict bias, am+pm',  {'bias_mode': 'strict', 'windows': ['am', 'pm']}),
    ('no bias, am+pm',      {'bias_mode': 'off', 'windows': ['am', 'pm']}),
    ('no bias, +london',    {'bias_mode': 'off', 'windows': ['london', 'am', 'pm']}),
]


def summarise(label, signals, trades, cost, months):
    st = S.compute_stats(trades, cost)
    n_sig = len(signals)
    n = st['total_trades']
    reach = (float((trades['mfe_r'] >= 1.5).mean() * 100)
             if n and 'mfe_r' in trades else 0.0)
    return {
        'technique': label,
        'signals': n_sig,
        'fills': n,
        'fill%': round(n / n_sig * 100, 1) if n_sig else 0.0,
        'per_month': round(n / months, 2) if months else 0.0,
        'reach1.5R%': round(reach, 1),
        'win%': st['win_pct'],
        'expR': st['expectancy_r'],
        'PF': st['profit_factor'],
        'maxDD_R': st['max_dd_r'],
        'medMFE': round(float(trades['mfe_r'].median()), 2) if n else 0.0,
        'medMAE': round(float(trades['mae_r'].median()), 2) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=_pl.Path, default=None)
    ap.add_argument('--group', default='indices')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--gates', default='all')
    ap.add_argument('--only', default=None,
                    help='comma-separated substrings; run only matching techniques')
    args = ap.parse_args()

    cfg_path = args.config or (_HERE / 'configs' /
                               ('smt_fx.yaml' if args.group == 'fx_dollar'
                                else 'smt_indices.yaml'))
    base = _load_yaml(cfg_path)
    base['group'] = args.group
    if args.start:
        base['start_date'] = args.start
    if args.end:
        base['end_date'] = args.end

    print(f"loading {args.group!r} ...", flush=True)
    data = load_group(args.group, base.get('entry_timeframe', '5min'))
    ref = next(iter(data['frames'].values()))
    lo = max(pd.Timestamp(base['start_date'], tz='UTC'), ref.index.min())
    hi = (pd.Timestamp(base['end_date'], tz='UTC') if base.get('end_date')
          else ref.index.max())
    months = max((hi - lo).days / 30.44, 1e-9)
    print(f"  {len(ref):,} bars   {lo.date()} -> {hi.date()}   {months:.1f} months")
    print(f"  instruments: {', '.join(data['symbols'])}"
          + (f"   inverse: {', '.join(data['inverse'])}" if data['inverse'] else ""))

    cost = cost_model_from_config(base)
    techs = TECHNIQUES
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(',') if k.strip()]
        techs = [t for t in TECHNIQUES
                 if any(k in t[0].lower() for k in keys)]
    gates = GATES if args.gates == 'all' else [g for g in GATES if g[0] in args.gates]

    for gate_label, gate_over in gates:
        rows = []
        for label, over in techs:
            cfg = {**base, **gate_over, **over}
            t0 = time.time()
            sigs, _ = collect_signals(cfg, data)
            trades, _ = simulate_signals(sigs, cfg, data)
            rows.append(summarise(label, sigs, trades, cost, months))
            print(f"    {label:<26} {len(sigs):>4} sig  "
                  f"{len(trades):>4} fills  ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        print(f"\n{'=' * 118}\nENTRY TECHNIQUE — {gate_label}\n{'=' * 118}")
        print(df.to_string(index=False))
        out = _HERE / 'results' / f'entry_study_{args.group}'
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{gate_label.replace(' ', '_').replace(',', '')}.csv",
                  index=False)
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
