#!/usr/bin/env python3
"""
SMT Model — the experiment suite
================================

Produces the evidence the analysis document is written from. Arm count is fixed
here, in advance, so the study cannot turn into a search for the best-looking
number across an open-ended set of variants.

Sections
  0  rejection funnel — where the gate stack actually filters
  1  ARM A / B / C    — does the SMT gate add anything at all?
  2  window           — AM (05:00-09:30) vs PM (12:00-15:00)
  3  instrument       — diverging vs leader vs fixed (the corpus contradiction)
  4  exits            — fixed RR, liquidity, time-of-day, partials, breakeven
  5  costs            — zero / historical / 1.5x / 2x stress
  6  validation       — in-sample 2015-2021 vs out-of-sample 2022-2026

Signals are collected once per *signal* configuration and re-simulated for every
execution variant, because cost, breakeven and partial schedules cannot change
which setups exist.

Usage:
  python experiments.py                     # index group, full history
  python experiments.py --group fx_dollar
  python experiments.py --sections 1,2,3
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib as _pl
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from smt_backtest import (load_group, collect_signals, simulate_signals,  # noqa: E402
                          _load_yaml)
from infrastructure import stats as S                                # noqa: E402
from infrastructure.execution import cost_model_from_config, CostModel  # noqa: E402

OUT = _HERE / 'results'
OOS_CUT = '2022-01-01'


# ─── helpers ─────────────────────────────────────────────────────────────────

def _row(label: str, trades: pd.DataFrame, cost=None, extra: dict = None) -> dict:
    st = S.compute_stats(trades, cost)
    out = {
        'arm': label,
        'n': int(st['total_trades']),
        'win%': st['win_pct'],
        'TP%': st['target_hit_pct'],
        'expR': st['expectancy_r'],
        'totalR': st['total_r'],
        'PF': st['profit_factor'],
        'maxDD_R': st['max_dd_r'],
        'sharpe': st['sharpe'],
        'maxLoss': st['max_loss_streak'],
        'holdMin': st['avg_holding_minutes'],
    }
    out.update(extra or {})
    return out


def _table(rows: List[dict], title: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
    if len(df) == 0:
        print("  (no rows)")
        return df
    print(df.to_string(index=False))
    return df


def _split(trades: pd.DataFrame, cut: str = OOS_CUT):
    if len(trades) == 0:
        return trades, trades
    ts = pd.to_datetime(trades['entry_time'], utc=True)
    c = pd.Timestamp(cut, tz='UTC')
    return trades[ts < c], trades[ts >= c]


def retarget(signals: List[dict], rr: float) -> List[dict]:
    """Clone signals with the target moved to a fixed R multiple.

    Entry and stop are untouched, so the trade *set* is frozen and only the exit
    changes. That is what makes the exit comparison a like-for-like test rather
    than six different strategies.
    """
    out = []
    for s in signals:
        t = copy.copy(s)
        risk = abs(float(s['entry']) - float(s['stop']))
        t['target'] = (float(s['entry']) + rr * risk if s['direction'] == 'long'
                       else float(s['entry']) - rr * risk)
        out.append(t)
    return out


COST_REGIMES = {
    'zero (control)': {'spread_mode': 'none', 'gap_fill': 'stop'},
    'historical spread': {'spread_mode': 'historical', 'gap_fill': 'open'},
    'spread x1.5': {'spread_mode': 'historical', 'spread_multiplier': 1.5,
                    'gap_fill': 'open'},
    'spread x2': {'spread_mode': 'historical', 'spread_multiplier': 2.0,
                  'gap_fill': 'open'},
}


# ─── the suite ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=_pl.Path,
                    default=_HERE / 'configs' / 'smt_indices.yaml')
    ap.add_argument('--group', default=None)
    ap.add_argument('--sections', default='0,1,2,3,4,5,6')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    base = _load_yaml(args.config)
    if args.group:
        base['group'] = args.group
    want = {s.strip() for s in args.sections.split(',')}
    tag = args.tag or base['group']
    outdir = OUT / f"experiments_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"loading group {base['group']!r} @ {base.get('entry_timeframe', '5min')} ...",
          flush=True)
    data = load_group(base['group'], base.get('entry_timeframe', '5min'))
    ref = next(iter(data['frames'].values()))
    print(f"  {len(ref):,} aligned bars  {ref.index.min()} -> {ref.index.max()}")
    print(f"  instruments: {', '.join(data['symbols'])}"
          + (f"   inverse: {', '.join(data['inverse'])}" if data['inverse'] else ""))

    cache: Dict[str, tuple] = {}

    def signals_for(name: str, **over):
        """Collect (and memoise) one signal configuration."""
        if name in cache:
            return cache[name]
        cfg = {**base, **over}
        t0 = time.time()
        sigs, funnel = collect_signals(cfg, data)
        print(f"  [{name}] {len(sigs):>4} signals  ({time.time() - t0:.0f}s)",
              flush=True)
        cache[name] = (sigs, funnel, cfg)
        return cache[name]

    hist_cost = cost_model_from_config(base)
    tables = {}

    # ── 0  funnel ───────────────────────────────────────────────────────────
    print("\ncollecting signal arms...", flush=True)
    sigs_A_pm, fun_A_pm, cfg_A_pm = signals_for('A_pm', windows=['pm'])

    if '0' in want:
        tot = sum(fun_A_pm.values())
        rows = [{'stage': k, 'n': v, 'share%': round(v / tot * 100, 1)}
                for k, v in sorted(fun_A_pm.items(), key=lambda kv: -kv[1])]
        tables['funnel'] = _table(
            rows, "0  REJECTION FUNNEL — arm A, PM window "
                  "(where the gate stack actually filters)")

    # ── 1  A / B / C ────────────────────────────────────────────────────────
    if '1' in want or '2' in want:
        sigs_A_am, _, _ = signals_for('A_am', windows=['am'])
        sigs_B_pm, _, _ = signals_for('B_pm', windows=['pm'], require_smt=False)
        sigs_B_am, _, _ = signals_for('B_am', windows=['am'], require_smt=False)
        # ARM C: divergence alone. No live pool, no rejection, no reversal -
        # enter at market on the signal bar. The strawman retail actually trades.
        sigs_C_pm, _, _ = signals_for(
            'C_pm', windows=['pm'], require_pool_live=False,
            require_rejection=False, require_reversal=False,
            entry_style='market')
        # AM+PM is the deployable configuration - both windows are doctrine and
        # each is its own setup - so it carries the cost and validation work.
        sigs_A_both, _, _ = signals_for('A_both', windows=['am', 'pm'])
        sigs_B_both, _, _ = signals_for('B_both', windows=['am', 'pm'],
                                        require_smt=False)

        rows = []
        for lbl, sg in (('A  SMT ON   (AM+PM)', sigs_A_both),
                        ('B  SMT OFF  (AM+PM)', sigs_B_both),
                        ('A  SMT gate ON   (PM)', sigs_A_pm),
                        ('B  SMT gate OFF  (PM)', sigs_B_pm),
                        ('C  SMT alone     (PM)', sigs_C_pm),
                        ('A  SMT gate ON   (AM)', sigs_A_am),
                        ('B  SMT gate OFF  (AM)', sigs_B_am)):
            tr, _ = simulate_signals(sg, base, data)
            rows.append(_row(lbl, tr, hist_cost))
        tables['arms'] = _table(
            rows, "1  DOES THE SMT GATE ADD ANYTHING?  "
                  "(A vs B is the whole question; historical spread)")

    # ── 2  window ───────────────────────────────────────────────────────────
    if '2' in want:
        rows = []
        for lbl, sg in (('AM 05:00-09:30', sigs_A_am),
                        ('PM 12:00-15:00', sigs_A_pm)):
            tr, _ = simulate_signals(sg, base, data)
            rows.append(_row(lbl, tr, hist_cost))
        tables['window'] = _table(rows, "2  COMPARISON WINDOW (arm A)")

    # ── 3  instrument selection ─────────────────────────────────────────────
    if '3' in want:
        rows = []
        for rule in ('diverging', 'leader', 'fixed'):
            # 'diverging' is the config's own default, so arm A already holds it.
            key = 'A_pm' if rule == base.get('instrument_selection',
                                             'diverging') else f'sel_{rule}'
            sg, _, _ = signals_for(key, windows=['pm'],
                                   instrument_selection=rule)
            tr, _ = simulate_signals(sg, base, data)
            rows.append(_row(f"{rule}", tr, hist_cost))
        tables['selection'] = _table(
            rows, "3  WHICH INSTRUMENT TO TRADE  "
                  "(the corpus teaches all three; PM window)")

    # ── 4  exits ────────────────────────────────────────────────────────────
    if '4' in want:
        # One frozen trade set: same entries and stops, target pushed far away
        # and the min-RR gate disabled so nothing is filtered out by the target.
        sigs_free, _, cfg_free = signals_for(
            'exit_base', windows=['am', 'pm'], target_mode='time',
            target_min_rr=0.0, cap_target_at_adr=False)

        rows = []
        for rr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            tr, _ = simulate_signals(retarget(sigs_free, rr), base, data)
            rows.append(_row(f"fixed {rr}R", tr, hist_cost))

        tr, _ = simulate_signals(sigs_free, base, data)
        rows.append(_row("time-of-day exit only", tr, hist_cost))

        sg_liq, _, _ = signals_for('exit_liq', windows=['am', 'pm'],
                                   target_mode='liquidity', target_min_rr=1.5)
        tr, _ = simulate_signals(sg_liq, base, data)
        rows.append(_row("liquidity target", tr, hist_cost))

        sg_cap, _, _ = signals_for('exit_adrcap', windows=['am', 'pm'],
                                   cap_target_at_adr=True)
        tr, _ = simulate_signals(sg_cap, base, data)
        rows.append(_row("2R + ADR cap on", tr, hist_cost))

        for be in (0.5, 1.0, 1.5):
            cfg = {**base, 'be_trigger_r': be}
            tr, _ = simulate_signals(retarget(sigs_free, 2.0), cfg, data)
            rows.append(_row(f"2R + BE at {be}R", tr, hist_cost))

        for frac, at in ((0.5, 1.0), (0.5, 1.5), (0.5, 2.0)):
            cfg = {**base, 'partials': [[frac, at]], 'partial_be': True}
            tr, _ = simulate_signals(retarget(sigs_free, 3.0), cfg, data)
            rows.append(_row(f"3R, {frac:.0%} off at {at}R +BE", tr, hist_cost))

        tables['exits'] = _table(
            rows, "4  EXIT ARCHITECTURE — one frozen trade set, "
                  "identical entries and stops")

        # MFE / MAE distribution: where price actually went.
        tr_free, _ = simulate_signals(sigs_free, base, data)
        if len(tr_free):
            q = [0.1, 0.25, 0.5, 0.75, 0.9]
            print("\n  MFE / MAE distribution (R), time-exit set — "
                  "what the trades were actually worth:")
            print(f"    {'quantile':<10}" + ''.join(f"{x:>9.0%}" for x in q))
            print(f"    {'MFE':<10}" +
                  ''.join(f"{tr_free['mfe_r'].quantile(x):>9.2f}" for x in q))
            print(f"    {'MAE':<10}" +
                  ''.join(f"{tr_free['mae_r'].quantile(x):>9.2f}" for x in q))
            for k in (1.0, 1.5, 2.0, 3.0):
                print(f"    reached {k}R MFE: "
                      f"{(tr_free['mfe_r'] >= k).mean() * 100:5.1f}% of trades")

    # ── 5  costs ────────────────────────────────────────────────────────────
    if '5' in want:
        rows = []
        for lbl, spec in COST_REGIMES.items():
            cfg = {**base, 'costs': spec}
            tr, _ = simulate_signals(signals_for('A_both', windows=['am', 'pm'])[0],
                                     cfg, data)
            rows.append(_row(lbl, tr, cost_model_from_config(cfg)))
        tables['costs'] = _table(
            rows, "5  EXECUTION COST REGIMES (arm A, AM+PM — identical trades, "
                  "only what they pay changes)")

    # ── 6  validation ───────────────────────────────────────────────────────
    if '6' in want:
        tr, _ = simulate_signals(signals_for('A_both', windows=['am', 'pm'])[0],
                                 base, data)
        ins, oos = _split(tr)
        rows = [_row(f"in-sample  (< {OOS_CUT})", ins, hist_cost),
                _row(f"out-of-sample (>= {OOS_CUT})", oos, hist_cost)]
        tables['validation'] = _table(
            rows, f"6  VALIDATION — split at {OOS_CUT}, chosen because the "
                  f"exploratory edge decays into that era")

        if len(tr):
            for by in ('year', 'instrument', 'window', 'side'):
                b = S.breakdown(tr, by)
                if len(b):
                    print(f"\n  by {by}:")
                    print(b.to_string(index=False))
            print("\n  exit reasons:")
            print(S.exit_reason_table(tr).to_string(index=False))
            tr.to_csv(outdir / 'arm_A_pm_trades.csv', index=False)

    for name, df in tables.items():
        if len(df):
            df.to_csv(outdir / f'{name}.csv', index=False)
    print(f"\nwrote tables to {outdir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
