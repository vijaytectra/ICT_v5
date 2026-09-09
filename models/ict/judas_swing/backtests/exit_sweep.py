"""Sweep the exit policy with the entries frozen.

The win-rate/payoff frontier study measured `corr(win rate, payoff) = -0.945`
across 13 configurations of one strategy: win rate and payoff are one number,
and the exit policy is what picks the point on the curve. It also measured
`corr(win rate, PF) = +0.667`, with mean PF 1.497 above a 70% win rate against
1.251 below - which is the only published evidence of a PF gradient along any
axis we can move without new alpha.

This runner tests that on our own trades. Every entry, stop and instrument is
taken verbatim from an existing trade log; only the exit rule changes. Each
variant is replayed bar by bar through `infrastructure.exits.simulate_exit` and
scored in-sample, out-of-sample, and through the promotion gate.

Two guards, because a sweep is exactly the machine that manufactures
in-sample winners:

* a **reproduction check** first - the baseline policy must return
  approximately the trade log it came from, or the replay is not modelling the
  same trade and nothing downstream means anything;
* every variant reports **out-of-sample separately**, and the ranking that
  matters is the OOS one.

Usage:
    python models/ict/judas_swing/backtests/exit_sweep.py
    python models/ict/judas_swing/backtests/exit_sweep.py --cut 2025-11-01
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_ROOT))

from infrastructure import DataLoader                      # noqa: E402
from infrastructure.exits import ExitPolicy, simulate_exit  # noqa: E402
from infrastructure.promotion import evaluate_gate          # noqa: E402

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / 'results' / 'exit_sweep'
LOG = HERE / 'results' / 'baseline' / 'fx_spread.csv'

# The baseline engine flattens at this New York hour; the replay must stop at the
# same wall-clock point or the comparison is not like for like.
EOD_HOUR_NY = 16


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(LOG, parse_dates=['entry_time', 'exit_time'])
    df['cost_r'] = (df['exec_cost_pts'] / df['risk_pts']).fillna(0.0)
    # The log's `planned_rr` is computed on mid prices while `entry_price`
    # carries the spread, so the two disagree on every one of the 73 trades -
    # by up to 1.49 R. Rebuilding the structural target from `planned_rr`
    # therefore lands on the wrong price. Derive it from the logged prices
    # instead, which are internally consistent.
    # `risk_pts` reconciles to |entry_mid - stop| exactly: the log measures R on
    # the mid entry and charges the spread separately. The replay must use the
    # same unit or every R is rescaled.
    df['struct_rr'] = (df['target'] - df['entry_price']).abs() / df['risk_pts']
    return df


def load_bars() -> dict:
    loader = DataLoader(timeframe='5min', data_dir=str(_ROOT / 'Data'))
    out = {}
    for sym in DataLoader.FX_SYMBOLS:
        out[sym] = loader._load_and_resample(sym).dropna(subset=['open'])
        print(f"  {sym}: {len(out[sym]):,} bars", flush=True)
    return out


def replay_window(bars: pd.DataFrame, entry_time: pd.Timestamp) -> pd.DataFrame:
    """Bars strictly after the fill, up to the session's New York flat time."""
    eod = (entry_time.tz_convert('America/New_York')
           .normalize()
           .replace(hour=EOD_HOUR_NY)
           .tz_convert('UTC'))
    if eod <= entry_time:
        eod = entry_time + pd.Timedelta(hours=8)
    return bars[(bars.index > entry_time) & (bars.index <= eod)]


def run_policy(trades: pd.DataFrame, bars: dict, policy: ExitPolicy,
               structural: bool = False) -> pd.DataFrame:
    rows = []
    for t in trades.itertuples():
        window = replay_window(bars[t.instrument], t.entry_time)
        p = (ExitPolicy(target_r=float(t.struct_rr),
                        partial_r=policy.partial_r,
                        partial_fraction=policy.partial_fraction,
                        breakeven_r=policy.breakeven_r,
                        time_stop_bars=policy.time_stop_bars)
             if structural else policy)
        # `entry_price` already carries the spread - `exec_cost_pts` is a
        # reporting decomposition of it, not a further charge. Passing both
        # double-counts the cost; with cost_r=0 the baseline policy reproduces
        # the logged engine to 0.0001 R/trade.
        res = simulate_exit(window, float(t.entry_price), float(t.stop),
                            t.direction, p, cost_r=0.0,
                            risk=float(t.risk_pts))
        # ... but that also means the spread was paid once, at entry. A partial
        # exit is a second trip to the market that the logged convention never
        # charges for, which would flatter every partial policy in this sweep.
        # Charge one extra half-turn on the slice actually taken off.
        surcharge = (0.5 * p.partial_fraction * float(t.cost_r)
                     if res.partial_hit else 0.0)
        rows.append({'entry_time': t.entry_time, 'instrument': t.instrument,
                     'pnl_r': res.pnl_r - surcharge,
                     'exit_reason': res.exit_reason,
                     'partial_hit': res.partial_hit, 'mfe_r': res.mfe_r})
    return pd.DataFrame(rows)


def _same_bar_conflicts(trades: pd.DataFrame, bars: dict) -> int:
    """Trades whose first stop touch and first target touch land on one bar."""
    n = 0
    for t in trades.itertuples():
        w = replay_window(bars[t.instrument], t.entry_time)
        if len(w) == 0:
            continue
        long = t.direction == 'long'
        sl = w['low'] <= t.stop if long else w['high'] >= t.stop
        tp = w['high'] >= t.target if long else w['low'] <= t.target
        if sl.any() and tp.any() and w.index[sl][0] == w.index[tp][0]:
            n += 1
    return n


def score(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {'n': 0, 'wr': 0.0, 'payoff': 0.0, 'pf': 0.0, 'exp_r': 0.0,
                'total_r': 0.0}
    r = df['pnl_r']
    w, l = r[r > 1e-9], r[r < -1e-9]
    gp, gl = float(w.sum()), float(-l.sum())
    return {
        'n': len(df),
        'wr': round(len(w) / len(df) * 100, 1),
        'payoff': round(float(w.mean() / -l.mean()), 3) if len(w) and len(l) else 0.0,
        'pf': round(gp / gl, 3) if gl > 0 else float('inf'),
        'exp_r': round(float(r.mean()), 4),
        'total_r': round(float(r.sum()), 2),
    }


def build_grid() -> list:
    """Points along the frontier, from patient-and-rare to quick-and-often."""
    grid = [('structural / no partial / no BE', ExitPolicy(), True)]

    for tr in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        grid.append((f'fixed {tr}R', ExitPolicy(target_r=tr), False))

    for frac, pr in itertools.product((0.25, 0.5, 0.75), (0.6, 0.8, 1.0, 1.5)):
        grid.append((f'structural + {frac:g}@{pr:g}R + BE{pr:g}',
                     ExitPolicy(partial_r=pr, partial_fraction=frac,
                                breakeven_r=pr), True))

    for be in (0.6, 1.0, 1.5):
        grid.append((f'structural + BE{be:g}',
                     ExitPolicy(breakeven_r=be), True))

    for tr, frac, pr in itertools.product((1.5, 2.0, 2.5), (0.5, 0.75),
                                          (0.6, 1.0)):
        grid.append((f'fixed {tr}R + {frac:g}@{pr:g}R + BE{pr:g}',
                     ExitPolicy(target_r=tr, partial_r=pr,
                                partial_fraction=frac, breakeven_r=pr), False))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', default='2025-11-01',
                    help='in-sample / out-of-sample boundary')
    args = ap.parse_args()

    trades = load_trades()
    print(f"trades: {len(trades)}  "
          f"{trades.entry_time.min().date()} -> {trades.entry_time.max().date()}")
    print("loading bars ...", flush=True)
    bars = load_bars()

    cut = pd.Timestamp(args.cut, tz='UTC')

    # --- reproduction control ---------------------------------------------
    base = run_policy(trades, bars, ExitPolicy(), structural=True)
    live = score(trades.rename(columns={'pnl_r': 'pnl_r'})[['pnl_r', 'exit_reason']])
    repro = score(base)
    print("\n" + "=" * 74)
    print("REPRODUCTION CONTROL - baseline policy must return the logged trades")
    print("=" * 74)
    print(f"{'':<12}{'n':>5}{'WR%':>7}{'payoff':>9}{'PF':>8}{'expR':>9}{'totalR':>9}")
    for name, s in (('logged', live), ('replayed', repro)):
        print(f"{name:<12}{s['n']:>5}{s['wr']:>7}{s['payoff']:>9}"
              f"{s['pf']:>8}{s['exp_r']:>9}{s['total_r']:>9}")
    drift = abs(repro['exp_r'] - live['exp_r'])
    print(f"\nexpectancy drift: {drift:.4f} R/trade "
          f"({drift / abs(live['exp_r']) * 100:.1f}% of the baseline)")
    if drift > 0.01:
        print("\n*** WARNING: the replay does not reproduce the logged engine. ***")
        print("*** Every sweep result below is unsafe to act on.             ***")

    same_bar = _same_bar_conflicts(trades, bars)
    print(f"\nbars holding both stop and target: {same_bar} of {len(trades)}"
          f" trades. The replay books these as losses and the logged engine as"
          f"\nwins; without tick data neither is knowable, and the pessimistic"
          f"\nreading is the one that cannot flatter the strategy.")

    # --- the sweep ---------------------------------------------------------
    rows = []
    for name, policy, structural in build_grid():
        res = run_policy(trades, bars, policy, structural=structural)
        res['entry_time'] = pd.to_datetime(res['entry_time'], utc=True)
        is_, oos = res[res.entry_time < cut], res[res.entry_time >= cut]
        s, si, so = score(res), score(is_), score(oos)
        gate = evaluate_gate(is_.assign(exit_reason=is_['exit_reason']),
                             oos.assign(exit_reason=oos['exit_reason']))
        rows.append({'policy': name, **{f'all_{k}': v for k, v in s.items()},
                     **{f'is_{k}': v for k, v in si.items()},
                     **{f'oos_{k}': v for k, v in so.items()},
                     'gate': gate.verdict,
                     'gate_fails': len(gate.failures)})

    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / 'exit_sweep.csv', index=False)

    print("\n" + "=" * 100)
    print("EXIT SWEEP - entries frozen, ranked by FULL-SAMPLE profit factor")
    print("=" * 100)
    print(f"{'policy':<40}{'WR%':>6}{'payoff':>8}{'PF':>7}{'expR':>8}"
          f"{'|':>3}{'oosWR':>7}{'oosPF':>7}{'oosExp':>8}  gate")
    print("-" * 100)
    for r in out.sort_values('all_pf', ascending=False).itertuples():
        print(f"{r.policy[:39]:<40}{r.all_wr:>6}{r.all_payoff:>8}{r.all_pf:>7}"
              f"{r.all_exp_r:>8}{'|':>3}{r.oos_wr:>7}{r.oos_pf:>7}"
              f"{r.oos_exp_r:>8}  {r.gate}")

    print(f"\nwritten: {RESULTS / 'exit_sweep.csv'}")
    print(f"policies clearing the promotion gate: "
          f"{(out.gate == 'PROMOTE').sum()} / {len(out)}")


if __name__ == '__main__':
    main()
