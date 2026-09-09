"""Exit-policy sweep on an 11-year index-CFD candidate set.

The judas exit sweep found breakeven-at-1.5R the best single change (PF
1.254 -> 1.614) but could not promote it: 18 out-of-sample trades against 36
policies is the textbook selection-bias setup. The answer needs a sample where
the promotion gate can actually rule.

**What this is not.** It is not the 12-year PF 1.54 production system; that
trade log is not in this repository. It is the L1/L2 candidate set from
`gated_portfolio.scan_session` - the common ICT skeleton of sweep a pool, shift
structure against it, enter on the resulting imbalance - run over ES/NQ/YM from
2015 to 2026. Those entries are unvalidated. What the large sample can settle is
the narrower question the judas sweep raised: **on this trade archetype, does an
exit policy change survive out-of-sample once there are enough trades to tell?**

Fills are simulated rather than assumed: the FVG entry is a resting limit, valid
until the session's flat time, and a candidate that never trades back to its
level is not a trade. That is its own funnel stage and it is reported.

Usage:
    python .../portfolio_exit_sweep.py --scan     # build candidates.csv (slow)
    python .../portfolio_exit_sweep.py            # sweep the cached candidates
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
from models.ict.gated_portfolio.gated_portfolio import scan_session  # noqa: E402
from models.ict.gated_portfolio.backtests.funnel_width import (       # noqa: E402
    INSTRUMENTS, WINDOWS, _between, _with_context, build_pools)

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / 'results'
CANDIDATES = RESULTS / 'candidates.csv'

EOD_HOUR_NY = 16


def scan_all(min_rr: float, timeframe: str) -> pd.DataFrame:
    """Every candidate, with a simulated limit fill."""
    loader = DataLoader(timeframe=timeframe, data_dir=str(_ROOT / 'Data'))
    rows = []
    for label, symbol in INSTRUMENTS.items():
        print(f"loading {label} ...", flush=True)
        df = loader._load_and_resample(symbol).dropna(subset=['open'])
        has_spread = 'spread' in df.columns
        by_day = {d: g for d, g in df.groupby(df.index.date)}
        print(f"scanning {label}: {len(by_day)} days", flush=True)

        for day_date in sorted(by_day):
            day = pd.Timestamp(day_date, tz='UTC')
            if day.weekday() >= 5:
                continue
            for window in WINDOWS:
                bars = _between(df, day, *WINDOWS[window])
                if len(bars) < 10:
                    continue
                pools = build_pools(df, day, window, by_day)
                if not pools:
                    continue
                scan = scan_session(bars, pools,
                                    context=_with_context(df, day, *WINDOWS[window]),
                                    session=window, min_rr=min_rr)
                for c in scan.candidates:
                    fill = _first_fill(df, c, day)
                    if fill is None:
                        rows.append({'instrument': label, 'filled': False})
                        continue
                    fill_time, spread = fill if has_spread else (fill[0], np.nan)
                    rows.append({
                        'instrument': label, 'filled': True,
                        'session': c.session, 'direction': c.direction,
                        'entry_time': fill_time, 'entry_price': c.entry,
                        'stop': c.stop, 'target': c.target,
                        'swept_pool': c.swept_pool, 'reward_risk': c.reward_risk,
                        'spread': spread,
                    })
    return pd.DataFrame(rows)


def _first_fill(df: pd.DataFrame, cand, day: pd.Timestamp):
    """First bar after the shift whose range contains the limit level."""
    end = _flat_time(day)
    w = df[(df.index > cand.mss_time) & (df.index <= end)]
    if len(w) == 0:
        return None
    touched = (w['low'] <= cand.entry) & (w['high'] >= cand.entry)
    if not touched.any():
        return None
    ts = w.index[touched][0]
    spread = float(w.loc[ts, 'spread']) if 'spread' in w.columns else np.nan
    return ts, spread


def _flat_time(day: pd.Timestamp) -> pd.Timestamp:
    return (day.tz_convert('America/New_York').normalize()
            .replace(hour=EOD_HOUR_NY).tz_convert('UTC'))


def replay_window(bars: pd.DataFrame, entry_time: pd.Timestamp) -> pd.DataFrame:
    end = _flat_time(entry_time.tz_convert('UTC').normalize())
    if end <= entry_time:
        end = entry_time + pd.Timedelta(hours=8)
    return bars[(bars.index > entry_time) & (bars.index <= end)]


def score(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {'n': 0, 'wr': 0.0, 'payoff': 0.0, 'pf': 0.0, 'exp_r': 0.0}
    r = df['pnl_r']
    w, l = r[r > 1e-9], r[r < -1e-9]
    gp, gl = float(w.sum()), float(-l.sum())
    return {'n': len(df),
            'wr': round(len(w) / len(df) * 100, 1),
            'payoff': round(float(w.mean() / -l.mean()), 3) if len(w) and len(l) else 0.0,
            'pf': round(gp / gl, 3) if gl > 0 else float('inf'),
            'exp_r': round(float(r.mean()), 4)}


def build_grid() -> list:
    grid = [('structural / no partial / no BE', ExitPolicy(), True)]
    for be in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        grid.append((f'structural + BE{be:g}', ExitPolicy(breakeven_r=be), True))
    for tr in (1.0, 1.5, 2.0, 2.5, 3.0):
        grid.append((f'fixed {tr:g}R', ExitPolicy(target_r=tr), False))
        grid.append((f'fixed {tr:g}R + BE1.5',
                     ExitPolicy(target_r=tr, breakeven_r=1.5), False))
    for frac, pr in itertools.product((0.25, 0.5, 0.75), (0.75, 1.0, 1.5)):
        grid.append((f'structural + {frac:g}@{pr:g}R + BE{pr:g}',
                     ExitPolicy(partial_r=pr, partial_fraction=frac,
                                breakeven_r=pr), True))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true', help='rebuild candidates.csv')
    ap.add_argument('--min-rr', type=float, default=1.5)
    ap.add_argument('--timeframe', default='5min')
    ap.add_argument('--cut', default='2023-01-01')
    args = ap.parse_args()

    if args.scan or not CANDIDATES.exists():
        cands = scan_all(args.min_rr, args.timeframe)
        RESULTS.mkdir(parents=True, exist_ok=True)
        cands.to_csv(CANDIDATES, index=False)
        print(f"written: {CANDIDATES}")
    cands = pd.read_csv(CANDIDATES, parse_dates=['entry_time'])

    total = len(cands)
    trades = cands[cands['filled'] == True].copy()          # noqa: E712
    print(f"\ncandidates {total:,}  filled {len(trades):,} "
          f"({len(trades) / total * 100:.1f}%)  "
          f"{trades.entry_time.min().date()} -> {trades.entry_time.max().date()}")

    trades['risk'] = (trades.entry_price - trades.stop).abs()
    trades['struct_rr'] = (trades.target - trades.entry_price).abs() / trades['risk']
    trades['cost_r'] = (trades['spread'].fillna(0.0) / trades['risk']).clip(upper=1.0)
    print(f"cost as a share of risk: median {trades.cost_r.median():.4f} R, "
          f"mean {trades.cost_r.mean():.4f} R")

    loader = DataLoader(timeframe=args.timeframe, data_dir=str(_ROOT / 'Data'))
    bars = {}
    for label, symbol in INSTRUMENTS.items():
        bars[label] = loader._load_and_resample(symbol).dropna(subset=['open'])

    cut = pd.Timestamp(args.cut, tz='UTC')
    rows = []
    for name, policy, structural in build_grid():
        out = []
        for t in trades.itertuples():
            p = (ExitPolicy(target_r=float(t.struct_rr),
                            partial_r=policy.partial_r,
                            partial_fraction=policy.partial_fraction,
                            breakeven_r=policy.breakeven_r)
                 if structural else policy)
            w = replay_window(bars[t.instrument], t.entry_time)
            res = simulate_exit(w, float(t.entry_price), float(t.stop),
                                t.direction, p, cost_r=float(t.cost_r),
                                risk=float(t.risk))
            surcharge = (0.5 * p.partial_fraction * float(t.cost_r)
                         if res.partial_hit else 0.0)
            out.append({'entry_time': t.entry_time,
                        'pnl_r': res.pnl_r - surcharge,
                        'exit_reason': res.exit_reason})
        res_df = pd.DataFrame(out)
        res_df['entry_time'] = pd.to_datetime(res_df['entry_time'], utc=True)
        is_, oos = res_df[res_df.entry_time < cut], res_df[res_df.entry_time >= cut]
        s, si, so = score(res_df), score(is_), score(oos)
        gate = evaluate_gate(is_, oos)
        rows.append({'policy': name, **{f'all_{k}': v for k, v in s.items()},
                     **{f'is_{k}': v for k, v in si.items()},
                     **{f'oos_{k}': v for k, v in so.items()},
                     'gate': gate.verdict, 'gate_fails': len(gate.failures)})
        print(f"  {name:<38} PF {s['pf']:>6.3f}  oosPF {so['pf']:>6.3f}  "
              f"{gate.verdict}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'portfolio_exit_sweep.csv', index=False)

    print("\n" + "=" * 104)
    print(f"PORTFOLIO EXIT SWEEP - {len(trades):,} trades, IS/OOS cut {args.cut}")
    print("=" * 104)
    print(f"{'policy':<40}{'WR%':>6}{'payoff':>8}{'PF':>7}{'expR':>8}"
          f"{'|':>3}{'oosN':>6}{'oosWR':>7}{'oosPF':>7}{'oosExp':>8}  gate")
    print("-" * 104)
    for r in out.sort_values('oos_pf', ascending=False).itertuples():
        print(f"{r.policy[:39]:<40}{r.all_wr:>6}{r.all_payoff:>8}{r.all_pf:>7}"
              f"{r.all_exp_r:>8}{'|':>3}{r.oos_n:>6}{r.oos_wr:>7}{r.oos_pf:>7}"
              f"{r.oos_exp_r:>8}  {r.gate}")
    print(f"\npromoted: {(out.gate == 'PROMOTE').sum()} / {len(out)}")
    print(f"written: {RESULTS / 'portfolio_exit_sweep.csv'}")


if __name__ == '__main__':
    main()
