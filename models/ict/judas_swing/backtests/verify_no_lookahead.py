#!/usr/bin/env python3
"""Prove that slicing a full-period trade log by date equals running that period alone.

The walk-forward result reuses full-period logs and slices them per window. That
is only legitimate if the model never uses future data. It should not - bias
comes from prior days and ADR from a trailing five-day window - but "should not"
is not evidence.

Exit code 0 means the shortcut is sound. Non-zero means the walk-forward number
is void and every window must be re-run through the model.

Usage:
  .venv/Scripts/python.exe models/ict/judas_swing/backtests/verify_no_lookahead.py
"""
from __future__ import annotations

import copy
import pathlib
import sys
import time
import multiprocessing as mp

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

FX = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
LOGS = _HERE / 'results' / 'configs'
COSTS = {'spread_mode': 'historical', 'gap_fill': 'open'}

# The configurations the walk-forward actually selected, so the verification
# tests the code paths the result depends on rather than an arbitrary pair.
CONFIGS = {
    'baseline': {},
    'mf0-edge-Fri': {'min_fvg_atr_mult': 0.0, 'fvg_entry_depth': 'edge',
                     'skip_weekdays': [4]},
}
COMPARE = ['date', 'instrument', 'direction', 'pnl_r']

_W: dict = {}


def _init():
    import warnings
    import yaml
    warnings.filterwarnings('ignore')
    from grid import load_snapshots
    from judas_backtest import DOLLARS_PER_POINT
    base = yaml.safe_load(open(_HERE / 'configs' / 'judas_fx.yaml'))
    base['costs'] = COSTS
    _W.update(load_snapshots=load_snapshots, dpp=DOLLARS_PER_POINT, base=base)


def _run_restricted(args):
    """Run one configuration over a restricted period, from scratch."""
    inst, overrides, start, end = args
    from grid import evaluate
    cfg = {**copy.deepcopy(_W['base']), **overrides}
    cfg['start_date'] = start
    snaps = _W['load_snapshots'](inst, start, end,
                                 cfg.get('entry_timeframe', '5min'), verbose=False)
    t = evaluate(snaps, cfg, _W['dpp'].get(inst, 1.0), inst)
    if len(t):
        t = t.copy()
        t['instrument'] = inst
    return inst, t


def main() -> int:
    from infrastructure.walkforward import make_windows, normalize_log, slice_dates

    windows = make_windows('2024-04-05', '2026-09-04')
    checks = [windows[0], windows[-1]]
    failures = 0
    t0 = time.time()
    pool = mp.Pool(processes=min(6, mp.cpu_count()), initializer=_init)

    for name, overrides in CONFIGS.items():
        path = LOGS / f'{name}.csv'
        if not path.exists():
            print(f"MISSING {path}")
            pool.close(); pool.join()
            return 2
        full = normalize_log(pd.read_csv(path))

        for w in checks:
            start = w.test_start.date().isoformat()
            end = w.test_end.date().isoformat()
            sliced = slice_dates(full, w.test_start, w.test_end)

            res = pool.map(_run_restricted,
                           [(i, overrides, start, end) for i in FX])
            frames = [t for _, t in res if len(t)]
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            fresh = normalize_log(fresh) if len(fresh) else fresh

            n_s, n_f = len(sliced), len(fresh)
            if n_s != n_f:
                print(f"FAIL {name} w{w.index} ({start}..{end}): "
                      f"sliced {n_s} trades, fresh run {n_f}")
                failures += 1
                continue

            if n_s:
                cols = [c for c in COMPARE
                        if c in sliced.columns and c in fresh.columns]
                a = sliced[cols].sort_values(cols).reset_index(drop=True)
                b = fresh[cols].sort_values(cols).reset_index(drop=True)
                # Compare VALUES, not dtypes. `DataFrame.equals` is dtype-strict,
                # and the two paths legitimately differ there: the sliced log
                # round-tripped through CSV (datetime64[us]) while the fresh run
                # builds Timestamps in memory (datetime64[ns]). Identical instants,
                # different dtype - that is not a lookahead failure.
                num = [c for c in cols if a[c].dtype.kind in 'fc']
                nonnum = [c for c in cols if c not in num]
                bad_cols = [c for c in nonnum if not (a[c].to_numpy() == b[c].to_numpy()).all()]
                maxdiff = (float(max((a[c] - b[c]).abs().max() for c in num))
                           if num else 0.0)
                # Float noise is expected: the same arithmetic in a different
                # order. Anything above ~1e-9 R would be a real divergence.
                if bad_cols or maxdiff > 1e-9:
                    print(f"FAIL {name} w{w.index}: {n_s} trades; "
                          f"differing columns={bad_cols}, max |R diff|={maxdiff:.3e}")
                    print(a.compare(b).head(10).to_string())
                    failures += 1
                    continue
                print(f"     [{name} w{w.index}] values identical "
                      f"(max |R diff| {maxdiff:.1e}, float noise only)")
            print(f"OK   {name} w{w.index} ({start}..{end}): {n_s} trades match"
                  f"   [{time.time()-t0:.0f}s]", flush=True)

    pool.close()
    pool.join()

    if failures:
        print(f"\nVERIFICATION FAILED ({failures}) - the walk-forward shortcut is "
              f"NOT valid. The reported number is void.")
        return 1
    print("\nVERIFICATION PASSED - date-slicing a full-period log is equivalent "
          "to running that period alone.")
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
