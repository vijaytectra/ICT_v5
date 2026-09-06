#!/usr/bin/env python3
"""Run walk-forward validation over the configuration logs and report.

Reports the stitched result as-is. There is deliberately no option to drop a
window, resize the windows, or re-rank after seeing the answer - those are
exactly the moves that would turn this back into the in-sample number it exists
to replace.

Usage:
  .venv/Scripts/python.exe models/ict/judas_swing/backtests/walkforward_run.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import infrastructure.stats as stats_mod                            # noqa: E402
from infrastructure.walkforward import (dedupe_logs, normalize_log,  # noqa: E402
                                        run_walkforward)

LOGS = _HERE / 'results' / 'configs'
OUT = _HERE / 'results' / 'walkforward'
FIRST, LAST = '2024-04-05', '2026-09-04'


def _fmt(st: dict) -> str:
    return (f"n={st['total_trades']:>4}  win={st['win_pct']:>5.1f}%  "
            f"PF={st['profit_factor']:>6.3f}  expR={st['expectancy_r']:>+7.3f}  "
            f"totR={st['total_r']:>+8.2f}  maxDD={st['max_dd_r']:>6.2f}")


def main() -> int:
    paths = sorted(LOGS.glob('*.csv'))
    if not paths:
        print(f"no logs in {LOGS} - run sweep_configs.py first")
        return 2
    logs = {p.stem: pd.read_csv(p) for p in paths}
    logs = {k: v for k, v in logs.items() if len(v)}
    print(f"loaded {len(logs)} configuration logs with trades")

    kept, dropped = dedupe_logs(logs)
    print(f"distinct configurations: {len(kept)}  "
          f"(collapsed {len(dropped)} duplicates)")

    stitched, selections = run_walkforward(kept, FIRST, LAST, min_trades=40)

    print("\n### PER-WINDOW SELECTIONS  (each chosen on training data only)")
    print(f"  {'#':>2} {'train ends':<11} {'test period':<24} {'chosen config':<26}"
          f"{'trainN':>7}{'trainExp':>10}{'elig':>6}{'testN':>7}{'testR':>9}")
    rows = []
    for sel in selections:
        w = sel.window
        part = stitched[stitched['window'] == w.index] if len(stitched) else stitched
        test_r = float(part['pnl_r'].sum()) if len(part) else 0.0
        flag = '  [FLOOR RELAXED]' if sel.floor_relaxed else ''
        print(f"  {w.index:>2} {str(w.train_end.date()):<11} "
              f"{str(w.test_start.date()) + '..' + str(w.test_end.date()):<24} "
              f"{sel.config:<26}{sel.train_trades:>7}{sel.train_expectancy:>+10.3f}"
              f"{sel.n_eligible:>6}{len(part):>7}{test_r:>+9.2f}{flag}")
        rows.append({'window': w.index, 'train_end': w.train_end.date(),
                     'test_start': w.test_start.date(), 'test_end': w.test_end.date(),
                     'config': sel.config, 'train_trades': sel.train_trades,
                     'train_expectancy': sel.train_expectancy,
                     'n_eligible': sel.n_eligible, 'floor_used': sel.floor_used,
                     'floor_relaxed': sel.floor_relaxed,
                     'test_trades': len(part), 'test_r': test_r})

    winners = [s.config for s in selections]
    distinct = len(set(winners))
    print(f"\n  distinct winners across {len(selections)} windows: {distinct}"
          f"   ({'stable' if distinct <= 2 else 'THRASHING - selection is unstable'})")

    print("\n### HONEST WALK-FORWARD RESULT")
    wf = stats_mod.compute_stats(stitched)
    print("  " + _fmt(wf))

    print("\n### COMPARISON")
    print(f"  {'walk-forward (honest)':<40} " + _fmt(wf))
    for label, name in [('in-sample winner', 'OF-mf0-r90_be1.25'),
                        ('current production config', 'baseline')]:
        if name in kept:
            print(f"  {label + '  (' + name + ')':<40} "
                  + _fmt(stats_mod.compute_stats(normalize_log(kept[name]))))

    pf = wf['profit_factor']
    verdict = ('EDGE SURVIVES honest selection' if pf > 1.3 else
               'WEAK BUT REAL - partly selection bias' if pf >= 1.0 else
               'SELECTION BIAS - the FX result does not survive')
    print("\n### VERDICT  (thresholds fixed in the spec before this run)")
    print(f"  walk-forward PF = {pf:.3f}   ->   {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(OUT / 'walkforward_trades.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'walkforward_windows.csv', index=False)
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
