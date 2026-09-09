#!/usr/bin/env python3
"""
F2 exposure: how many fills land on the bar that completes the FVG?

`find_fvgs` timestamps a gap to its MIDDLE candle (n), but the gap is only
knowable once candle n+1 has closed. The limit entry scans bars strictly after
that timestamp, so the very first bar it can fill on is n+1 - a bar whose close
was needed to confirm the pattern being traded.

This does not rewrite the detector (the audit says leave it). It measures how
much of the result actually depends on that bar, so "currently harmless" is a
measured statement rather than an assumption.

Usage: python f2_exposure.py results/baseline/fx_zerocost.csv [bar_minutes]
"""
import sys

import pandas as pd


def main(path: str, bar_minutes: int = 5) -> int:
    t = pd.read_csv(path)
    limit = t[t['entry_style'] == 'limit'].copy()
    if len(limit) == 0:
        print('No limit entries in this log; F2 cannot apply.')
        return 0

    entry = pd.to_datetime(limit['entry_time'], utc=True)
    # entry_time in the log is the FILL time; the FVG timestamp is upstream, so
    # reconstruct the exposure from the gap between signal and fill instead.
    if 'signal_time' in limit.columns:
        sig = pd.to_datetime(limit['signal_time'], utc=True)
        lag = (entry - sig).dt.total_seconds() / 60.0
    else:
        print('NOTE: trade log has no signal_time column; re-run the backtest '
              'after adding it to measure this exactly.')
        return 1

    on_confirm_bar = (lag <= bar_minutes + 1e-9)
    print(f"  limit-entry trades      : {len(limit)}")
    print(f"  filled on the FVG's own "
          f"confirming bar (n+1)     : {int(on_confirm_bar.sum())} "
          f"({on_confirm_bar.mean() * 100:.1f}%)")
    exposed = limit[on_confirm_bar.to_numpy()]
    rest = limit[~on_confirm_bar.to_numpy()]
    for lbl, part in (('exposed', exposed), ('clean  ', rest)):
        if len(part):
            print(f"  {lbl}: n={len(part):<4} totalR={part['pnl_r'].sum():+.2f}  "
                  f"expR={part['pnl_r'].mean():+.3f}")
    if len(rest):
        print(f"\n  Result with the exposed trades removed: "
              f"n={len(rest)}  expR={rest['pnl_r'].mean():+.3f}  "
              f"totalR={rest['pnl_r'].sum():+.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1],
                          int(sys.argv[2]) if len(sys.argv) > 2 else 5))
