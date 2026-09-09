"""Step 1 of the gated-portfolio build order: how wide is the funnel?

The design needs 2-3 admitted trades a week. Admission is a *rate*, so the
question that has to be answered before anything else is how many raw candidates
per week the opportunity set actually produces. If the funnel is narrow, raising
selectivity necessarily costs trade count and constraints 1 and 2 cannot both be
satisfied - and we would rather know that now than after building five layers on
top of the assumption.

This runner sweeps three index CFDs across three measurement windows, builds the
standard pool set available at each window's open, and reports where every pool
drops out of the funnel:

    pools -> sweeps -> reversals -> entries -> rr_pass

Nothing here is a backtest. No fills, no P&L. It counts opportunities.

Usage:
    python models/ict/gated_portfolio/backtests/funnel_width.py
    python models/ict/gated_portfolio/backtests/funnel_width.py --weeks 104
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_ROOT))

from infrastructure import DataLoader                       # noqa: E402
from models.ict.gated_portfolio.gated_portfolio import (    # noqa: E402
    FUNNEL_STAGES, scan_session)

RESULTS = pathlib.Path(__file__).parent / 'results'

INSTRUMENTS = {
    'NQ': 'USATECHIDXUSD_ALL',
    'ES': 'USA500IDXUSD_ALL',
    'YM': 'USA30IDXUSD_ALL',
}

# Measurement windows in UTC. Deliberately wider than the killzones: the
# killzone set is internally inconsistent in this repo (finding 8 of the judas
# baseline) and narrowing on a disputed definition would understate the funnel.
# These three cover the tradeable day without overlapping.
WINDOWS = {
    'london': ('07:00', '11:00'),
    'ny_am': ('13:30', '16:30'),
    'ny_pm': ('16:30', '20:00'),
}

# Which reference ranges are already formed, and therefore usable as pools, by
# the time each window opens.
POOLS_FOR = {
    'london': ('prior_day', 'asia', 'prior_week'),
    'ny_am': ('prior_day', 'asia', 'london', 'prior_week'),
    'ny_pm': ('prior_day', 'asia', 'london', 'ny_am', 'prior_week'),
}


# Hours of bars before the window opens, handed to the scanner as structural
# context. A sweep on the first bar of the NY open breaks a swing that formed
# hours earlier; without this the scan reports "no reversal" for reasons of
# slicing rather than of price.
CONTEXT_HOURS = 6


def _between(df: pd.DataFrame, day: pd.Timestamp, start: str, end: str):
    lo = pd.Timestamp(f"{day.date()} {start}", tz='UTC')
    hi = pd.Timestamp(f"{day.date()} {end}", tz='UTC')
    return df[(df.index >= lo) & (df.index < hi)]


def _with_context(df: pd.DataFrame, day: pd.Timestamp, start: str, end: str):
    lo = pd.Timestamp(f"{day.date()} {start}", tz='UTC') - pd.Timedelta(hours=CONTEXT_HOURS)
    hi = pd.Timestamp(f"{day.date()} {end}", tz='UTC')
    return df[(df.index >= lo) & (df.index < hi)]


def _hl(frame: pd.DataFrame, name: str) -> list[dict]:
    """Two pools - the high and the low - from a reference range."""
    if frame is None or len(frame) == 0:
        return []
    return [
        {'name': f'{name}_high', 'level': float(frame['high'].max()), 'side': 'high'},
        {'name': f'{name}_low', 'level': float(frame['low'].min()), 'side': 'low'},
    ]


def build_pools(df: pd.DataFrame, day: pd.Timestamp, window: str,
                by_day: dict) -> list[dict]:
    """Pool set formed *before* ``window`` opens on ``day``."""
    wanted = POOLS_FOR[window]
    pools: list[dict] = []
    prev_day = day - pd.Timedelta(days=1)

    if 'prior_day' in wanted:
        pools += _hl(by_day.get(prev_day.date()), 'pdh_pdl')
    if 'asia' in wanted:
        # Asia runs 18:00 the previous day to 00:00 on the session date.
        prev = by_day.get(prev_day.date())
        if prev is not None:
            pools += _hl(prev[prev.index.hour >= 18], 'asia')
    if 'london' in wanted:
        pools += _hl(_between(df, day, *WINDOWS['london']), 'london')
    if 'ny_am' in wanted:
        pools += _hl(_between(df, day, *WINDOWS['ny_am']), 'ny_am')
    if 'prior_week' in wanted:
        wk_start = day - pd.Timedelta(days=day.weekday() + 7)
        wk_end = wk_start + pd.Timedelta(days=7)
        wk = df[(df.index >= wk_start) & (df.index < wk_end)]
        pools += _hl(wk, 'prior_week')

    return pools


def scan_instrument(label: str, df: pd.DataFrame, min_rr: float) -> pd.DataFrame:
    """One row per (day, window) with its funnel stage counts."""
    by_day = {d: g for d, g in df.groupby(df.index.date)}
    days = sorted(by_day)
    rows = []

    for day_date in days:
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
            context = _with_context(df, day, *WINDOWS[window])
            scan = scan_session(bars, pools, context=context,
                                session=window, min_rr=min_rr)
            row = {'instrument': label, 'date': day_date, 'window': window,
                   'candidates': len(scan.candidates)}
            row.update(scan.stages)
            row['best_rr'] = (max(c.reward_risk for c in scan.candidates)
                              if scan.candidates else 0.0)
            rows.append(row)

    return pd.DataFrame(rows)


def report(df: pd.DataFrame, min_rr: float) -> None:
    span_days = (df['date'].max() - df['date'].min()).days
    weeks = max(span_days / 7.0, 1.0)

    print(f"\n{'=' * 68}")
    print(f"FUNNEL WIDTH  |  {df['date'].min()} -> {df['date'].max()}  "
          f"({weeks:.0f} weeks)  |  RR floor {min_rr}")
    print('=' * 68)

    print(f"\n{'stage':<12}{'total':>10}{'per week':>12}{'survival':>12}")
    print('-' * 46)
    prev = None
    for stage in FUNNEL_STAGES:
        total = int(df[stage].sum())
        surv = '' if prev in (None, 0) else f"{total / prev * 100:>11.1f}%"
        print(f"{stage:<12}{total:>10,}{total / weeks:>12.2f}{surv:>12}")
        prev = total

    print(f"\n{'per instrument':<16}{'rr_pass':>10}{'per week':>12}")
    print('-' * 38)
    for inst, g in df.groupby('instrument'):
        n = int(g['rr_pass'].sum())
        print(f"{inst:<16}{n:>10,}{n / weeks:>12.2f}")

    print(f"\n{'per window':<16}{'rr_pass':>10}{'per week':>12}")
    print('-' * 38)
    for win, g in df.groupby('window'):
        n = int(g['rr_pass'].sum())
        print(f"{win:<16}{n:>10,}{n / weeks:>12.2f}")

    raw = df['rr_pass'].sum() / weeks
    print(f"\n{'-' * 68}")
    print(f"raw candidates clearing the {min_rr} R floor: {raw:.2f} / week")
    for target in (2, 3):
        if raw > 0:
            print(f"  -> admission rate for {target}/week: "
                  f"{target / raw * 100:.1f}%  (top {target / raw * 100:.0f}%)")
    if raw < 3:
        print("  -> FUNNEL TOO NARROW: cannot admit 3/week even at 100%.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weeks', type=int, default=None,
                    help='limit history (default: all)')
    ap.add_argument('--timeframe', default='5T')
    ap.add_argument('--min-rr', type=float, default=1.5)
    args = ap.parse_args()

    loader = DataLoader(timeframe=args.timeframe, weeks=args.weeks,
                        data_dir=str(_ROOT / 'Data'))

    frames = []
    for label, symbol in INSTRUMENTS.items():
        print(f"loading {label} ({symbol}) ...", flush=True)
        df = loader._load_and_resample(symbol).dropna(subset=['open'])
        print(f"  {len(df):,} bars  {df.index.min()} -> {df.index.max()}",
              flush=True)
        print(f"scanning {label} ...", flush=True)
        frames.append(scan_instrument(label, df, args.min_rr))

    out = pd.concat(frames, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / 'funnel_width.csv'
    out.to_csv(path, index=False)
    report(out, args.min_rr)
    print(f"\nwritten: {path}")


if __name__ == '__main__':
    main()
