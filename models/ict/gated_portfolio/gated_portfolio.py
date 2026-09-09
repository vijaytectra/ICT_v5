"""Gated Portfolio Queue - L1 candidate generation and L2 rejection gates.

See `gated-portfolio.md` for the design. This module implements the two layers
that have to exist before anything else can be measured:

* **L1** - generate candidates from the common ICT skeleton (sweep a pool, shift
  structure against it, enter on the resulting imbalance) rather than from any
  one model's idiosyncratic rules. Model-agnostic on purpose: the point of the
  funnel-width measurement is to size the opportunity set that *any* of our
  models draws from.
* **L2** - the hard rejection gates, of which only the dynamic reward-risk floor
  is implemented here. Targets come from real structural levels, so R is
  computed, never assumed.

`scan_session` reports where each pool dropped out of the funnel, which is the
number the build order needs first: if the funnel is not wide enough, no amount
of grading can raise win rate and hold trade count at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from concepts import detect_liquidity_sweep, detect_mss, find_fvgs

FUNNEL_STAGES = ('pools', 'sweeps', 'reversals', 'entries', 'rr_pass')

# A swept buy-side pool implies downside delivery, and vice versa.
_AFTER_SWEEP = {
    'high': {'direction': 'short', 'mss': 'bearish', 'fvg': 'bearish',
             'target_side': 'low'},
    'low': {'direction': 'long', 'mss': 'bullish', 'fvg': 'bullish',
            'target_side': 'high'},
}


@dataclass(frozen=True)
class Candidate:
    """A fully specified would-be trade. Not yet admitted - see L3/L4."""
    session: str
    direction: str
    swept_pool: str
    swept_level: float
    sweep_time: pd.Timestamp
    mss_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    target_pool: str
    reward_risk: float


@dataclass(frozen=True)
class SessionScan:
    candidates: list[Candidate]
    stages: dict


def scan_session(df: pd.DataFrame, pools: list[dict], *,
                 context: Optional[pd.DataFrame] = None, session: str = '',
                 min_rr: float = 1.5, max_bars_after_sweep: int = 30,
                 swing_lookback: int = 3) -> SessionScan:
    """Scan one session's bars against one set of liquidity pools.

    ``pools`` are dicts of ``{name, level, side}`` where side is 'high' (buy-side)
    or 'low' (sell-side). Every pool is tested independently, so one session can
    produce several candidates - that is the funnel width we are trying to size.

    ``df`` is the window a sweep must happen *in*. ``context`` is the longer
    frame - the window plus whatever precedes it - used to read structure. The
    split matters: a sweep landing on the first bar of the NY open has no
    earlier bars inside the window, so the swing its shift would break was
    formed before the window opened. Scanning the window alone reports "no
    reversal" for reasons of slicing rather than of price. ``context`` defaults
    to ``df``, which is the degenerate case.
    """
    stages = {k: 0 for k in FUNNEL_STAGES}
    stages['pools'] = len(pools)
    candidates: list[Candidate] = []

    if df is None or len(df) == 0:
        return SessionScan(candidates=[], stages=stages)
    frame = df if context is None else context

    for pool in pools:
        side = pool['side']
        plan = _AFTER_SWEEP.get(side)
        if plan is None:
            continue

        sweep = detect_liquidity_sweep(df, pool['level'], side)
        if sweep is None or sweep['classification'] != 'sweep':
            continue
        stages['sweeps'] += 1

        mss = detect_mss(frame, direction=plan['mss'],
                         swing_lookback=swing_lookback,
                         max_bars_after_sweep=max_bars_after_sweep,
                         sweep_time=sweep['sweep_time'])
        if mss is None:
            continue
        stages['reversals'] += 1

        entry_fvg = _first_fvg_after(frame, mss['break_time'], plan['fvg'])
        if entry_fvg is None:
            continue
        stages['entries'] += 1

        stop = _sweep_extreme(df, sweep, side)
        entry = float(entry_fvg.entry)
        risk = abs(entry - stop)
        if risk <= 0:
            continue

        target = _nearest_target(pools, plan['target_side'], entry,
                                 plan['direction'])
        if target is None:
            continue

        reward_risk = abs(target['level'] - entry) / risk
        if reward_risk < min_rr:
            continue
        stages['rr_pass'] += 1

        candidates.append(Candidate(
            session=session,
            direction=plan['direction'],
            swept_pool=pool['name'],
            swept_level=float(pool['level']),
            sweep_time=sweep['sweep_time'],
            mss_time=mss['break_time'],
            entry=entry,
            stop=stop,
            target=float(target['level']),
            target_pool=target['name'],
            reward_risk=reward_risk,
        ))

    return SessionScan(candidates=candidates, stages=stages)


def _first_fvg_after(df: pd.DataFrame, after: pd.Timestamp, direction: str):
    """The first imbalance printed at or after the structure shift."""
    tail = df[df.index >= after]
    if len(tail) < 3:
        return None
    fvgs = find_fvgs(tail, direction=direction)
    return fvgs[0] if fvgs else None


def _sweep_extreme(df: pd.DataFrame, sweep: dict, side: str) -> float:
    """The stop sits beyond the extreme the sweep actually reached."""
    bar = df.iloc[sweep['sweep_bar']]
    return float(bar['high'] if side == 'high' else bar['low'])


def _nearest_target(pools: list[dict], target_side: str, entry: float,
                    direction: str) -> Optional[dict]:
    """Nearest opposite-side pool that lies ahead of entry in the trade direction.

    Nearest, not furthest: the first structural level in the path is the one the
    trade can realistically reach, so it is the honest denominator for R. A
    further pool would inflate the ratio by pricing in a move the setup has not
    earned.
    """
    ahead = [p for p in pools
             if p['side'] == target_side
             and ((p['level'] < entry) if direction == 'short'
                  else (p['level'] > entry))]
    if not ahead:
        return None
    return (max(ahead, key=lambda p: p['level']) if direction == 'short'
            else min(ahead, key=lambda p: p['level']))
