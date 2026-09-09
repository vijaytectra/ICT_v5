"""
SMT Divergence (Smart Money Technique)
=======================================

A crack in correlation between correlated instruments: one makes a new swing
extreme while the other(s) fail to confirm.

Two APIs live here.

``find_smt_events`` is the real detector. It is **window-anchored** and
**timestamp-aligned**: it finds relative swing extremes inside one of ICT's
stated comparison windows, then reads *every* instrument at those same
timestamps. That same-time-of-day comparison is criterion c2 of the knowledge
base's ``correlated-pairs-smt`` page and its most-cited failure mode ("don't
compare 09:00 EURUSD with 10:00 GBPUSD"). It is only sound on a shared bar
index, which ``DataLoader.read_smt_group`` guarantees.

``detect_smt`` is the original rolling-window detector, kept **unchanged** and
**deprecated** because ``models/ict/sniper_model`` is built on it. It compares
``high.iloc[-3:].max()`` against ``high.iloc[-6:-3].max()``, which has no swing
anchor, no liquidity pool, no cross-instrument time alignment and no expiry, so
it reports a divergence at nearly every bar. Do not use it in new work.

See smt-divergence.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from infrastructure.registry import concept
from infrastructure.time_utils import localize_like, window_positions


@dataclass
class SMTEvent:
    """One crack in correlation, anchored to two comparable pivot times.

    Attributes:
        time:         Signal bar - the second (later) pivot's timestamp.
        side:         'low'  - lows were compared (a bullish reading)
                      'high' - highs were compared (a bearish reading)
        leader:       Instruments that DID make the new extreme (were swept).
        diverging:    Instruments that FAILED to confirm it.
        pivot_times:  (t1, t2) - the two comparable pivots.
        levels:       symbol -> (v1, v2), the extreme each instrument printed
                      at t1 and t2 in its own quote units.
        clarity:      How decisively the diverging side refused, in scale units
                      (see `scale`). The *smallest* refusal across diverging
                      instruments, so one lukewarm member cannot inflate it.
        sweep_depth:  How far past its prior level the leader ran, in scale
                      units. The *largest* across leaders.
        window:       The NY-local comparison window this came from.
        scale:        symbol -> the denominator used to normalise clarity and
                      sweep_depth (ADR when the caller supplies one).
    """
    time: pd.Timestamp
    side: str
    leader: Tuple[str, ...]
    diverging: Tuple[str, ...]
    pivot_times: Tuple[pd.Timestamp, pd.Timestamp]
    levels: Dict[str, Tuple[float, float]]
    clarity: float
    sweep_depth: float
    window: Tuple[str, str]
    scale: Dict[str, float] = field(default_factory=dict)
    # The bar on which this event first became KNOWABLE. `time` is the pivot
    # bar itself, and a pivot needs `swing_lookback` bars after it before it is
    # a pivot at all - plus `match_tolerance_bars` if the comparison read a
    # neighbourhood. Acting on `time` is lookahead: it buys the exact low with
    # hindsight. Measured on a strawman arm that did exactly that, 75/75
    # signals were armed 0 minutes after a pivot that needed 10 minutes to
    # confirm, and it turned a losing rule into Sharpe 3.1.
    confirmed_time: Optional[pd.Timestamp] = None
    correlation: Optional[float] = None

    @property
    def direction(self) -> str:
        """Trade direction this reading argues for."""
        return 'long' if self.side == 'low' else 'short'

    def __repr__(self) -> str:
        return (f"SMTEvent({self.time:%Y-%m-%d %H:%M} {self.side} "
                f"leader={'/'.join(self.leader)} div={'/'.join(self.diverging)} "
                f"clarity={self.clarity:.3f} sweep={self.sweep_depth:.3f})")


def _window_positions(df: pd.DataFrame, session_date: pd.Timestamp,
                      window: Tuple[str, str]) -> np.ndarray:
    """Integer positions of bars on `session_date` inside a NY-local window.

    DST-aware - never assume a fixed UTC offset, or the whole comparison window
    slides by an hour twice a year. Backed by the memoised ``ny_parts`` so a
    per-session loop does not re-convert the entire frame each time.
    """
    return window_positions(df.index, session_date, window[0], window[1])


def _pivots(values: np.ndarray, lookback: int, low: bool) -> np.ndarray:
    """Positions of local extrema with `lookback` strictly worse bars each side."""
    n = len(values)
    mask = np.zeros(n, bool)
    if n < 2 * lookback + 1:
        return np.flatnonzero(mask)
    core = values[lookback:n - lookback]
    ok = np.ones(len(core), bool)
    for j in range(1, lookback + 1):
        left = values[lookback - j:n - lookback - j]
        right = values[lookback + j:n - lookback + j]
        ok &= ((core < left) & (core < right)) if low else ((core > left) & (core > right))
    mask[lookback:n - lookback] = ok
    return np.flatnonzero(mask)


def realised_correlation(
    correlated: Dict[str, pd.DataFrame],
    end_pos: int,
    lookback: int,
    inverse: Sequence[str] = (),
) -> float:
    """Weakest pairwise log-return correlation in the group, as of `end_pos`.

    SMT assumes the instruments are correlated *right now*. When they are not,
    a "divergence" is just two markets doing unrelated things, and the signal
    is noise dressed as information. Measured on this repo's index group, the
    quartile with the LARGEST divergence produced the worst outcomes and the
    highest adverse excursion - the signature of decorrelation rather than an
    engineered sweep.

    Inverse members are sign-flipped before comparison, so a healthy
    EURUSD/DXY pairing scores near +1 rather than near -1, and one number can
    gate every group.

    Uses only bars up to and including `end_pos`. The minimum across pairs is
    returned so one decoupled member cannot hide behind two aligned ones.
    """
    symbols = list(correlated)
    if len(symbols) < 2 or lookback < 8:
        return float('nan')
    start = max(end_pos - lookback + 1, 0)
    if end_pos - start < 8:
        return float('nan')

    rets = {}
    for s in symbols:
        c = correlated[s]['close'].to_numpy()[start:end_pos + 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.diff(np.log(np.where(c > 0, c, np.nan)))
        if s in inverse:
            r = -r
        rets[s] = r

    worst = np.inf
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = rets[symbols[i]], rets[symbols[j]]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 8:
                continue
            sa, sb = a[m], b[m]
            if sa.std() == 0 or sb.std() == 0:
                continue
            worst = min(worst, float(np.corrcoef(sa, sb)[0, 1]))
    return worst if np.isfinite(worst) else float('nan')


def _effective_side(symbol: str, side: str, inverse: Sequence[str]) -> str:
    """Which of the instrument's own extremes corresponds to the group's `side`.

    An inversely correlated member (DXY against the FX majors) confirms a group
    LOW by making its own HIGH. Comparing its low there would invert the signal
    and report every tandem move as a divergence.
    """
    if symbol not in inverse:
        return side
    return 'high' if side == 'low' else 'low'


@concept("smt-divergence", depends_on=["swing-points"])
def find_smt_events(
    correlated: Dict[str, pd.DataFrame],
    session_date: pd.Timestamp,
    window: Tuple[str, str],
    side: str,
    scale: Optional[Dict[str, float]] = None,
    inverse: Sequence[str] = (),
    swing_lookback: int = 2,
    pivot_dedupe_bars: int = 2,
    match_tolerance_bars: int = 1,
    min_window_bars: int = 12,
    all_pairs: bool = False,
    corr_lookback: int = 288,
    min_abs_corr: float = 0.0,
) -> List[SMTEvent]:
    """Find SMT divergences inside one NY-local comparison window.

    The method follows the 2017/2022 index teaching literally: bracket a window
    (05:00-09:30 or 12:00-15:00 NY), find the *relative* highs and lows that
    form inside it - "don't think like they're goalposts", the extremes need not
    sit at the window edges - and compare those same moments across every
    instrument.

    Pivots are collected from the union of all instruments and deduped, so a
    swing that only one index printed still becomes a comparison point. Each
    instrument is then read in a +/-`match_tolerance_bars` neighbourhood of both
    pivot times, which absorbs the one-bar timing jitter between correlated
    markets without letting the comparison drift to a different swing.

    Args:
        correlated:           symbol -> OHLCV, **sharing one bar index**. Use
                              ``DataLoader.read_smt_group``.
        session_date:         Calendar date of the session (NY local).
        window:               ('HH:MM','HH:MM') NY-local comparison window.
        side:                 'low' compares lows (bullish reading);
                              'high' compares highs (bearish reading).
        scale:                symbol -> denominator for clarity / sweep_depth,
                              normally the instrument's ADR. Falls back to each
                              instrument's own window range, which keeps the
                              numbers comparable but not ADR-anchored.
        inverse:              Symbols that move against the group (e.g. DXY).
        swing_lookback:       Bars each side required to confirm a pivot.
        pivot_dedupe_bars:    Pivots closer than this are treated as one moment.
        match_tolerance_bars: Neighbourhood read around each pivot time.
        min_window_bars:      Windows shorter than this are skipped (half days).
        all_pairs:            False returns only the final comparison (the most
                              recent pivot pair), which is what a live trader
                              sees. True returns every consecutive pivot pair in
                              the window - used by research scripts.

    Returns:
        Chronological list of SMTEvent. Empty when the instruments moved in
        tandem, when there are too few pivots, or when the window is too short.
    """
    if side not in ('low', 'high'):
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")
    symbols = list(correlated)
    if len(symbols) < 2:
        return []

    ref = correlated[symbols[0]]
    for s in symbols[1:]:
        if not correlated[s].index.equals(ref.index):
            raise ValueError(
                "SMT requires a shared bar index across instruments - "
                f"{s!r} does not match {symbols[0]!r}. Load the group with "
                "DataLoader.read_smt_group, which intersects the indices."
            )

    pos = _window_positions(ref, session_date, window)
    if len(pos) < min_window_bars:
        return []
    times = ref.index[pos]

    # Series each instrument is compared on, honouring inverse correlation.
    eff = {s: _effective_side(s, side, inverse) for s in symbols}
    series = {s: (correlated[s]['low'] if eff[s] == 'low' else correlated[s]['high'])
              .to_numpy()[pos] for s in symbols}

    denom = {}
    for s in symbols:
        if scale and s in scale and scale[s] and np.isfinite(scale[s]) and scale[s] > 0:
            denom[s] = float(scale[s])
        else:
            w = correlated[s].iloc[pos]
            rng = float(w['high'].max() - w['low'].min())
            denom[s] = rng if rng > 0 else np.nan

    # Union of pivots across instruments, deduped into single "moments".
    found = set()
    for s in symbols:
        found.update(_pivots(series[s], swing_lookback, low=(eff[s] == 'low')).tolist())
    deduped: List[int] = []
    for p in sorted(found):
        if not deduped or p - deduped[-1] > pivot_dedupe_bars:
            deduped.append(p)
    if len(deduped) < 2:
        return []

    pairs = (list(zip(deduped[:-1], deduped[1:])) if all_pairs
             else [(deduped[-2], deduped[-1])])

    tol = match_tolerance_bars
    n = len(pos)
    events: List[SMTEvent] = []
    for i1, i2 in pairs:
        made: Dict[str, bool] = {}
        margin: Dict[str, float] = {}
        levels: Dict[str, Tuple[float, float]] = {}
        usable = True
        for s in symbols:
            a = series[s]
            w1 = a[max(i1 - tol, 0):min(i1 + tol + 1, n)]
            w2 = a[max(i2 - tol, 0):min(i2 + tol + 1, n)]
            if len(w1) == 0 or len(w2) == 0:
                usable = False
                break
            if eff[s] == 'low':
                v1, v2 = float(w1.min()), float(w2.min())
                made[s] = v2 < v1
                signed = v1 - v2           # >0 when a new low was made
            else:
                v1, v2 = float(w1.max()), float(w2.max())
                made[s] = v2 > v1
                signed = v2 - v1           # >0 when a new high was made
            levels[s] = (v1, v2)
            d = denom[s]
            margin[s] = signed / d if d and np.isfinite(d) and d > 0 else np.nan
        if not usable:
            continue

        n_made = sum(made.values())
        # 0 or all = the instruments moved in tandem; there is no crack.
        if n_made == 0 or n_made == len(symbols):
            continue

        leaders = tuple(s for s in symbols if made[s])
        diverging = tuple(s for s in symbols if not made[s])
        # A diverging instrument's margin is negative (it fell short). Clarity is
        # the least decisive refusal, so a single lukewarm member caps it.
        refusals = [-margin[s] for s in diverging if np.isfinite(margin[s])]
        depths = [margin[s] for s in leaders if np.isfinite(margin[s])]
        # Earliest bar on which a live trader could have known this event
        # existed: the pivot needs `swing_lookback` bars to its right, and the
        # comparison read up to `match_tolerance_bars` past it.
        lag = max(swing_lookback, match_tolerance_bars)
        conf_pos = min(i2 + lag, n - 1)

        # Correlation is measured on the full series up to the confirmation
        # bar, not on the handful of bars inside the window - a 12-bar window
        # cannot estimate a correlation, and using window bars only would also
        # make the number depend on the session length.
        corr = realised_correlation(correlated, int(pos[conf_pos]),
                                    corr_lookback, inverse)
        if min_abs_corr > 0 and (not np.isfinite(corr) or corr < min_abs_corr):
            continue
        events.append(SMTEvent(
            time=times[i2],
            side=side,
            leader=leaders,
            diverging=diverging,
            pivot_times=(times[i1], times[i2]),
            levels=levels,
            clarity=float(min(refusals)) if refusals else float('nan'),
            sweep_depth=float(max(depths)) if depths else float('nan'),
            window=window,
            scale=denom,
            confirmed_time=times[conf_pos],
            correlation=corr,
        ))
    return events


@concept("smt-divergence-breach", depends_on=["smt-divergence"])
def find_smt_breach_events(
    correlated: Dict[str, pd.DataFrame],
    session_date: pd.Timestamp,
    window: Tuple[str, str],
    side: str,
    scale: Optional[Dict[str, float]] = None,
    inverse: Sequence[str] = (),
    min_pool_bars: int = 6,
    pool_gap_bars: int = 1,
    min_window_bars: int = 12,
    max_events: int = 1,
    corr_lookback: int = 288,
    min_abs_corr: float = 0.0,
) -> List[SMTEvent]:
    """SMT detected at the moment of the breach, not from two confirmed pivots.

    ``find_smt_events`` compares two swing pivots. A pivot does not exist until
    ``swing_lookback`` bars have printed to its right, so that reading is late
    by construction and can only ever support a retrace entry.

    ICT describes something different and strictly causal: a level forms early
    in the session, and the signal is price **running through it now** while the
    correlated instruments refuse to follow — "that run under the 5 a.m. low on
    NASDAQ, that's a stop run" (M10-AM [14:54]). The pool is history; the breach
    is live.

    At bar ``t`` the pool for each instrument is the running extreme of the
    window over ``[window_start, t - pool_gap_bars]``, which uses only closed
    bars. A breach is ``low[t] < pool`` (mirrored for highs, and for inverse
    members). Mixed breach/no-breach across the group is the divergence.

    Because nothing here needs a bar from the future, ``confirmed_time`` equals
    ``time``: the event is actionable on the bar it prints.

    Args:
        min_pool_bars:  Bars required before a pool is considered established.
        pool_gap_bars:  Bars excluded between the pool window and the test bar,
                        so the bar being tested cannot define its own pool.
        max_events:     Events to return per window. 1 = the first raid only,
                        which is the one a session-anchored trader takes.

    Returns:
        Chronological list of SMTEvent, at most ``max_events`` long.
    """
    if side not in ('low', 'high'):
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")
    symbols = list(correlated)
    if len(symbols) < 2:
        return []

    ref = correlated[symbols[0]]
    for s in symbols[1:]:
        if not correlated[s].index.equals(ref.index):
            raise ValueError(
                "SMT requires a shared bar index across instruments - "
                f"{s!r} does not match {symbols[0]!r}."
            )

    pos = _window_positions(ref, session_date, window)
    if len(pos) < min_window_bars:
        return []
    times = ref.index[pos]

    eff = {s: _effective_side(s, side, inverse) for s in symbols}
    series = {s: (correlated[s]['low'] if eff[s] == 'low' else correlated[s]['high'])
              .to_numpy()[pos] for s in symbols}

    denom = {}
    for s in symbols:
        if scale and s in scale and scale[s] and np.isfinite(scale[s]) and scale[s] > 0:
            denom[s] = float(scale[s])
        else:
            w = correlated[s].iloc[pos]
            rng = float(w['high'].max() - w['low'].min())
            denom[s] = rng if rng > 0 else np.nan

    # Running pool per instrument, using only bars strictly before the test bar.
    pools = {}
    for s in symbols:
        a = series[s]
        pools[s] = (np.minimum.accumulate(a) if eff[s] == 'low'
                    else np.maximum.accumulate(a))

    events: List[SMTEvent] = []
    n = len(pos)
    start = max(min_pool_bars, pool_gap_bars + 1)
    for t in range(start, n):
        p = t - pool_gap_bars - 1
        if p < 0:
            continue
        breached, margin, levels = {}, {}, {}
        for s in symbols:
            a = series[s]
            pool = float(pools[s][p])
            now = float(a[t])
            if eff[s] == 'low':
                breached[s] = now < pool
                signed = pool - now
            else:
                breached[s] = now > pool
                signed = now - pool
            levels[s] = (pool, now)
            d = denom[s]
            margin[s] = signed / d if d and np.isfinite(d) and d > 0 else np.nan

        k = sum(breached.values())
        if k == 0 or k == len(symbols):
            continue

        corr = realised_correlation(correlated, int(pos[t]), corr_lookback, inverse)
        if min_abs_corr > 0 and (not np.isfinite(corr) or corr < min_abs_corr):
            continue

        leaders = tuple(s for s in symbols if breached[s])
        diverging = tuple(s for s in symbols if not breached[s])
        refusals = [-margin[s] for s in diverging if np.isfinite(margin[s])]
        depths = [margin[s] for s in leaders if np.isfinite(margin[s])]
        events.append(SMTEvent(
            time=times[t],
            side=side,
            leader=leaders,
            diverging=diverging,
            pivot_times=(times[p], times[t]),
            levels=levels,
            clarity=float(min(refusals)) if refusals else float('nan'),
            sweep_depth=float(max(depths)) if depths else float('nan'),
            window=window,
            scale=denom,
            # Nothing from the future was used, so the event is actionable on
            # the bar it prints.
            confirmed_time=times[t],
            correlation=corr,
        ))
        if len(events) >= max_events:
            break
    return events


def negated_at(
    correlated: Dict[str, pd.DataFrame],
    event: SMTEvent,
    until: Optional[pd.Timestamp] = None,
    inverse: Sequence[str] = (),
) -> Optional[pd.Timestamp]:
    """First time the divergence closes, or None while it still stands.

    An SMT dies when a diverging instrument goes on to take **its own** prior
    compared level - the level it refused. The knowledge base's ``smt-failure``
    formula compares asset B's low against asset A's new low, which cannot be
    right across instruments quoted on different scales (ES at 5,000 against NQ
    at 20,000); the repo spec's phrasing, "the non-confirming instrument later
    takes its level anyway", is the one implemented here.

    Args:
        correlated: Same frames the event came from.
        event:      The event to test.
        until:      Stop scanning at this time (typically session end).
        inverse:    Symbols that move against the group.

    Returns:
        Timestamp of the first negating bar, or None.
    """
    hits = []
    for s in event.diverging:
        df = correlated.get(s)
        if df is None or s not in event.levels:
            continue
        v1, _ = event.levels[s]
        after = df[df.index > localize_like(pd.Timestamp(event.time), df.index)]
        if until is not None:
            after = after[after.index <= localize_like(pd.Timestamp(until), after.index)]
        if len(after) == 0:
            continue
        if _effective_side(s, event.side, inverse) == 'low':
            breach = after[after['low'] < v1]
        else:
            breach = after[after['high'] > v1]
        if len(breach):
            hits.append(breach.index[0])
    return min(hits) if hits else None


def smt_is_live(
    correlated: Dict[str, pd.DataFrame],
    event: SMTEvent,
    as_of: pd.Timestamp,
    inverse: Sequence[str] = (),
) -> bool:
    """True while the divergence still stands at `as_of`.

    Used to cancel an unfilled entry: a setup whose primary confluence has
    already closed is not the setup that was signalled.
    """
    t = negated_at(correlated, event, until=as_of, inverse=inverse)
    return t is None


# ---------------------------------------------------------------------------
# Deprecated
# ---------------------------------------------------------------------------

def detect_smt(
    correlated: Dict[str, pd.DataFrame],
    kind: str = 'high',
    swing_window: int = 3,
) -> Optional[dict]:
    """DEPRECATED - rolling-window detector. Use :func:`find_smt_events`.

    Behaviour is preserved byte-for-byte because ``models/ict/sniper_model`` is
    built on it and changing it would silently move that model's results.

    Why it is unsound: it compares the max/min of the last `swing_window` bars
    against the preceding `swing_window` bars. There is no swing anchor, no
    liquidity pool, no comparison window, no cross-instrument time alignment and
    no expiry, so it reports a "divergence" at nearly every bar.
    """
    warnings.warn(
        "detect_smt is deprecated and unsound; use find_smt_events()",
        DeprecationWarning, stacklevel=2,
    )
    if not correlated or len(correlated) < 2:
        return None

    extremes = {}
    prior_extremes = {}

    for name, df in correlated.items():
        if df is None or len(df) < swing_window + 1:
            return None
        if kind == 'high':
            extremes[name] = float(df['high'].iloc[-swing_window:].max())
            prior_extremes[name] = float(df['high'].iloc[-2 * swing_window:-swing_window].max())
        else:
            extremes[name] = float(df['low'].iloc[-swing_window:].min())
            prior_extremes[name] = float(df['low'].iloc[-2 * swing_window:-swing_window].min())

    if kind == 'high':
        leaders = [n for n in extremes if extremes[n] > prior_extremes[n]]
        diverging = [n for n in extremes if extremes[n] < prior_extremes[n]]
    else:
        leaders = [n for n in extremes if extremes[n] < prior_extremes[n]]
        diverging = [n for n in extremes if extremes[n] > prior_extremes[n]]

    if not leaders or not diverging:
        return None

    leader = leaders[0]
    div = diverging[0]

    leader_df = correlated[leader]
    if kind == 'high':
        t = leader_df['high'].iloc[-swing_window:].idxmax()
    else:
        t = leader_df['low'].iloc[-swing_window:].idxmin()

    return {
        'kind': kind,
        'leader': leader,
        'diverging': diverging,
        'leader_extreme': extremes[leader],
        'diverging_extreme': extremes[div],
        'time': t,
    }


def smt_confirmed(
    correlated: Dict[str, pd.DataFrame],
    kind: str = 'high',
    swing_window: int = 3,
) -> bool:
    """DEPRECATED - see :func:`detect_smt`."""
    return detect_smt(correlated, kind, swing_window) is not None
