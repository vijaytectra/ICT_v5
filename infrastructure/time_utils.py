"""
Timezone helpers shared across concepts and models.

All ICT time logic runs in America/New_York (DST-aware). Data is stored UTC.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

# Cache of NY-local calendar dates and minute-of-day per bar index.
#
# Session-scanning models call ny_index() once per session per instrument, and
# each call re-converts the WHOLE frame. On the 806k-bar index group that is a
# tz conversion of 806,715 timestamps repeated thousands of times, and it
# dominated the SMT backtest runtime (37s per simulated year, almost all of it
# here). The parts are a pure function of the index, so they are derived once.
#
# Keyed by id() with the index itself held alongside, because id() is reused
# after garbage collection; comparing identity makes a stale hit impossible.
_NY_PARTS_CACHE: "OrderedDict[int, tuple]" = OrderedDict()
_NY_PARTS_MAX = 32


def localize_like(ts: pd.Timestamp, ref: pd.DatetimeIndex) -> pd.Timestamp:
    """Match ts tz-awareness to ref index (tz-aware or tz-naive)."""
    if ref.tz is not None and ts.tz is None:
        return ts.tz_localize(ref.tz)
    if ref.tz is None and ts.tz is not None:
        return ts.replace(tzinfo=None)
    return ts


def ny_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return df.index converted to America/New_York. Treats tz-naive as UTC."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    return idx.tz_convert('America/New_York')


def ny_parts(index: pd.DatetimeIndex) -> tuple:
    """NY-local ``(dates, minute_of_day)`` arrays for a bar index, memoised.

    Same values ``ny_index`` would give, as plain numpy arrays and computed once
    per index object. Use this instead of ``ny_index`` in any loop that runs per
    session — it is DST-aware in exactly the same way, since the conversion
    itself is unchanged.
    """
    key = id(index)
    hit = _NY_PARTS_CACHE.get(key)
    if hit is not None and hit[0] is index:
        _NY_PARTS_CACHE.move_to_end(key)
        return hit[1], hit[2]

    ny = index.tz_localize('UTC') if index.tz is None else index
    ny = ny.tz_convert('America/New_York')
    dates = np.asarray(ny.date)
    minutes = np.asarray(ny.hour) * 60 + np.asarray(ny.minute)

    _NY_PARTS_CACHE[key] = (index, dates, minutes)
    _NY_PARTS_CACHE.move_to_end(key)
    while len(_NY_PARTS_CACHE) > _NY_PARTS_MAX:
        _NY_PARTS_CACHE.popitem(last=False)
    return dates, minutes


def window_positions(index: pd.DatetimeIndex, session_date, start_ny: str,
                     end_ny: str) -> np.ndarray:
    """Integer positions of bars on `session_date` inside a NY-local window."""
    dates, minutes = ny_parts(index)
    sh, sm = map(int, start_ny.split(':'))
    eh, em = map(int, end_ny.split(':'))
    d = pd.Timestamp(session_date).date()
    return np.flatnonzero((dates == d) & (minutes >= sh * 60 + sm)
                          & (minutes < eh * 60 + em))


def before_positions(index: pd.DatetimeIndex, session_date,
                     end_ny: str) -> np.ndarray:
    """Integer positions of bars on `session_date` before a NY-local time."""
    dates, minutes = ny_parts(index)
    eh, em = map(int, end_ny.split(':'))
    d = pd.Timestamp(session_date).date()
    return np.flatnonzero((dates == d) & (minutes < eh * 60 + em))
