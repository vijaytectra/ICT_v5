"""Walk-forward validation.

Model-agnostic. Consumes per-configuration trade logs and answers one question:
what would this selection process have earned in real time?

The engine never sees a model. It sees trade logs keyed by configuration name,
each carrying a `date` column and a `pnl_r` column. Everything here is pure -
no file I/O, no global state - so it can be tested on synthetic logs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

DATE_COL = 'date'
R_COL = 'pnl_r'


@dataclass(frozen=True)
class Window:
    """One train/test split. Both bounds are inclusive.

    ``train_end`` and ``test_end`` come from ``Period.end_time``: the last
    nanosecond of the quarter (e.g. ``2025-03-31 23:59:59.999999999``), not
    midnight of the last day. A caller filtering rows with
    ``df[DATE_COL] <= window.train_end`` gets the full last day included, as
    intended - but comparing against a bare date/midnight timestamp instead
    would silently drop that final day.
    """
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(first_date, last_date, min_train_quarters: int = 4,
                 test_quarters: int = 1) -> list[Window]:
    """Expanding-train, fixed-length-test quarterly windows.

    Training always starts at the first quarter and grows; only the test period
    slides. That mirrors re-tuning on the whole history you actually have.
    """
    if min_train_quarters < 1:
        raise ValueError(
            f"min_train_quarters must be >= 1, got {min_train_quarters}")
    if test_quarters < 1:
        raise ValueError(
            f"test_quarters must be >= 1, got {test_quarters}")

    quarters = pd.period_range(pd.Timestamp(first_date),
                               pd.Timestamp(last_date), freq='Q')
    windows: list[Window] = []
    for i in range(min_train_quarters, len(quarters), test_quarters):
        test_qs = quarters[i:i + test_quarters]
        if len(test_qs) < test_quarters:
            break
        windows.append(Window(
            index=len(windows),
            train_start=quarters[0].start_time,
            train_end=quarters[i - 1].end_time,
            test_start=test_qs[0].start_time,
            test_end=test_qs[-1].end_time,
        ))
    return windows


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical types for every column `log_fingerprint` uses.

    - `date` and `entry_time` (when present) become naive Timestamps.
    - `instrument` and `direction` (when present) are upper-cased and stripped,
      so 'eurusd' and 'EURUSD' canonicalise to the same value.
    - `pnl_r` becomes numeric. A value that fails to parse and was not already
      NaN in the input raises ValueError naming the count and offending row
      indices - it is coerced, never silently dropped. A value that was
      already NaN in the input (a genuinely missing R) passes through as NaN
      without raising, since that's a data property, not a parse failure. A
      non-finite value (+inf/-inf) always raises, whether or not it was
      already infinite in the input - unlike a missing R, an infinite R is
      never a legitimate data state, and letting it through would corrupt any
      mean computed over the column (a single +inf/-inf silently dominates or
      NaNs out an expectancy).
    - Rows are sorted ascending by `date`.

    Logs are written by several runners; this is the one place that decides what
    a trade log looks like to the engine.
    """
    missing = [c for c in (DATE_COL, R_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"trade log missing required columns: {missing}")
    out = df.copy()

    dates = pd.to_datetime(out[DATE_COL])
    if getattr(dates.dt, 'tz', None) is not None:
        dates = dates.dt.tz_localize(None)
    out[DATE_COL] = dates

    if 'entry_time' in out.columns:
        entry_times = pd.to_datetime(out['entry_time'])
        if getattr(entry_times.dt, 'tz', None) is not None:
            entry_times = entry_times.dt.tz_localize(None)
        out['entry_time'] = entry_times

    for col in ('instrument', 'direction'):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.upper()

    raw_r = out[R_COL]
    coerced_r = pd.to_numeric(raw_r, errors='coerce')
    newly_nan = coerced_r.isna() & ~raw_r.isna()
    non_finite = coerced_r.notna() & ~np.isfinite(coerced_r)
    invalid = newly_nan | non_finite
    if invalid.any():
        bad_idx = list(out.index[invalid])
        shown = bad_idx[:5]
        suffix = ', ...' if len(bad_idx) > 5 else ''
        raise ValueError(
            f"{R_COL} has {len(bad_idx)} unparseable or non-finite value(s) "
            f"at row index {shown}{suffix}")
    out[R_COL] = coerced_r

    return out.sort_values(DATE_COL).reset_index(drop=True)


def log_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a log's trade identity, for collapsing duplicates.

    Valid only within a single process/run: the hash is taken over pandas'
    `to_csv()` serialisation of the normalized frame, and pandas gives no
    cross-version guarantee for float/timestamp text formatting. Never persist
    a fingerprint to disk or compare fingerprints computed under different
    pandas versions.
    """
    norm = normalize_log(df)
    cols = [c for c in (DATE_COL, 'instrument', 'direction', 'entry_time', R_COL)
            if c in norm.columns]
    payload = norm[cols].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def dedupe_logs(logs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame],
                                                        dict[str, str]]:
    """Collapse configurations whose trade logs are identical.

    Returns (kept, dropped) where `dropped` maps each removed name to the name
    it duplicates. Names are processed in sorted order so the survivor is
    deterministic.
    """
    kept: dict[str, pd.DataFrame] = {}
    seen: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for name in sorted(logs):
        fp = log_fingerprint(logs[name])
        if fp in seen:
            dropped[name] = seen[fp]
        else:
            seen[fp] = name
            kept[name] = logs[name]
    return kept, dropped


class ConfigStat(NamedTuple):
    """One configuration's training-window trade count and expectancy.

    Named fields so the ranking key in `select_config` reads as intent
    (`.expectancy`, `.n_trades`) instead of an unlabelled `kv[1][0]`/`kv[1][1]`
    that a future edit could transpose without any type error to catch it.
    """
    n_trades: int
    expectancy: float


@dataclass(frozen=True)
class Selection:
    """What one training window chose, and on what evidence."""
    window: Window
    config: str
    train_trades: int
    train_expectancy: float
    n_eligible: int
    floor_used: int
    floor_relaxed: bool


def slice_dates(df: pd.DataFrame, start: pd.Timestamp,
                end: pd.Timestamp) -> pd.DataFrame:
    """Rows whose date falls in [start, end]. Assumes a normalised log."""
    mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    return df.loc[mask]


def select_config(logs: dict[str, pd.DataFrame], window: Window,
                  min_trades: int = 40,
                  fallback_min_trades: int = 25) -> Selection:
    """Pick the best configuration using ONLY the window's training period.

    Ranked by expectancy (mean R per trade) among configurations clearing the
    trade floor. The floor exists because expectancy on a handful of trades is
    noise, and an unfloored search reliably selects that noise.

    If nothing clears `min_trades` the floor drops to `fallback_min_trades` and
    the Selection records that it happened. If nothing clears that either, this
    raises rather than quietly returning a fluke.
    """
    if min_trades < 1:
        raise ValueError(f"min_trades must be >= 1, got {min_trades}")
    if fallback_min_trades < 1:
        raise ValueError(
            f"fallback_min_trades must be >= 1, got {fallback_min_trades}")
    if fallback_min_trades > min_trades:
        raise ValueError(
            f"fallback_min_trades ({fallback_min_trades}) must be <= "
            f"min_trades ({min_trades}) - the fallback is a more lenient "
            f"floor, so it can never exceed the primary one")

    stats: dict[str, ConfigStat] = {}
    for name, log in logs.items():
        train = slice_dates(normalize_log(log), window.train_start, window.train_end)
        # Drop unresolved trades ONCE, before both the count and the mean, so the
        # eligibility floor and the ranking share a denominator by construction.
        # `normalize_log` guarantees no silently-introduced NaN, but not "no NaN":
        # a genuinely missing R value is legitimate. Without this line `len()`
        # would count such a row toward the floor while `.mean()` skips it, which
        # silently tilts which configuration wins.
        train = train.dropna(subset=[R_COL])
        if len(train):
            stats[name] = ConfigStat(n_trades=len(train),
                                     expectancy=float(train[R_COL].mean()))

    for floor in (min_trades, fallback_min_trades):
        eligible = {n: v for n, v in stats.items() if v.n_trades >= floor}
        if eligible:
            # expectancy desc, then trade count desc, then name asc
            best = min(eligible.items(),
                       key=lambda kv: (-kv[1].expectancy, -kv[1].n_trades, kv[0]))
            name, stat = best
            return Selection(
                window=window, config=name, train_trades=stat.n_trades,
                train_expectancy=stat.expectancy, n_eligible=len(eligible),
                floor_used=floor, floor_relaxed=(floor != min_trades),
            )

    if stats:
        best_name, best_stat = max(stats.items(), key=lambda kv: kv[1].n_trades)
        detail = f"best was '{best_name}' with {best_stat.n_trades} training trades"
    else:
        detail = "no configuration had any training trades"

    raise ValueError(
        f"window {window.index}: no configuration reached {fallback_min_trades} "
        f"training trades ({detail})"
    )
