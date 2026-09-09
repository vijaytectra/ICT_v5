"""Leakage tripwire - does this model still earn when the market is removed?

The reference repo that claimed the highest win rate in our audit shipped saved
metrics of Sharpe 41 and Calmar 9,576. Numbers like that are not an edge, they
are a model that can see the bar it is trading. The defence is cheap and it has
to be automatic, because the run that most needs checking is the one that just
produced a number you liked.

Method: re-run the model against price series whose *bar order* has been
shuffled. Each bar keeps its own geometry - its gap, range, and body - so the
distribution of returns and the volatility are unchanged, and a strategy that
trades every bar unconditionally scores exactly the same. What shuffling
destroys is the multi-bar structure: the gaps, order blocks, sweeps and shifts
a model claims to read. Two things then follow:

* if the model earns on shuffled bars, the edge is coming from the engine -
  a detector reading forward, an optimistic fill, a same-bar entry and exit;
* if the model's real result sits inside the shuffled distribution, there is no
  leakage but also no demonstrated edge.

Only the first is a bug. Both are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

PRICE_COLS = ('open', 'high', 'low', 'close')


def shuffle_bars(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Reorder whole bars, preserving each bar's geometry and the timestamps.

    Every bar is reduced to four multiplicative factors - its gap from the
    previous close, and its high, low and close relative to its own open. The
    order of those units is permuted and the series is re-chained from the
    original opening price. Because each unit came from a real bar, the result
    is coherent by construction: ``high >= max(open, close)`` and
    ``low <= min(open, close)`` hold without clamping.

    The index is left untouched, so session, killzone and day-of-week logic
    still sees the same calendar.
    """
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"shuffle_bars needs columns {missing}")
    if len(df) == 0:
        return df.copy()

    o = df['open'].to_numpy(dtype=float)
    h = df['high'].to_numpy(dtype=float)
    l = df['low'].to_numpy(dtype=float)
    c = df['close'].to_numpy(dtype=float)

    gap = np.ones(len(df))
    gap[1:] = o[1:] / c[:-1]
    hf, lf, cf = h / o, l / o, c / o

    order = np.random.default_rng(seed).permutation(len(df))

    out = df.iloc[order].copy()
    prev_close = o[0]
    new_o = np.empty(len(df))
    new_h = np.empty(len(df))
    new_l = np.empty(len(df))
    new_c = np.empty(len(df))
    for k, src in enumerate(order):
        new_o[k] = prev_close * gap[src]
        new_h[k] = new_o[k] * hf[src]
        new_l[k] = new_o[k] * lf[src]
        new_c[k] = new_o[k] * cf[src]
        prev_close = new_c[k]

    out['open'], out['high'] = new_o, new_h
    out['low'], out['close'] = new_l, new_c
    out.index = df.index
    return out


@dataclass(frozen=True)
class TripwireResult:
    observed_expectancy: float
    null_expectancies: list[float]
    null_mean: float
    null_std: float
    percentile: float
    leakage_suspected: bool
    edge_significant: bool
    passed: bool
    detail: str

    def describe(self) -> str:
        return (f"{'PASS' if self.passed else 'FAIL'} - {self.detail}\n"
                f"  observed   {self.observed_expectancy:+.4f} R\n"
                f"  null       {self.null_mean:+.4f} +/- {self.null_std:.4f} R "
                f"over {len(self.null_expectancies)} shuffles\n"
                f"  percentile {self.percentile:.1f}")


def _expectancy(trades: pd.DataFrame) -> float:
    if trades is None or len(trades) == 0 or 'pnl_r' not in trades:
        return 0.0
    return float(trades['pnl_r'].astype(float).mean())


def run_tripwire(run_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 df: pd.DataFrame,
                 n_runs: int = 50,
                 seed: int = 0,
                 leakage_tolerance_r: float = 0.05,
                 significance_percentile: float = 95.0) -> TripwireResult:
    """Run ``run_fn`` once on ``df`` and ``n_runs`` times on shuffled copies.

    ``run_fn`` takes a bar frame and returns a trade log with a ``pnl_r``
    column. It is called once with the real frame first, so a model that caches
    on first call still sees the real data in the position it expects.

    Leakage is called when the mean shuffled expectancy exceeds both
    ``leakage_tolerance_r`` and twice its own standard error - the second term
    stops a small null sample from raising a false alarm on noise alone.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    observed = _expectancy(run_fn(df))

    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_runs)
    nulls = [_expectancy(run_fn(shuffle_bars(df, seed=int(s)))) for s in seeds]

    arr = np.asarray(nulls, dtype=float)
    null_mean = float(arr.mean())
    null_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sem = null_std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    leakage = bool(null_mean > max(leakage_tolerance_r, 2.0 * sem))
    threshold = float(np.percentile(arr, significance_percentile))
    significant = bool(observed > threshold)
    percentile = float((arr < observed).mean() * 100.0)

    if leakage:
        detail = (f"leakage suspected - shuffled bars still earn "
                  f"{null_mean:+.4f} R/trade; the edge is in the engine, "
                  f"not the market")
    elif not significant:
        detail = (f"no leakage, but the real result {observed:+.4f} R does not "
                  f"clear the {significance_percentile:.0f}th percentile of the "
                  f"null ({threshold:+.4f} R) - no demonstrated edge")
    else:
        detail = (f"clean - shuffled bars earn {null_mean:+.4f} R/trade while "
                  f"the real series earns {observed:+.4f} R")

    return TripwireResult(
        observed_expectancy=observed,
        null_expectancies=nulls,
        null_mean=null_mean,
        null_std=null_std,
        percentile=percentile,
        leakage_suspected=leakage,
        edge_significant=significant,
        passed=(not leakage) and significant,
        detail=detail,
    )
