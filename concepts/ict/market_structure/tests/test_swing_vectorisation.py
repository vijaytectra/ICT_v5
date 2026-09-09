"""identify_swings is vectorised; pin it against the definition it replaced.

The scanner used to walk every bar in Python and assign with .loc, which made it
90% of backtest runtime. It is now two rolling-window comparisons. The reference
below is the original loop transcribed verbatim - if the fast path ever drifts
from the fractal definition, these fail.
"""
import numpy as np
import pandas as pd
import pytest

from concepts.ict.market_structure.market_structure import SwingPointScanner

COLS = ('swing_high', 'swing_low', 'swing_high_price', 'swing_low_price')


def reference_swings(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
    """The pre-vectorisation loop, kept verbatim as the specification."""
    df = df_in.copy()
    for c in COLS:
        df[c] = np.nan
    if len(df) < (2 * n + 1):
        return df
    for i in range(n, len(df) - n):
        ch = df['high'].iloc[i]
        if (all(df['high'].iloc[i - j] < ch for j in range(1, n + 1)) and
                all(df['high'].iloc[i + j] < ch for j in range(1, n + 1))):
            df.loc[df.index[i], 'swing_high'] = 1
            df.loc[df.index[i], 'swing_high_price'] = ch
    for i in range(n, len(df) - n):
        cl = df['low'].iloc[i]
        if (all(df['low'].iloc[i - j] > cl for j in range(1, n + 1)) and
                all(df['low'].iloc[i + j] > cl for j in range(1, n + 1))):
            df.loc[df.index[i], 'swing_low'] = 1
            df.loc[df.index[i], 'swing_low_price'] = cl
    return df


def _frame(n_bars: int, seed: int, *, nan_count: int = 0, round_dp: int | None = None):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n_bars))
    high = close + np.abs(rng.normal(0, 0.4, n_bars))
    low = close - np.abs(rng.normal(0, 0.4, n_bars))
    df = pd.DataFrame(
        {'open': close, 'high': high, 'low': low, 'close': close,
         'volume': rng.integers(1, 500, n_bars)},
        index=pd.date_range('2024-01-01', periods=n_bars, freq='2min', tz='UTC'),
    )
    if round_dp is not None:          # force ties: strict inequality must hold
        df['high'] = df['high'].round(round_dp)
        df['low'] = df['low'].round(round_dp)
    if nan_count:                     # gaps: a NaN neighbour disqualifies the bar
        for col in ('high', 'low'):
            df.iloc[rng.choice(n_bars, nan_count, replace=False),
                    df.columns.get_loc(col)] = np.nan
    return df


@pytest.mark.parametrize('lookback', [1, 3, 5, 10])
@pytest.mark.parametrize('kwargs', [
    {},
    {'nan_count': 40},
    {'round_dp': 1},
], ids=['clean', 'with-gaps', 'with-ties'])
def test_matches_the_loop_it_replaced(lookback, kwargs):
    df = _frame(600, seed=7, **kwargs)
    fast = SwingPointScanner(df, lookback=lookback).identify_swings()
    slow = reference_swings(df, lookback)
    for c in COLS:
        np.testing.assert_array_equal(
            fast[c].to_numpy(dtype=float), slow[c].to_numpy(dtype=float),
            err_msg=f"{c} differs at lookback={lookback}")


@pytest.mark.parametrize('lookback', [1, 5])
def test_frame_shorter_than_one_window_yields_no_swings(lookback):
    df = _frame(2 * lookback, seed=1)
    out = SwingPointScanner(df, lookback=lookback).identify_swings()
    for c in COLS:
        assert out[c].isna().all()


def test_edges_are_never_swings():
    """The loop only considered range(n, len - n); the fast path must too."""
    df = _frame(200, seed=3)
    n = 4
    out = SwingPointScanner(df, lookback=n).identify_swings()
    assert out['swing_high'].iloc[:n].isna().all()
    assert out['swing_high'].iloc[-n:].isna().all()
    assert out['swing_low'].iloc[:n].isna().all()
    assert out['swing_low'].iloc[-n:].isna().all()
