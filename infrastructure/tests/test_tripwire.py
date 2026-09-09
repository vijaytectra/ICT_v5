"""Leakage tripwire tests. Synthetic bars only - no data files needed.

The tripwire re-runs a model against price series whose *bar order* has been
shuffled. Shuffling whole bars preserves each bar's geometry and the distribution
of returns, so a strategy that trades every bar unconditionally scores the same.
What it destroys is multi-bar structure - the gaps, order blocks, sweeps and
shifts a model claims to read. A model that still earns on shuffled bars is
earning from the engine, not the market.
"""
import numpy as np
import pandas as pd
import pytest

from infrastructure.tripwire import run_tripwire, shuffle_bars


def _bars(closes, start='2024-01-01'):
    """Build a coherent OHLC frame from a close path."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    idx = pd.date_range(start, periods=len(closes), freq='5min', tz='UTC')
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows,
                         'close': closes, 'volume': 100.0}, index=idx)


def _trending():
    """50 up bars then 50 down bars - strong, obvious serial structure."""
    up = np.arange(100.0, 150.0)
    down = np.arange(150.0, 100.0, -1.0)
    return _bars(np.concatenate([up, down]))


def _trade_log(rs):
    return pd.DataFrame({
        'pnl_r': list(rs),
        'exit_reason': ['TP' if r > 0 else 'SL' for r in rs],
    })


# --- shuffle_bars ---------------------------------------------------------

def test_shuffle_preserves_length_and_index():
    df = _trending()
    out = shuffle_bars(df, seed=1)
    assert len(out) == len(df)
    assert out.index.equals(df.index)


def test_shuffle_preserves_ohlc_coherence():
    out = shuffle_bars(_trending(), seed=1)
    assert (out['high'] >= out[['open', 'close']].max(axis=1) - 1e-9).all()
    assert (out['low'] <= out[['open', 'close']].min(axis=1) + 1e-9).all()
    assert (out['high'] >= out['low']).all()


def test_shuffle_actually_reorders_the_series():
    df = _trending()
    out = shuffle_bars(df, seed=1)
    assert not np.allclose(out['close'].values, df['close'].values)


def test_shuffle_is_deterministic_for_a_seed():
    df = _trending()
    a = shuffle_bars(df, seed=7)
    b = shuffle_bars(df, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_series():
    df = _trending()
    a = shuffle_bars(df, seed=1)
    b = shuffle_bars(df, seed=2)
    assert not np.allclose(a['close'].values, b['close'].values)


def test_shuffle_preserves_the_multiset_of_bar_shapes():
    """Every bar's own geometry survives; only its position changes."""
    df = _trending()
    out = shuffle_bars(df, seed=3)
    orig = np.sort(np.round((df['close'] / df['open']).values, 10))
    new = np.sort(np.round((out['close'] / out['open']).values, 10))
    assert np.allclose(orig, new)


def test_shuffle_preserves_non_price_columns():
    out = shuffle_bars(_trending(), seed=1)
    assert 'volume' in out.columns


# --- run_tripwire ---------------------------------------------------------

def _momentum_model(df):
    """Buy after two consecutive up bars; +1 R if the next bar is up.

    Real structure-dependent: profitable on a trending series, worthless once
    the bars are shuffled.
    """
    up = (df['close'] > df['open']).values
    rs = [1.0 if up[i] else -1.0
          for i in range(2, len(up)) if up[i - 1] and up[i - 2]]
    return _trade_log(rs)


def _always_wins_model(df):
    """Stands in for a lookahead bug: earns regardless of the data."""
    return _trade_log([1.0] * 40)


def test_structure_dependent_model_beats_its_null():
    res = run_tripwire(_momentum_model, _trending(), n_runs=25, seed=0)
    assert res.observed_expectancy > res.null_mean
    assert res.edge_significant is True
    assert res.leakage_suspected is False
    assert res.passed is True


def test_model_that_wins_on_shuffled_bars_raises_the_leakage_alarm():
    res = run_tripwire(_always_wins_model, _trending(), n_runs=25, seed=0)
    assert res.leakage_suspected is True
    assert res.passed is False
    assert 'leakage' in res.detail.lower()


def test_null_distribution_has_one_entry_per_run():
    res = run_tripwire(_momentum_model, _trending(), n_runs=12, seed=0)
    assert len(res.null_expectancies) == 12


def test_tripwire_is_deterministic_for_a_seed():
    a = run_tripwire(_momentum_model, _trending(), n_runs=10, seed=5)
    b = run_tripwire(_momentum_model, _trending(), n_runs=10, seed=5)
    assert a.null_expectancies == b.null_expectancies
    assert a.observed_expectancy == b.observed_expectancy


def test_model_receives_shuffled_frames_not_the_original():
    df = _trending()
    seen = []

    def spy(frame):
        seen.append(frame['close'].values.copy())
        return _trade_log([1.0, -1.0])

    run_tripwire(spy, df, n_runs=3, seed=0)
    assert len(seen) == 4          # 1 real run + 3 null runs
    assert np.allclose(seen[0], df['close'].values)
    for nulled in seen[1:]:
        assert not np.allclose(nulled, df['close'].values)


def test_model_producing_no_trades_does_not_raise():
    res = run_tripwire(lambda df: _trade_log([]), _trending(), n_runs=5, seed=0)
    assert res.observed_expectancy == 0.0
    assert res.passed is False


def test_percentile_reports_where_the_real_result_sits_in_the_null():
    res = run_tripwire(_momentum_model, _trending(), n_runs=20, seed=0)
    assert 0.0 <= res.percentile <= 100.0
    assert res.percentile > 90.0


def test_n_runs_must_be_positive():
    with pytest.raises(ValueError):
        run_tripwire(_momentum_model, _trending(), n_runs=0, seed=0)
