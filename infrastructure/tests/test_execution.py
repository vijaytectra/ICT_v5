"""Tests for the execution cost model.

The important ones are the last two: they pin the difference between a price the
market genuinely jumped over and a price it simply traded through on a bar the
exit scan does not visit. Conflating those turned a -1R stop into a -20R loss on
an NFP candle, and the only thing that caught it was the P&L being absurd.
"""

import pandas as pd
import pytest

from infrastructure.execution import (CostModel, ZERO_COST, half_spread,
                                      simulate_entry, simulate_exit,
                                      cost_model_from_config, SL, TP, EOD_LOSS)


def _bars(rows, start='2024-01-02 12:00', freq='5min', spread=0.0002):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz='UTC')
    df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'], index=idx)
    df['volume'] = 1.0
    df['spread'] = spread
    return df


def _signal(direction='long', entry=1.0, stop=0.99, target=1.03, t=None,
            style='limit'):
    return {'direction': direction, 'entry': entry, 'stop': stop,
            'target': target, 'entry_time': t, 'entry_style': style,
            'session': 'london', 'rr': abs(target - entry) / abs(entry - stop)}


# ── cost model plumbing ──────────────────────────────────────────────────────

def test_zero_cost_is_zero():
    assert ZERO_COST.is_zero
    assert half_spread({'spread': 0.0005}, ZERO_COST) == 0.0


def test_negative_and_missing_spread_clamp_to_zero():
    """Crossed quotes in the source data are artefacts, never a credit."""
    c = CostModel(spread_mode='historical')
    assert half_spread({'spread': -0.0003}, c) == 0.0
    assert half_spread({'spread': float('nan')}, c) == 0.0
    assert half_spread({}, c) == 0.0
    assert half_spread({'spread': 0.0004}, c) == pytest.approx(0.0002)


def test_costs_must_be_non_negative():
    with pytest.raises(ValueError):
        CostModel(slippage_stop=-0.0001)
    with pytest.raises(ValueError):
        CostModel(spread_mode='bogus')
    with pytest.raises(ValueError):
        CostModel(gap_fill='sometimes')


def test_cost_model_from_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match='Unknown cost config'):
        cost_model_from_config({'costs': {'sprad_mode': 'historical'}})
    assert cost_model_from_config({}).is_zero


# ── directional cost application ─────────────────────────────────────────────

def test_long_pays_the_ask_and_sells_the_bid():
    df = _bars([(1.00, 1.00, 1.00, 1.00),        # signal bar
                (1.00, 1.00, 0.995, 0.999),      # fill bar - touches the limit
                (0.999, 1.031, 0.998, 1.030)], spread=0.0002)
    sig = _signal('long', entry=1.00, stop=0.99, target=1.03, t=df.index[0])
    cost = CostModel(spread_mode='historical')
    fill = simulate_entry(df, sig, None, cost)
    # Bought the ask: mid + half the 2-pip spread.
    assert fill['fill_price'] == pytest.approx(1.0001)
    ex = simulate_exit(df, sig, fill, df.index[-1], cost)
    # Sold the bid at the target.
    assert ex['exit_price'] == pytest.approx(1.0299)
    assert ex['exit_reason'] == TP
    assert ex['spread_cost_pts'] == pytest.approx(0.0002)


def test_short_is_the_mirror():
    df = _bars([(1.00, 1.00, 1.00, 1.00),
                (1.00, 1.005, 1.00, 1.004),
                (1.004, 1.005, 0.969, 0.970)], spread=0.0002)
    sig = _signal('short', entry=1.00, stop=1.01, target=0.97, t=df.index[0])
    cost = CostModel(spread_mode='historical')
    fill = simulate_entry(df, sig, None, cost)
    assert fill['fill_price'] == pytest.approx(0.9999)     # sold the bid
    ex = simulate_exit(df, sig, fill, df.index[-1], cost)
    assert ex['exit_price'] == pytest.approx(0.9701)       # bought the ask
    assert ex['exit_reason'] == TP


def test_slippage_is_always_adverse():
    df = _bars([(1.00, 1.00, 1.00, 1.00),
                (1.00, 1.00, 0.995, 0.999),
                (0.999, 0.999, 0.985, 0.986)])
    sig = _signal('long', entry=1.00, stop=0.99, target=1.03, t=df.index[0])
    plain = CostModel(gap_fill='stop')
    slipped = CostModel(slippage_stop=0.0005, gap_fill='stop')
    a = simulate_exit(df, sig, simulate_entry(df, sig, None, plain),
                      df.index[-1], plain)
    b = simulate_exit(df, sig, simulate_entry(df, sig, None, slipped),
                      df.index[-1], slipped)
    assert b['exit_price'] < a['exit_price']
    assert b['pnl_pts'] < a['pnl_pts']
    assert a['exit_reason'] == b['exit_reason'] == SL


# ── the gap rule ─────────────────────────────────────────────────────────────

def test_real_gap_fills_at_the_open():
    """Price was never available at the stop, so the open is the honest fill."""
    df = _bars([(1.00, 1.00, 1.00, 1.00),        # signal bar
                (1.00, 1.00, 0.998, 0.999),      # fill bar, nowhere near the stop
                (0.980, 0.982, 0.975, 0.976)])   # opens far below it
    sig = _signal('long', entry=1.00, stop=0.99, target=1.03, t=df.index[0])
    cost = CostModel(gap_fill='open')
    fill = simulate_entry(df, sig, None, cost)
    ex = simulate_exit(df, sig, fill, df.index[-1], cost)
    assert ex['exit_reason'] == SL
    assert ex['exit_mid'] == pytest.approx(0.980)
    assert ex['pnl_pts'] < -0.019          # genuinely worse than a -0.01 stop


def test_same_bar_breach_is_not_a_gap():
    """The regression that manufactured a -20R loss.

    The entry bar itself trades through the stop, then the next bar opens far
    beyond it. The stop was reachable on the entry bar, so the fill is the stop -
    not the next open. Without this, an NFP candle that fills the limit and blows
    through the stop in one bar gets 'stopped' 60 pips away.
    """
    df = _bars([(1.00, 1.00, 1.00, 1.00),        # signal bar
                (1.00, 1.00, 0.985, 0.986),      # fills the limit AND breaches 0.99
                (0.986, 0.987, 0.980, 0.981)])   # opens below the stop
    sig = _signal('long', entry=1.00, stop=0.99, target=1.03, t=df.index[0])
    cost = CostModel(gap_fill='open')
    fill = simulate_entry(df, sig, None, cost)
    ex = simulate_exit(df, sig, fill, df.index[-1], cost)
    assert ex['exit_reason'] == SL
    assert ex['exit_mid'] == pytest.approx(0.99), 'stop was reachable; fill at the stop'
    assert ex['pnl_pts'] == pytest.approx(-0.01)


def test_gap_fill_stop_reproduces_legacy():
    """Default gap handling never fills worse than the stop."""
    df = _bars([(1.00, 1.00, 1.00, 1.00),
                (1.00, 1.00, 0.998, 0.999),
                (0.980, 0.982, 0.975, 0.976)])
    sig = _signal('long', entry=1.00, stop=0.99, target=1.03, t=df.index[0])
    fill = simulate_entry(df, sig, None, ZERO_COST)
    ex = simulate_exit(df, sig, fill, df.index[-1], ZERO_COST)
    assert ex['exit_mid'] == pytest.approx(0.99)
    assert ex['pnl_pts'] == pytest.approx(-0.01)


# ── exit reasons ─────────────────────────────────────────────────────────────

def test_eod_loss_is_labelled_separately_from_a_stop():
    df = _bars([(1.00, 1.00, 1.00, 1.00),
                (1.00, 1.00, 0.999, 0.9995),
                (0.9995, 0.9996, 0.9930, 0.9940)])
    sig = _signal('long', entry=1.00, stop=0.98, target=1.05, t=df.index[0])
    fill = simulate_entry(df, sig, None, ZERO_COST)
    ex = simulate_exit(df, sig, fill, df.index[-1], ZERO_COST)
    assert ex['exit_reason'] == EOD_LOSS
    assert ex['pnl_pts'] < 0


def test_market_entry_fills_at_the_next_open_not_the_signal_close():
    df = _bars([(1.00, 1.01, 1.00, 1.008),       # confirmation candle
                (1.004, 1.05, 1.003, 1.049)])    # filled at THIS open
    sig = _signal('long', entry=1.008, stop=0.99, target=1.04,
                  t=df.index[0], style='market')
    fill = simulate_entry(df, sig, None, ZERO_COST)
    assert fill['fill_price'] == pytest.approx(1.004)
    assert fill['fill_time'] == df.index[1]
