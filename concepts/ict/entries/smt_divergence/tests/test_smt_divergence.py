"""
Tests for smt_divergence.py — find_smt_events(), negated_at(), smt_is_live().

Every fixture is hand-built so the correct answer is known by construction. The
properties under test are the ones the previous rolling-window detector got
wrong: window anchoring, same-timestamp comparison across instruments, inverse
correlation, tandem rejection, and expiry.

Bars are 5-minute and placed in the NY PM comparison window (12:00-15:00 NY) on
a summer date, so the UTC offset is -4 and 16:00-19:00 UTC is the window.
"""

import numpy as np
import pandas as pd
import pytest

from concepts import find_smt_events, negated_at, smt_is_live
from concepts.ict.entries.smt_divergence.smt_divergence import SMTEvent

SESSION = pd.Timestamp('2024-07-10')
PM = ('12:00', '15:00')
AM = ('05:00', '09:30')


def frame(lows, highs=None, start='2024-07-10 16:00', freq='5min'):
    """Aligned OHLCV frame from a low series (highs mirror unless given)."""
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else lows + 10.0
    idx = pd.date_range(start, periods=len(lows), freq=freq, tz='UTC')
    return pd.DataFrame({
        'open': (highs + lows) / 2,
        'high': highs,
        'low': lows,
        'close': (highs + lows) / 2,
        'volume': 1.0,
    }, index=idx)


def vshape(first_low, second_low, n=30):
    """Two clean pivot lows inside one window: dip at bar 7 and bar 21."""
    base = np.full(n, first_low + 40.0)
    base[7] = first_low
    base[6] = base[8] = first_low + 12.0
    base[5] = base[9] = first_low + 25.0
    base[21] = second_low
    base[20] = base[22] = second_low + 12.0
    base[19] = base[23] = second_low + 25.0
    return base


def inverted(first_high, second_high, n=30):
    """Mirror of vshape for an inversely correlated instrument."""
    base = np.full(n, first_high - 40.0)
    base[7] = first_high
    base[6] = base[8] = first_high - 12.0
    base[5] = base[9] = first_high - 25.0
    base[21] = second_high
    base[20] = base[22] = second_high - 12.0
    base[19] = base[23] = second_high - 25.0
    return base


class TestTandemIsNotSMT:

    def test_all_make_new_low_returns_nothing(self):
        # Every instrument makes a lower low -> correlation intact, no crack.
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 190)),
             'C': frame(vshape(300, 290))}
        assert find_smt_events(c, SESSION, PM, 'low') == []

    def test_none_make_new_low_returns_nothing(self):
        c = {'A': frame(vshape(100, 110)),
             'B': frame(vshape(200, 210)),
             'C': frame(vshape(300, 310))}
        assert find_smt_events(c, SESSION, PM, 'low') == []


class TestDivergenceDetection:

    def test_one_leader_two_diverging(self):
        # A sweeps to a lower low; B and C hold higher lows -> bullish SMT.
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210)),
             'C': frame(vshape(300, 310))}
        ev = find_smt_events(c, SESSION, PM, 'low')
        assert len(ev) == 1
        e = ev[0]
        assert e.side == 'low'
        assert e.direction == 'long'
        assert e.leader == ('A',)
        assert set(e.diverging) == {'B', 'C'}

    def test_bearish_side_uses_highs(self):
        c = {'A': frame(np.full(30, 50.0), highs=inverted(100, 110)),
             'B': frame(np.full(30, 50.0), highs=inverted(200, 190)),
             'C': frame(np.full(30, 50.0), highs=inverted(300, 290))}
        ev = find_smt_events(c, SESSION, PM, 'high')
        assert len(ev) == 1
        assert ev[0].leader == ('A',)
        assert ev[0].direction == 'short'

    def test_signal_time_is_the_second_pivot(self):
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210))}
        e = find_smt_events(c, SESSION, PM, 'low')[0]
        assert e.time == e.pivot_times[1]
        assert e.pivot_times[0] < e.pivot_times[1]

    def test_levels_recorded_per_instrument(self):
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210))}
        e = find_smt_events(c, SESSION, PM, 'low')[0]
        assert e.levels['A'] == (100.0, 90.0)      # made the new low
        assert e.levels['B'] == (200.0, 210.0)     # refused it


class TestClarityAndSweepDepth:

    def test_signs_and_scale(self):
        # scale=100 -> A ran 10 points past (0.10), B fell 10 short (0.10).
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210))}
        e = find_smt_events(c, SESSION, PM, 'low',
                            scale={'A': 100.0, 'B': 100.0})[0]
        assert e.sweep_depth == pytest.approx(0.10)
        assert e.clarity == pytest.approx(0.10)

    def test_clarity_is_the_weakest_refusal(self):
        # B refuses by 10, C refuses by 50. Clarity must take B's 10, so one
        # lukewarm member cannot inflate the reading.
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210)),
             'C': frame(vshape(300, 350))}
        e = find_smt_events(c, SESSION, PM, 'low',
                            scale={'A': 100.0, 'B': 100.0, 'C': 100.0})[0]
        assert e.clarity == pytest.approx(0.10)

    def test_sweep_depth_is_the_deepest_leader(self):
        c = {'A': frame(vshape(100, 90)),    # 10 past
             'B': frame(vshape(200, 160)),   # 40 past
             'C': frame(vshape(300, 310))}
        e = find_smt_events(c, SESSION, PM, 'low',
                            scale={'A': 100.0, 'B': 100.0, 'C': 100.0})[0]
        assert e.sweep_depth == pytest.approx(0.40)


class TestInverseCorrelation:
    """DXY confirms an FX low by making its own HIGH, not its own low."""

    @staticmethod
    def _dxy(first_high, second_high):
        """A real inverse instrument: highs AND lows both move together."""
        highs = inverted(first_high, second_high)
        return frame(highs - 10.0, highs=highs)

    def test_inverse_member_moving_in_tandem_is_not_a_divergence(self):
        # EU and GU make lower lows; DXY makes a higher high -> all confirming.
        c = {'EU': frame(vshape(100, 90)),
             'GU': frame(vshape(200, 190)),
             'DX': self._dxy(300, 310)}
        assert find_smt_events(c, SESSION, PM, 'low', inverse=('DX',)) == []

    def test_inverse_member_can_be_the_diverging_one(self):
        # EU and GU make lower lows; DXY FAILS to make a higher high.
        c = {'EU': frame(vshape(100, 90)),
             'GU': frame(vshape(200, 190)),
             'DX': self._dxy(300, 290)}
        ev = find_smt_events(c, SESSION, PM, 'low', inverse=('DX',))
        assert len(ev) == 1
        assert ev[0].diverging == ('DX',)
        assert set(ev[0].leader) == {'EU', 'GU'}

    def test_omitting_the_inverse_flag_manufactures_a_false_signal(self):
        # A genuinely tandem session: EU/GU lower lows, DXY higher high.
        # Told DXY is inverse -> correctly silent.
        # Not told -> DXY's LOW rose, so it is scored as "failed to make a lower
        # low" and a divergence is reported that never existed. This is exactly
        # the KB's "wrong pair correlation" mistake, and it is why `inverse` is
        # a required part of the group definition rather than a nicety.
        c = {'EU': frame(vshape(100, 90)),
             'GU': frame(vshape(200, 190)),
             'DX': self._dxy(300, 310)}
        assert find_smt_events(c, SESSION, PM, 'low', inverse=('DX',)) == []
        spurious = find_smt_events(c, SESSION, PM, 'low', inverse=())
        assert len(spurious) == 1
        assert spurious[0].diverging == ('DX',)


class TestWindowAnchoring:

    def test_bars_outside_the_window_are_ignored(self):
        # Divergence sits in the PM window; asking for the AM window sees none.
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210))}
        assert find_smt_events(c, SESSION, AM, 'low') == []

    def test_wrong_session_date_sees_nothing(self):
        c = {'A': frame(vshape(100, 90)),
             'B': frame(vshape(200, 210))}
        assert find_smt_events(c, pd.Timestamp('2024-07-11'), PM, 'low') == []

    def test_short_window_is_skipped(self):
        # A half-session: too few bars in the window to compare two relative
        # extremes honestly, so the detector declines rather than guessing.
        c = {'A': frame([104, 103, 100, 103, 106, 105, 96, 99]),
             'B': frame([204, 203, 200, 203, 206, 205, 206, 209])}
        assert find_smt_events(c, SESSION, PM, 'low', min_window_bars=12) == []

    def test_dst_winter_window_resolves_to_a_different_utc_hour(self):
        # January: NY is UTC-5, so 12:00 NY is 17:00 UTC (not 16:00 as in July).
        jan = pd.Timestamp('2024-01-10')
        c = {'A': frame(vshape(100, 90), start='2024-01-10 17:00'),
             'B': frame(vshape(200, 210), start='2024-01-10 17:00')}
        assert len(find_smt_events(c, jan, PM, 'low')) == 1
        # The same bars read as 16:00 UTC would be 11:00 NY - before the window.
        c2 = {'A': frame(vshape(100, 90), start='2024-01-10 16:00'),
              'B': frame(vshape(200, 210), start='2024-01-10 16:00')}
        ev = find_smt_events(c2, jan, PM, 'low')
        assert ev == [] or ev[0].time.tz_convert('America/New_York').hour >= 12


class TestAlignmentContract:

    def test_misaligned_indices_raise(self):
        a = frame(vshape(100, 90))
        b = frame(vshape(200, 210))
        b.index = b.index + pd.Timedelta(minutes=1)
        with pytest.raises(ValueError, match='shared bar index'):
            find_smt_events({'A': a, 'B': b}, SESSION, PM, 'low')

    def test_single_instrument_returns_nothing(self):
        assert find_smt_events({'A': frame(vshape(100, 90))},
                               SESSION, PM, 'low') == []

    def test_invalid_side_raises(self):
        c = {'A': frame(vshape(100, 90)), 'B': frame(vshape(200, 210))}
        with pytest.raises(ValueError, match="side must be"):
            find_smt_events(c, SESSION, PM, 'sideways')


class TestNegation:

    def _event_and_frames(self, tail_low_b):
        """Bullish SMT, then B's post-signal behaviour decides negation."""
        n = 40
        a = vshape(100, 90, n=n)
        b = vshape(200, 210, n=n)
        b[30:] = tail_low_b            # what B does after the signal
        c = {'A': frame(a), 'B': frame(b)}
        ev = find_smt_events(c, SESSION, PM, 'low')
        assert len(ev) == 1, "fixture must produce exactly one event"
        return c, ev[0]

    def test_divergence_stands_when_b_holds_its_level(self):
        c, e = self._event_and_frames(tail_low_b=205.0)   # stays above 200
        assert negated_at(c, e) is None
        assert smt_is_live(c, e, as_of=c['A'].index[-1])

    def test_divergence_dies_when_b_takes_its_own_level(self):
        c, e = self._event_and_frames(tail_low_b=195.0)   # trades through 200
        t = negated_at(c, e)
        assert t is not None
        assert t > e.time
        assert not smt_is_live(c, e, as_of=c['A'].index[-1])

    def test_until_bounds_the_scan(self):
        c, e = self._event_and_frames(tail_low_b=195.0)
        # Cut the scan before the breach: the divergence is still live then.
        assert negated_at(c, e, until=c['A'].index[29]) is None
        assert smt_is_live(c, e, as_of=c['A'].index[29])


class TestAllPairs:

    def test_default_returns_only_the_latest_comparison(self):
        # Three pivots -> two consecutive pairs. A live trader sees the last.
        n = 44
        a = np.full(n, 140.0)
        b = np.full(n, 240.0)
        for arr, (p1, p2, p3) in ((a, (100, 90, 95)), (b, (200, 210, 205))):
            for pos, val in zip((7, 21, 35), (p1, p2, p3)):
                arr[pos] = val
                arr[pos - 1] = arr[pos + 1] = val + 12.0
                arr[pos - 2] = arr[pos + 2] = val + 25.0
        c = {'A': frame(a), 'B': frame(b)}
        assert len(find_smt_events(c, SESSION, PM, 'low')) == 1
        assert len(find_smt_events(c, SESSION, PM, 'low', all_pairs=True)) >= 1


class TestBreachMode:
    """`find_smt_breach_events` — the strictly-causal reading.

    The pivot detector cannot fire until `swing_lookback` bars have printed to
    the right of the extreme, so it is late by construction. The breach reading
    treats the pool as history and the run through it as the live signal, which
    is what ICT describes ("that run under the 5 a.m. low ... that's a stop
    run") and the only version a turtle-soup entry can actually be filled on.
    """

    @staticmethod
    def _series(n, base, dip_at=None, dip_to=None):
        a = np.full(n, float(base))
        if dip_at is not None:
            a[dip_at] = float(dip_to)
        return a

    def test_no_lookahead_event_is_actionable_on_its_own_bar(self):
        from concepts import find_smt_breach_events
        a = self._series(30, 100.0, dip_at=20, dip_to=90.0)
        b = self._series(30, 200.0)
        c = {'A': frame(a), 'B': frame(b)}
        ev = find_smt_breach_events(c, SESSION, PM, 'low')
        assert len(ev) == 1
        # This is the property the pivot detector cannot have.
        assert ev[0].confirmed_time == ev[0].time

    def test_one_breaches_other_holds(self):
        from concepts import find_smt_breach_events
        a = self._series(30, 100.0, dip_at=20, dip_to=90.0)
        b = self._series(30, 200.0)
        c = {'A': frame(a), 'B': frame(b)}
        ev = find_smt_breach_events(c, SESSION, PM, 'low')
        assert ev[0].leader == ('A',)
        assert ev[0].diverging == ('B',)
        assert ev[0].direction == 'long'

    def test_tandem_breach_is_not_a_divergence(self):
        from concepts import find_smt_breach_events
        a = self._series(30, 100.0, dip_at=20, dip_to=90.0)
        b = self._series(30, 200.0, dip_at=20, dip_to=190.0)
        c = {'A': frame(a), 'B': frame(b)}
        assert find_smt_breach_events(c, SESSION, PM, 'low') == []

    def test_pool_excludes_the_bar_being_tested(self):
        # The pool a bar is judged against must be built only from bars that had
        # already closed. If the test bar could contribute to its own pool, no
        # bar could ever breach and the detector would be silent forever.
        from concepts import find_smt_breach_events
        a = np.full(40, 100.0)
        a[8] = 95.0                       # first dip - becomes the pool
        a[20] = 90.0                      # runs through it
        b = np.full(40, 200.0)            # holds
        c = {'A': frame(a), 'B': frame(b)}
        ev = find_smt_breach_events(c, SESSION, PM, 'low',
                                    pool_gap_bars=1, max_events=5)
        at20 = [e for e in ev if e.time == frame(a).index[20]]
        assert at20, "the run through 95 must be reported"
        pool, now = at20[0].levels['A']
        assert pool == pytest.approx(95.0)   # from bar 8, not from bar 20
        assert now == pytest.approx(90.0)

    def test_inverse_member_confirms_by_making_its_own_high(self):
        from concepts import find_smt_breach_events
        eu = self._series(30, 100.0, dip_at=20, dip_to=90.0)   # breaches low
        highs = np.full(30, 300.0)
        highs[20] = 310.0                                       # DXY breaks high
        c = {'EU': frame(eu), 'DX': frame(highs - 10.0, highs=highs)}
        # Both confirming -> tandem -> silent.
        assert find_smt_breach_events(c, SESSION, PM, 'low', inverse=('DX',)) == []

    def test_max_events_caps_the_result(self):
        from concepts import find_smt_breach_events
        a = np.full(40, 100.0)
        for i, v in ((20, 90.0), (26, 85.0), (32, 80.0)):
            a[i] = v
        b = np.full(40, 200.0)
        c = {'A': frame(a), 'B': frame(b)}
        assert len(find_smt_breach_events(c, SESSION, PM, 'low', max_events=1)) == 1
        assert len(find_smt_breach_events(c, SESSION, PM, 'low', max_events=5)) >= 2

    def test_correlation_gate_rejects_decorrelated_groups(self):
        from concepts import find_smt_breach_events
        rng = np.random.default_rng(0)
        n = 40
        a = 100.0 + rng.normal(0, 1, n).cumsum()
        b = 200.0 + rng.normal(0, 1, n).cumsum()   # independent
        a[30] = a[:29].min() - 5.0                 # force a breach
        c = {'A': frame(a), 'B': frame(b)}
        loose = find_smt_breach_events(c, SESSION, PM, 'low', min_abs_corr=0.0)
        strict = find_smt_breach_events(c, SESSION, PM, 'low', min_abs_corr=0.9)
        assert len(strict) <= len(loose)
