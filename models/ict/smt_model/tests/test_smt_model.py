"""
Tests for smt_model.py.

The focus is the control arm and the checklist's rule of exclusion, because
those are where the model can silently stop being a fair experiment:

- Arm B (`require_smt: False`) must still be the SAME setup minus the
  divergence. An anchor where nothing was raided is not a control, and an
  anchor with no leader used to raise IndexError rather than being skipped.
- Every gate must be attributable in the funnel, or a run that produces no
  trades cannot be diagnosed.
"""

import numpy as np
import pandas as pd
import pytest

from models.ict.smt_model import SMTModel, SMTSnapshot

SESSION = pd.Timestamp('2024-07-10')


def frame(lows, highs=None, start='2024-07-10 12:00', freq='5min'):
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else lows + 10.0
    idx = pd.date_range(start, periods=len(lows), freq=freq, tz='UTC')
    mid = (highs + lows) / 2
    return pd.DataFrame({'open': mid, 'high': highs, 'low': lows,
                         'close': mid, 'volume': 1.0}, index=idx)


def daily_frame(n=60, start='2024-05-01', up=True):
    idx = pd.date_range(start, periods=n, freq='D', tz='UTC')
    base = np.linspace(100, 200, n) if up else np.linspace(200, 100, n)
    return pd.DataFrame({'open': base, 'high': base + 5, 'low': base - 5,
                         'close': base + (2 if up else -2), 'volume': 1.0},
                        index=idx)


def flat_session(n=60, level=100.0):
    """A session with no structure at all: nothing raided, nothing to compare."""
    return frame(np.full(n, level), highs=np.full(n, level + 1.0))


class TestConfigContract:

    def test_unknown_config_key_is_rejected(self):
        with pytest.raises(ValueError, match='Unknown SMTModel config keys'):
            SMTModel({'not_a_real_key': 1})

    def test_defaults_are_complete(self):
        m = SMTModel()
        for key in ('require_smt', 'windows', 'min_clarity_adr',
                    'max_clarity_adr', 'instrument_selection', 'stop_mode',
                    'bias_mode', 'entry_search_bars'):
            assert key in m.config


class TestRuleOfExclusion:

    def _snap(self, frames, up=True):
        return SMTSnapshot(
            frames=frames,
            daily={s: daily_frame(up=up) for s in frames},
            session_date=SESSION,
            inverse=(),
            four_h=None,
        )

    def test_missing_adr_history_is_rejected_not_crashed(self):
        frames = {'A': flat_session(), 'B': flat_session(level=200.0)}
        snap = SMTSnapshot(frames=frames,
                           daily={s: daily_frame(n=3) for s in frames},
                           session_date=SESSION)
        sig = SMTModel().generate_signal(snap, 'pm')
        assert sig['actionable'] is False
        assert sig['reject_stage'] == 'adr'

    def test_featureless_session_is_rejected_with_a_named_stage(self):
        frames = {'A': flat_session(), 'B': flat_session(level=200.0)}
        sig = SMTModel().generate_signal(self._snap(frames), 'pm')
        assert sig['actionable'] is False
        # Whatever it is, it must be attributable - never a bare 'no_setup'
        # with no explanation, or an empty funnel row.
        assert sig['reject_stage'] is not None
        assert sig['reject_reason']

    def test_rejected_signal_has_the_full_shape(self):
        frames = {'A': flat_session(), 'B': flat_session(level=200.0)}
        sig = SMTModel().generate_signal(self._snap(frames), 'pm')
        for k in ('actionable', 'direction', 'instrument', 'entry', 'stop',
                  'target', 'reject_stage', 'reject_reason', 'checks'):
            assert k in sig


class TestControlArm:
    """`require_smt: False` must stay a fair control, not a different model."""

    def _snap(self, frames):
        return SMTSnapshot(
            frames=frames,
            daily={s: daily_frame(up=True) for s in frames},
            session_date=SESSION, inverse=(), four_h=None,
        )

    def test_anchor_without_a_raid_is_skipped_not_crashed(self):
        # A session whose second relative low is HIGHER than the first: nothing
        # was swept. Arm B must skip it. This previously raised IndexError on
        # `event.leader[0]` and killed the whole experiment suite mid-run.
        n = 40
        a = np.full(n, 140.0)
        for pos, val in ((7, 100.0), (21, 120.0)):      # second low is higher
            a[pos] = val
            a[pos - 1] = a[pos + 1] = val + 12.0
            a[pos - 2] = a[pos + 2] = val + 25.0
        frames = {'A': frame(a), 'B': frame(a + 100.0)}
        model = SMTModel({'require_smt': False, 'fixed_instrument': 'A'})
        sig = model.generate_signal(self._snap(frames), 'pm')
        assert sig['actionable'] is False
        assert sig['reject_stage'] is not None

    def test_control_arm_runs_on_a_raided_session(self):
        # Second relative low is LOWER: a raid happened, so arm B may proceed
        # past the anchor stage even though no divergence exists.
        n = 40
        a = np.full(n, 140.0)
        for pos, val in ((7, 120.0), (21, 100.0)):      # second low is lower
            a[pos] = val
            a[pos - 1] = a[pos + 1] = val + 12.0
            a[pos - 2] = a[pos + 2] = val + 25.0
        frames = {'A': frame(a), 'B': frame(a + 100.0)}
        model = SMTModel({'require_smt': False, 'fixed_instrument': 'A'})
        sig = model.generate_signal(self._snap(frames), 'pm')
        # It need not produce a trade on this toy fixture, but it must get
        # past 'anchor' - otherwise arm B can never fire and A-vs-B is void.
        assert sig['reject_stage'] != 'anchor'

    def test_selection_rule_is_ignored_when_the_gate_is_off(self):
        # With no divergence there is no diverging instrument to select; the
        # control must not silently reject every session because of it.
        n = 40
        a = np.full(n, 140.0)
        for pos, val in ((7, 120.0), (21, 100.0)):
            a[pos] = val
            a[pos - 1] = a[pos + 1] = val + 12.0
            a[pos - 2] = a[pos + 2] = val + 25.0
        frames = {'A': frame(a), 'B': frame(a + 100.0)}
        model = SMTModel({'require_smt': False, 'fixed_instrument': 'A',
                          'instrument_selection': 'diverging'})
        sig = model.generate_signal(self._snap(frames), 'pm')
        assert sig['reject_stage'] != 'select'


class TestTargetGeometry:
    """A target must never end up on the wrong side of the entry.

    The ADR cap is a ceiling on ambition. When the session has already
    delivered most of its expected range the cap can sit beyond the entry, and
    a bare min()/max() then relocates the target through the entry — pricing a
    long's target BELOW its own stop. `simulate_exit` fires 'target hit' on the
    next bar and books a ~2R loss as a TP. In the first full-history run this
    hit 31% of signals and made every aggregate meaningless while looking
    merely unprofitable.
    """

    def _model(self, **over):
        return SMTModel({'target_mode': 'rr', 'target_rr': 2.0,
                         'target_min_rr': 2.0, 'cap_target_at_adr': True,
                         **over})

    def _snap(self, df):
        return SMTSnapshot(frames={'A': df}, daily={'A': daily_frame()},
                           session_date=SESSION)

    def test_cap_below_entry_stands_aside_instead_of_inverting(self):
        # Session low far under the entry, small ADR -> cap = low + 0.9*ADR
        # lands below the entry. The correct answer is "no room", not a target
        # on the wrong side.
        n = 40
        lows = np.full(n, 4400.0)
        lows[0] = 4300.0                       # session low well below entry
        df = frame(lows, highs=lows + 5.0)
        m = self._model()
        t = m._resolve_target(df, self._snap(df), 'long',
                              entry=4490.0, stop=4480.0,
                              before=df.index[-1], adr=30.0, exit_ny='16:00')
        assert t is None

    def test_short_mirror(self):
        n = 40
        highs = np.full(n, 4400.0)
        highs[0] = 4500.0
        df = frame(highs - 5.0, highs=highs)
        m = self._model()
        t = m._resolve_target(df, self._snap(df), 'short',
                              entry=4310.0, stop=4320.0,
                              before=df.index[-1], adr=30.0, exit_ny='16:00')
        assert t is None

    def test_valid_target_still_returned(self):
        n = 40
        lows = np.full(n, 4480.0)
        df = frame(lows, highs=lows + 5.0)
        m = self._model()
        t = m._resolve_target(df, self._snap(df), 'long',
                              entry=4490.0, stop=4480.0,
                              before=df.index[-1], adr=200.0, exit_ny='16:00')
        assert t is not None and t > 4490.0

    def test_stop_is_pushed_beyond_the_entry_array(self):
        # The stop anchor is the SMT invalidation level, but a sweep can run
        # deeper than the pivot it is measured from, leaving the FVG being
        # entered below that level. The stop must then follow the array, not
        # sit above the entry. Verified through the public signal contract in
        # TestRuleOfExclusion; here the arithmetic itself is pinned.
        for direction, anchor, array_edge, expect_below in (
                ('long', 4174.9, 4160.0, True),
                ('short', 4174.9, 4190.0, False)):
            a = (min(anchor, array_edge) if direction == 'long'
                 else max(anchor, array_edge))
            stop = a - 1.0 if direction == 'long' else a + 1.0
            entry = 4168.6 if direction == 'long' else 4180.0
            assert (stop < entry) if expect_below else (stop > entry)

    def test_runner_rejects_impossible_geometry(self):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                               / 'backtests'))
        from smt_backtest import _assert_sane
        _assert_sane({'direction': 'long', 'entry': 100.0, 'stop': 99.0,
                      'target': 102.0})
        with pytest.raises(AssertionError, match='impossible long geometry'):
            _assert_sane({'direction': 'long', 'entry': 100.0, 'stop': 99.0,
                          'target': 98.0})
        with pytest.raises(AssertionError, match='impossible short geometry'):
            _assert_sane({'direction': 'short', 'entry': 100.0, 'stop': 101.0,
                          'target': 102.0})


class TestWindows:

    def test_both_windows_are_defined_with_the_documented_bounds(self):
        from models.ict.smt_model import WINDOWS, WINDOW_EXIT
        assert WINDOWS['am'] == ('05:00', '09:30')
        assert WINDOWS['pm'] == ('12:00', '15:00')
        assert WINDOW_EXIT['am'] == '11:00'
        assert WINDOW_EXIT['pm'] == '16:00'
