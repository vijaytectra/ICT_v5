"""Promotion gate tests. Synthetic logs only - no data files needed.

The gate is the written answer to "is this model good enough?", so these tests
are mostly about it saying *no* for the right reason, and naming that reason.
"""
import pandas as pd
import pytest

from infrastructure.promotion import (DEFAULT_CRITERIA, GateCriteria,
                                      evaluate_gate)


def _log(rs, start='2025-01-01'):
    """Minimal trade log carrying the columns compute_stats needs."""
    dates = pd.date_range(start, periods=len(rs), freq='D')
    return pd.DataFrame({
        'date': [d.date().isoformat() for d in dates],
        'entry_time': dates,
        'exit_time': dates + pd.Timedelta(hours=1),
        'pnl_r': list(rs),
        'exit_reason': ['TP' if r > 0 else 'SL' for r in rs],
        'instrument': ['NQ'] * len(rs),
        'holding_minutes': [60.0] * len(rs),
    })


def _healthy(n=40, win_every=3):
    """A log that clears every default threshold: ~33% WR at 2.5R."""
    return [2.5 if i % win_every == 0 else -1.0 for i in range(n)]


def _named(result, name):
    matches = [c for c in result.criteria if c.name == name]
    assert matches, f"no criterion named {name!r} in {[c.name for c in result.criteria]}"
    return matches[0]


# --- the happy path -------------------------------------------------------

def test_healthy_out_of_sample_promotes():
    res = evaluate_gate(_log(_healthy(60)), _log(_healthy(40)))
    assert res.passed is True
    assert res.verdict == 'PROMOTE'


def test_every_criterion_is_reported_even_when_passing():
    res = evaluate_gate(_log(_healthy(60)), _log(_healthy(40)))
    assert {c.name for c in res.criteria} == {
        'oos_sample', 'oos_expectancy', 'oos_profit_factor',
        'expectancy_decay', 'loss_streak'}
    assert all(c.passed for c in res.criteria)


# --- each criterion rejects on its own ------------------------------------

def test_too_few_out_of_sample_trades_rejects():
    res = evaluate_gate(_log(_healthy(60)), _log(_healthy(9)))
    assert res.passed is False
    assert res.verdict == 'REJECT'
    assert _named(res, 'oos_sample').passed is False


def test_negative_out_of_sample_expectancy_rejects():
    losing = [1.0 if i % 5 == 0 else -1.0 for i in range(40)]
    res = evaluate_gate(_log(_healthy(60)), _log(losing))
    assert res.passed is False
    assert _named(res, 'oos_expectancy').passed is False


def test_profit_factor_below_threshold_rejects():
    # 40 trades, expectancy positive but PF only ~1.07
    marginal = [2.0 if i % 3 == 0 else -0.93 for i in range(42)]
    res = evaluate_gate(_log(_healthy(60)), _log(marginal))
    assert _named(res, 'oos_profit_factor').passed is False
    assert res.passed is False


def test_expectancy_halving_out_of_sample_rejects():
    """In-sample +1.0 R, out-of-sample +0.1 R: positive, but 90% decayed."""
    strong = [3.0 if i % 2 == 0 else -1.0 for i in range(60)]      # +1.0 R
    faded = [2.5 if i % 4 == 0 else -0.7 for i in range(40)]        # ~+0.10 R
    res = evaluate_gate(_log(strong), _log(faded))
    decay = _named(res, 'expectancy_decay')
    assert decay.passed is False
    assert res.passed is False


def test_long_losing_streak_relative_to_sample_rejects():
    """30 trades with a 12-long streak: 40% of the sample in one run."""
    rs = [-1.0] * 12 + [2.5] * 10 + [-1.0] * 4 + [2.5] * 4
    res = evaluate_gate(_log(_healthy(60)), _log(rs))
    assert _named(res, 'loss_streak').passed is False
    assert res.passed is False


# --- reporting behaviour --------------------------------------------------

def test_all_failures_are_reported_not_just_the_first():
    res = evaluate_gate(_log(_healthy(60)), _log([-1.0] * 8))
    failed = {c.name for c in res.criteria if not c.passed}
    assert {'oos_sample', 'oos_expectancy', 'oos_profit_factor'} <= failed


def test_criterion_carries_observed_and_threshold():
    res = evaluate_gate(_log(_healthy(60)), _log(_healthy(9)))
    sample = _named(res, 'oos_sample')
    assert sample.observed == 9
    assert sample.threshold == DEFAULT_CRITERIA.min_oos_trades


def test_empty_out_of_sample_log_rejects_without_raising():
    res = evaluate_gate(_log(_healthy(60)), _log([]))
    assert res.passed is False
    assert _named(res, 'oos_sample').observed == 0


# --- criteria are configurable, and the defaults are the documented ones ---

def test_defaults_match_the_documented_gate():
    assert DEFAULT_CRITERIA.min_oos_trades == 30
    assert DEFAULT_CRITERIA.min_oos_expectancy_r == 0.0
    assert DEFAULT_CRITERIA.min_oos_profit_factor == 1.15
    assert DEFAULT_CRITERIA.max_expectancy_decay == 0.5
    assert DEFAULT_CRITERIA.max_loss_streak_fraction == 0.15


def test_looser_criteria_can_promote_what_defaults_reject():
    oos = _log(_healthy(20))
    assert evaluate_gate(_log(_healthy(60)), oos).passed is False
    loose = GateCriteria(min_oos_trades=15)
    assert evaluate_gate(_log(_healthy(60)), oos, loose).passed is True


def test_zero_or_negative_in_sample_expectancy_fails_decay_criterion():
    """Nothing to decay from is not a pass - it means the model never worked."""
    flat = [1.0 if i % 2 == 0 else -1.0 for i in range(60)]   # expectancy 0
    res = evaluate_gate(_log(flat), _log(_healthy(40)))
    assert _named(res, 'expectancy_decay').passed is False
