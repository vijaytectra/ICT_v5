"""Walk-forward engine tests. Synthetic logs only - no data files needed."""
import pandas as pd
import pytest

from infrastructure.walkforward import (Window, dedupe_logs, log_fingerprint,
                                        make_windows, normalize_log,
                                        select_config, slice_dates)


def _log(dates, rs, instrument='EURUSD'):
    """Minimal trade log with the columns the engine and compute_stats need."""
    return pd.DataFrame({
        'date': [pd.Timestamp(d).date().isoformat() for d in dates],
        'pnl_r': list(rs),
        'exit_reason': ['TP' if r > 0 else 'SL' for r in rs],
        'instrument': [instrument] * len(rs),
        'holding_minutes': [60.0] * len(rs),
    })


def test_make_windows_expanding_quarterly():
    ws = make_windows('2024-04-05', '2026-09-04',
                      min_train_quarters=4, test_quarters=1)
    assert len(ws) == 6

    first = ws[0]
    assert first.index == 0
    assert first.train_start.date().isoformat() == '2024-04-01'
    assert first.train_end.date().isoformat() == '2025-03-31'
    assert first.test_start.date().isoformat() == '2025-04-01'
    assert first.test_end.date().isoformat() == '2025-06-30'

    last = ws[-1]
    assert last.test_start.date().isoformat() == '2026-07-01'


def test_make_windows_are_expanding_not_rolling():
    ws = make_windows('2024-04-05', '2026-09-04')
    assert len({w.train_start for w in ws}) == 1


def test_make_windows_never_let_training_touch_test():
    ws = make_windows('2024-04-05', '2026-09-04')
    assert all(w.train_end < w.test_start for w in ws)


def test_make_windows_test_periods_do_not_overlap():
    ws = make_windows('2024-04-05', '2026-09-04')
    for a, b in zip(ws, ws[1:]):
        assert a.test_end < b.test_start


def test_make_windows_drops_incomplete_trailing_test_period():
    # 10 quarters total; min_train_quarters=4 leaves 6 quarters (indices 4-9)
    # for testing. test_quarters=4 fits one full window (indices 4-7) and
    # leaves a 2-quarter remainder (indices 8-9) that must be dropped, not
    # returned as a short/partial window.
    ws = make_windows('2024-04-05', '2026-09-04',
                      min_train_quarters=4, test_quarters=4)
    assert len(ws) == 1
    for w in ws:
        # a full test_quarters=4 span: Q2 2025 through Q1 2026 inclusive
        assert w.test_start.date().isoformat() == '2025-04-01'
        assert w.test_end.date().isoformat() == '2026-03-31'


def test_make_windows_returns_empty_when_min_train_quarters_exceeds_available():
    ws = make_windows('2024-04-05', '2026-09-04', min_train_quarters=20)
    assert ws == []


def test_make_windows_rejects_non_positive_test_quarters():
    with pytest.raises(ValueError, match='test_quarters'):
        make_windows('2024-04-05', '2026-09-04', test_quarters=0)


def test_make_windows_rejects_non_positive_min_train_quarters():
    with pytest.raises(ValueError, match='min_train_quarters'):
        make_windows('2024-04-05', '2026-09-04', min_train_quarters=0)


def test_normalize_log_parses_dates_and_sorts():
    raw = _log(['2024-03-01', '2024-01-01', '2024-02-01'], [1.0, -1.0, 2.0])
    out = normalize_log(raw)
    assert list(out['date']) == [pd.Timestamp('2024-01-01'),
                                 pd.Timestamp('2024-02-01'),
                                 pd.Timestamp('2024-03-01')]
    assert out['date'].dt.tz is None


def test_identical_logs_share_a_fingerprint():
    a = _log(['2024-01-01', '2024-01-05'], [1.0, -1.0])
    b = _log(['2024-01-01', '2024-01-05'], [1.0, -1.0])
    c = _log(['2024-01-01', '2024-01-05'], [1.0, -0.5])
    assert log_fingerprint(a) == log_fingerprint(b)
    assert log_fingerprint(a) != log_fingerprint(c)


def test_dedupe_keeps_one_and_reports_the_rest():
    a = _log(['2024-01-01'], [1.0])
    dup = _log(['2024-01-01'], [1.0])
    other = _log(['2024-01-01'], [-1.0])
    kept, dropped = dedupe_logs({'cfg_a': a, 'cfg_b': dup, 'cfg_c': other})
    assert set(kept) == {'cfg_a', 'cfg_c'}
    assert dropped == {'cfg_b': 'cfg_a'}


def test_normalize_log_raises_on_malformed_pnl_r():
    raw = _log(['2024-01-01', '2024-01-02'], [1.0, -1.0])
    raw['pnl_r'] = raw['pnl_r'].astype(object)
    raw.loc[1, 'pnl_r'] = 'oops'
    with pytest.raises(ValueError, match='pnl_r'):
        normalize_log(raw)


def test_normalize_log_allows_preexisting_nan_pnl_r():
    raw = _log(['2024-01-01', '2024-01-02'], [1.0, float('nan')])
    out = normalize_log(raw)
    assert out['pnl_r'].isna().sum() == 1


def test_fingerprint_same_for_entry_time_timestamp_vs_iso_string():
    a = _log(['2024-01-01'], [1.0])
    a['entry_time'] = [pd.Timestamp('2024-01-01 09:30:00')]
    b = _log(['2024-01-01'], [1.0])
    b['entry_time'] = ['2024-01-01T09:30:00']
    assert log_fingerprint(a) == log_fingerprint(b)


def test_fingerprint_same_for_instrument_case_difference():
    a = _log(['2024-01-01'], [1.0], instrument='eurusd')
    b = _log(['2024-01-01'], [1.0], instrument='EURUSD')
    assert log_fingerprint(a) == log_fingerprint(b)


TRAIN_W = Window(
    index=0,
    train_start=pd.Timestamp('2024-01-01'),
    train_end=pd.Timestamp('2024-12-31 23:59:59'),
    test_start=pd.Timestamp('2025-01-01'),
    test_end=pd.Timestamp('2025-03-31 23:59:59'),
)


def test_slice_dates_is_inclusive_on_both_bounds():
    log = _log(['2024-01-01', '2024-06-01', '2025-02-01'], [1.0, 1.0, 1.0])
    out = slice_dates(normalize_log(log), TRAIN_W.train_start, TRAIN_W.train_end)
    assert len(out) == 2


def test_select_applies_the_trade_floor():
    lucky = _log(pd.date_range('2024-02-01', periods=6, freq='7D'), [3.0] * 6)
    real = _log(pd.date_range('2024-01-02', periods=50, freq='5D'),
                [0.4, -1.0] * 25)
    sel = select_config({'lucky': lucky, 'real': real}, TRAIN_W, min_trades=40)
    assert sel.config == 'real'
    assert sel.n_eligible == 1
    assert sel.floor_relaxed is False


def test_select_relaxes_the_floor_and_flags_it():
    only = _log(pd.date_range('2024-01-02', periods=30, freq='9D'), [0.5] * 30)
    sel = select_config({'only': only}, TRAIN_W,
                        min_trades=40, fallback_min_trades=25)
    assert sel.config == 'only'
    assert sel.floor_relaxed is True
    assert sel.floor_used == 25


def test_select_raises_when_nothing_qualifies():
    tiny = _log(pd.date_range('2024-01-02', periods=5, freq='10D'), [1.0] * 5)
    with pytest.raises(ValueError, match='no configuration'):
        select_config({'tiny': tiny}, TRAIN_W,
                      min_trades=40, fallback_min_trades=25)


def test_select_breaks_ties_by_trade_count_then_name():
    many = _log(pd.date_range('2024-01-02', periods=50, freq='5D'), [0.5] * 50)
    few = _log(pd.date_range('2024-01-02', periods=40, freq='6D'), [0.5] * 40)
    sel = select_config({'zz_many': many, 'aa_few': few}, TRAIN_W, min_trades=40)
    assert sel.config == 'zz_many'

    twin_a = _log(pd.date_range('2024-01-02', periods=50, freq='5D'), [0.5] * 50)
    twin_b = _log(pd.date_range('2024-01-03', periods=50, freq='5D'), [0.5] * 50)
    sel2 = select_config({'bbb': twin_b, 'aaa': twin_a}, TRAIN_W, min_trades=40)
    assert sel2.config == 'aaa'


def test_select_uses_one_denominator_for_floor_and_ranking():
    """Unresolved (NaN) R values must not count toward the eligibility floor
    while being skipped by the mean - that mismatch silently tilts selection."""
    rs = [0.5] * 39 + [float('nan')] * 5
    borderline = _log(pd.date_range('2024-01-02', periods=44, freq='6D'), rs)
    with pytest.raises(ValueError, match='no configuration'):
        select_config({'borderline': borderline}, TRAIN_W,
                      min_trades=40, fallback_min_trades=40)


def test_select_ignores_test_period_trades():
    """A config that is terrible in training and brilliant in the test period
    must still lose selection - that is the whole point."""
    bad_then_good = _log(
        list(pd.date_range('2024-01-02', periods=45, freq='6D'))
        + list(pd.date_range('2025-01-05', periods=10, freq='2D')),
        [-0.5] * 45 + [9.0] * 10)
    steady = _log(pd.date_range('2024-01-02', periods=45, freq='6D'), [0.2] * 45)
    sel = select_config({'bad_then_good': bad_then_good, 'steady': steady},
                        TRAIN_W, min_trades=40)
    assert sel.config == 'steady'
