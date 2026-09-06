"""Walk-forward engine tests. Synthetic logs only - no data files needed."""
import pandas as pd
import pytest

from infrastructure.walkforward import (Window, dedupe_logs, log_fingerprint,
                                        make_windows, normalize_log)


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
