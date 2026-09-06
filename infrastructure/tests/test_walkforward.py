"""Walk-forward engine tests. Synthetic logs only - no data files needed."""
import pandas as pd
import pytest

from infrastructure.walkforward import Window, make_windows


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
