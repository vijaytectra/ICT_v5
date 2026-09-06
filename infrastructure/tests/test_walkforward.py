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
