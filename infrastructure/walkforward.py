"""Walk-forward validation.

Model-agnostic. Consumes per-configuration trade logs and answers one question:
what would this selection process have earned in real time?

The engine never sees a model. It sees trade logs keyed by configuration name,
each carrying a `date` column and a `pnl_r` column. Everything here is pure -
no file I/O, no global state - so it can be tested on synthetic logs.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DATE_COL = 'date'
R_COL = 'pnl_r'


@dataclass(frozen=True)
class Window:
    """One train/test split. Both bounds are inclusive."""
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(first_date, last_date, min_train_quarters: int = 4,
                 test_quarters: int = 1) -> list[Window]:
    """Expanding-train, fixed-length-test quarterly windows.

    Training always starts at the first quarter and grows; only the test period
    slides. That mirrors re-tuning on the whole history you actually have.
    """
    quarters = pd.period_range(pd.Timestamp(first_date),
                               pd.Timestamp(last_date), freq='Q')
    windows: list[Window] = []
    for i in range(min_train_quarters, len(quarters), test_quarters):
        test_qs = quarters[i:i + test_quarters]
        if len(test_qs) < test_quarters:
            break
        windows.append(Window(
            index=len(windows),
            train_start=quarters[0].start_time,
            train_end=quarters[i - 1].end_time,
            test_start=test_qs[0].start_time,
            test_end=test_qs[-1].end_time,
        ))
    return windows
