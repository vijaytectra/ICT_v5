# Walk-Forward Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what the Judas Swing selection process would have earned in real time, by re-running the full 113-configuration search inside expanding training windows and only ever scoring a configuration on data chosen without it.

**Architecture:** A model-agnostic engine in `infrastructure/walkforward.py` consumes per-configuration trade logs and performs window generation, deduplication, selection and stitching. A model-specific layer under `models/ict/judas_swing/backtests/` defines the search space, regenerates the logs reproducibly, verifies the no-lookahead assumption against the real model, and runs the report.

**Tech Stack:** Python 3.14, pandas, pytest 9.1.1, existing `infrastructure.stats.compute_stats`, existing `grid.evaluate` / `judas_backtest.execute_signal`.

**Spec:** `docs/superpowers/specs/2026-09-06-walkforward-validation-design.md`

---

## File Structure

**Create:**

| file | responsibility |
|---|---|
| `infrastructure/walkforward.py` | Engine. Windows, dedupe, selection, stitching. No model knowledge, no I/O. |
| `infrastructure/tests/test_walkforward.py` | Engine unit tests on synthetic logs. Fast, no data files. |
| `models/ict/judas_swing/backtests/config_space.py` | The 113-configuration search space as data. Single source of truth. |
| `models/ict/judas_swing/backtests/sweep_configs.py` | Runs every configuration once over the full period, writes `results/configs/<name>.csv`. |
| `models/ict/judas_swing/backtests/verify_no_lookahead.py` | Proves date-slicing a full-period log equals running that period alone. |
| `models/ict/judas_swing/backtests/walkforward_run.py` | Loads logs, runs the engine, prints and saves the report. |

**Modify:** none. Nothing existing changes, so no regression risk to the current backtests.

**Conventions this follows:** engine in `infrastructure/` because it is reusable (CLAUDE.md: "A reusable engine, when extracted, belongs in `infrastructure/`"); runners colocated in the model's `backtests/`; outputs to `backtests/results/` (gitignored); repo root resolved by walking up to `pyproject.toml`.

---

## Task 1: Window generation

**Files:**
- Create: `infrastructure/walkforward.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Create `infrastructure/tests/test_walkforward.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'infrastructure.walkforward'`

- [ ] **Step 3: Write minimal implementation**

Create `infrastructure/walkforward.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: PASS, 4 passed

- [ ] **Step 5: Commit**

```bash
git add infrastructure/walkforward.py infrastructure/tests/test_walkforward.py
git commit -m "Add expanding-window generation for walk-forward validation"
```

---

## Task 2: Log normalisation and deduplication

Two configurations that produce byte-identical trades are one configuration. If 113 names collapse to 90 distinct logs, the search space is smaller than claimed and the selection-bias estimate changes.

**Files:**
- Modify: `infrastructure/walkforward.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `infrastructure/tests/test_walkforward.py`:

```python
from infrastructure.walkforward import dedupe_logs, log_fingerprint, normalize_log


def _log(dates, rs, instrument='EURUSD'):
    """Minimal trade log with the columns the engine and compute_stats need."""
    return pd.DataFrame({
        'date': [pd.Timestamp(d).date().isoformat() for d in dates],
        'pnl_r': list(rs),
        'exit_reason': ['TP' if r > 0 else 'SL' for r in rs],
        'instrument': [instrument] * len(rs),
        'holding_minutes': [60.0] * len(rs),
    })


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'dedupe_logs'`

- [ ] **Step 3: Write minimal implementation**

Add to `infrastructure/walkforward.py` (after the imports, add `import hashlib`):

```python
def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `date` as a naive Timestamp column, sorted ascending.

    Logs are written by several runners; this is the one place that decides what
    a trade log looks like to the engine.
    """
    missing = [c for c in (DATE_COL, R_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"trade log missing required columns: {missing}")
    out = df.copy()
    dates = pd.to_datetime(out[DATE_COL])
    if getattr(dates.dt, 'tz', None) is not None:
        dates = dates.dt.tz_localize(None)
    out[DATE_COL] = dates
    out[R_COL] = pd.to_numeric(out[R_COL], errors='coerce')
    return out.sort_values(DATE_COL).reset_index(drop=True)


def log_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of a log's trade identity, for collapsing duplicates."""
    norm = normalize_log(df)
    cols = [c for c in (DATE_COL, 'instrument', 'direction', 'entry_time', R_COL)
            if c in norm.columns]
    payload = norm[cols].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def dedupe_logs(logs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame],
                                                        dict[str, str]]:
    """Collapse configurations whose trade logs are identical.

    Returns (kept, dropped) where `dropped` maps each removed name to the name
    it duplicates. Names are processed in sorted order so the survivor is
    deterministic.
    """
    kept: dict[str, pd.DataFrame] = {}
    seen: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for name in sorted(logs):
        fp = log_fingerprint(logs[name])
        if fp in seen:
            dropped[name] = seen[fp]
        else:
            seen[fp] = name
            kept[name] = logs[name]
    return kept, dropped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add infrastructure/walkforward.py infrastructure/tests/test_walkforward.py
git commit -m "Add trade-log normalisation and duplicate collapsing"
```

---

## Task 3: Configuration selection

The trade-count floor is the point of this task. Without it the search picks a configuration that took six trades and won five.

**Files:**
- Modify: `infrastructure/walkforward.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `infrastructure/tests/test_walkforward.py`:

```python
from infrastructure.walkforward import select_config, slice_dates

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'Selection'`

- [ ] **Step 3: Write minimal implementation**

Add to `infrastructure/walkforward.py`:

```python
@dataclass(frozen=True)
class Selection:
    """What one training window chose, and on what evidence."""
    window: Window
    config: str
    train_trades: int
    train_expectancy: float
    n_eligible: int
    floor_used: int
    floor_relaxed: bool


def slice_dates(df: pd.DataFrame, start: pd.Timestamp,
                end: pd.Timestamp) -> pd.DataFrame:
    """Rows whose date falls in [start, end]. Assumes a normalised log."""
    mask = (df[DATE_COL] >= start) & (df[DATE_COL] <= end)
    return df.loc[mask]


def select_config(logs: dict[str, pd.DataFrame], window: Window,
                  min_trades: int = 40,
                  fallback_min_trades: int = 25) -> Selection:
    """Pick the best configuration using ONLY the window's training period.

    Ranked by expectancy (mean R per trade) among configurations clearing the
    trade floor. The floor exists because expectancy on a handful of trades is
    noise, and an unfloored search reliably selects that noise.

    If nothing clears `min_trades` the floor drops to `fallback_min_trades` and
    the Selection records that it happened. If nothing clears that either, this
    raises rather than quietly returning a fluke.
    """
    stats: dict[str, tuple[int, float]] = {}
    for name, log in logs.items():
        train = slice_dates(normalize_log(log), window.train_start, window.train_end)
        # Drop unresolved trades ONCE, before both the count and the mean, so the
        # eligibility floor and the ranking share a denominator by construction.
        # `normalize_log` guarantees no silently-introduced NaN, but not "no NaN":
        # a genuinely missing R value is legitimate. Without this line `len()`
        # would count such a row toward the floor while `.mean()` skips it, which
        # silently tilts which configuration wins.
        train = train.dropna(subset=[R_COL])
        if len(train):
            stats[name] = (len(train), float(train[R_COL].mean()))

    for floor in (min_trades, fallback_min_trades):
        eligible = {n: v for n, v in stats.items() if v[0] >= floor}
        if eligible:
            # expectancy desc, then trade count desc, then name asc
            best = min(eligible.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))
            name, (n, exp) = best
            return Selection(
                window=window, config=name, train_trades=n,
                train_expectancy=exp, n_eligible=len(eligible),
                floor_used=floor, floor_relaxed=(floor != min_trades),
            )

    raise ValueError(
        f"window {window.index}: no configuration reached {fallback_min_trades} "
        f"training trades (best was {max((v[0] for v in stats.values()), default=0)})"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: PASS, 13 passed

- [ ] **Step 5: Commit**

```bash
git add infrastructure/walkforward.py infrastructure/tests/test_walkforward.py
git commit -m "Add floored expectancy selection for walk-forward windows"
```

---

## Task 4: Test-slice collection with a leak guard

**Files:**
- Modify: `infrastructure/walkforward.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `infrastructure/tests/test_walkforward.py`:

```python
from infrastructure.walkforward import collect_test_trades


def test_collect_returns_only_test_period_trades():
    log = _log(pd.date_range('2024-06-01', periods=100, freq='3D'), [0.1] * 100)
    out = collect_test_trades(log, TRAIN_W)
    assert len(out) > 0
    assert (out['date'] > TRAIN_W.train_end).all()
    assert (out['date'] >= TRAIN_W.test_start).all()
    assert (out['date'] <= TRAIN_W.test_end).all()


def test_collect_returns_empty_frame_when_config_did_not_trade():
    log = _log(pd.date_range('2024-02-01', periods=10, freq='3D'), [0.1] * 10)
    out = collect_test_trades(log, TRAIN_W)
    assert len(out) == 0
    assert list(out.columns) == list(normalize_log(log).columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'collect_test_trades'`

- [ ] **Step 3: Write minimal implementation**

Add to `infrastructure/walkforward.py`:

```python
def collect_test_trades(log: pd.DataFrame, window: Window) -> pd.DataFrame:
    """The chosen configuration's trades inside the test period only.

    The assertion is the whole safety net of this module: if a training-period
    trade ever reaches the stitched result, every downstream number is inflated
    and nothing warns you. Cheap to check, catastrophic to miss.
    """
    out = slice_dates(normalize_log(log), window.test_start, window.test_end)
    if len(out) and not (out[DATE_COL] > window.train_end).all():
        raise AssertionError(
            f"window {window.index}: test slice contains training dates"
        )
    return out.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: PASS, 15 passed

- [ ] **Step 5: Commit**

```bash
git add infrastructure/walkforward.py infrastructure/tests/test_walkforward.py
git commit -m "Add test-slice collection with training-leak assertion"
```

---

## Task 5: Orchestration and stitching

**Files:**
- Modify: `infrastructure/walkforward.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `infrastructure/tests/test_walkforward.py`:

```python
from infrastructure.walkforward import run_walkforward


def test_walkforward_exposes_a_config_that_only_worked_in_training():
    """The behaviour that justifies this whole module.

    `mirage` looks best in every training window and loses in every test
    window. A naive full-sample backtest would crown it. Walk-forward must
    select it and then report the losses it actually produced.
    """
    dates = pd.date_range('2024-04-01', '2026-09-01', freq='3D')
    mirage_r = [2.0 if d < pd.Timestamp('2025-04-01') else -1.0 for d in dates]
    steady_r = [0.05] * len(dates)
    logs = {'mirage': _log(dates, mirage_r), 'steady': _log(dates, steady_r)}

    stitched, selections = run_walkforward(logs, '2024-04-05', '2026-09-04',
                                           min_trades=40)

    assert len(selections) == 6
    assert selections[0].config == 'mirage'
    assert len(stitched) > 0
    assert stitched['pnl_r'].sum() < 0


def test_walkforward_stitched_periods_are_disjoint():
    dates = pd.date_range('2024-04-01', '2026-09-01', freq='3D')
    logs = {'a': _log(dates, [0.1] * len(dates)),
            'b': _log(dates, [0.2] * len(dates))}
    stitched, selections = run_walkforward(logs, '2024-04-05', '2026-09-04',
                                           min_trades=40)
    assert stitched['window'].is_monotonic_increasing
    for w in selections:
        rows = stitched[stitched['window'] == w.window.index]
        assert (rows['date'] > w.window.train_end).all()


def test_walkforward_is_deterministic():
    dates = pd.date_range('2024-04-01', '2026-09-01', freq='3D')
    logs = {'a': _log(dates, [0.1] * len(dates)),
            'b': _log(dates, [0.1] * len(dates)),
            'c': _log(dates, [0.2] * len(dates))}
    r1 = run_walkforward(logs, '2024-04-05', '2026-09-04', min_trades=40)
    r2 = run_walkforward(logs, '2024-04-05', '2026-09-04', min_trades=40)
    assert [s.config for s in r1[1]] == [s.config for s in r2[1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_walkforward'`

- [ ] **Step 3: Write minimal implementation**

Add to `infrastructure/walkforward.py`:

```python
def run_walkforward(logs: dict[str, pd.DataFrame], first_date, last_date,
                    min_train_quarters: int = 4, test_quarters: int = 1,
                    min_trades: int = 40,
                    fallback_min_trades: int = 25
                    ) -> tuple[pd.DataFrame, list[Selection]]:
    """Select inside each training window, trade the winner forward, stitch.

    Returns (stitched_trades, selections). The stitched frame carries a
    `window` column so per-window results stay recoverable; it is the honest
    result and is returned whatever it says.
    """
    windows = make_windows(first_date, last_date,
                           min_train_quarters=min_train_quarters,
                           test_quarters=test_quarters)
    selections: list[Selection] = []
    frames: list[pd.DataFrame] = []
    for window in windows:
        sel = select_config(logs, window, min_trades=min_trades,
                            fallback_min_trades=fallback_min_trades)
        selections.append(sel)
        part = collect_test_trades(logs[sel.config], window)
        if len(part):
            part = part.copy()
            part['window'] = window.index
            part['config'] = sel.config
            frames.append(part)

    stitched = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=[DATE_COL, R_COL, 'window', 'config']))
    return stitched, selections
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py -v`
Expected: PASS, 18 passed

- [ ] **Step 5: Commit**

```bash
git add infrastructure/walkforward.py infrastructure/tests/test_walkforward.py
git commit -m "Add walk-forward orchestration and stitching"
```

---

## Task 6: Reproducible configuration search space

The 113 logs currently exist only in a session scratchpad. They must be regenerable from committed code or the result cannot be reproduced.

**Files:**
- Create: `models/ict/judas_swing/backtests/config_space.py`
- Test: `infrastructure/tests/test_walkforward.py`

- [ ] **Step 1: Write the failing test**

Append to `infrastructure/tests/test_walkforward.py`:

```python
def test_config_space_is_well_formed():
    import sys, pathlib
    bt = pathlib.Path(__file__).resolve().parents[2] / 'models/ict/judas_swing/backtests'
    sys.path.insert(0, str(bt))
    from config_space import CONFIG_SPACE, EXIT_KEYS, signal_key

    assert len(CONFIG_SPACE) >= 100
    assert all(isinstance(v, dict) for v in CONFIG_SPACE.values())
    assert 'baseline' in CONFIG_SPACE
    assert CONFIG_SPACE['baseline'] == {}
    # names become filenames
    assert all('/' not in k and '\\' not in k for k in CONFIG_SPACE)
    # grouping by signal key must actually save work
    groups = {signal_key(v) for v in CONFIG_SPACE.values()}
    assert len(groups) < len(CONFIG_SPACE)
    # exit-only keys must never appear in a signal key
    for cfg in CONFIG_SPACE.values():
        assert not (set(signal_key(cfg)) & EXIT_KEYS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py::test_config_space_is_well_formed -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config_space'`

- [ ] **Step 3: Write minimal implementation**

Create `models/ict/judas_swing/backtests/config_space.py`:

```python
"""The configuration search space, as data.

This is the single source of truth for what was searched. The walk-forward
result is only meaningful if each training window re-runs *this* space - the
same one that produced the original selection - so it lives in one place and is
imported by both the sweep runner and the walk-forward runner.

Keys become filenames, so they carry no path separators.
"""
from __future__ import annotations

import itertools

# Changing any of these alters exit management only; the signals a
# configuration produces are identical. Grouping on the complement lets the
# sweep generate signals once per group instead of once per configuration.
EXIT_KEYS = frozenset({'be_trigger_r', 'partials', 'partial_be'})


def signal_key(cfg: dict) -> tuple:
    """Hashable identity of the signal-affecting part of a configuration."""
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()
                        if k not in EXIT_KEYS))


def _build() -> dict[str, dict]:
    space: dict[str, dict] = {'baseline': {}}

    # -- exit management (one signal config, many exits) --------------------
    for be in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        space[f'be_{be}R'] = {'be_trigger_r': be}
    for frac, lvl in itertools.product((0.25, 0.334, 0.5), (0.75, 1.0, 1.5, 2.0)):
        space[f'part{int(frac * 100)}@{lvl}R+BE'] = {
            'partials': [[frac, lvl]], 'partial_be': True}
    for frac, lvl in ((0.5, 1.0), (0.5, 1.5), (0.334, 1.0)):
        space[f'part{int(frac * 100)}@{lvl}R_noBE'] = {
            'partials': [[frac, lvl]], 'partial_be': False}
    space['ladder33@1+33@2'] = {'partials': [[0.334, 1.0], [0.334, 2.0]]}
    space['ladder50@1+25@2'] = {'partials': [[0.5, 1.0], [0.25, 2.0]]}

    # -- targets -------------------------------------------------------------
    for rr in (1.5, 2.0, 3.0, 3.5, 4.0):
        space[f'liq_rr{rr}'] = {'target_min_rr': rr}
    for rr in (2.0, 2.5, 3.0):
        space[f'adr_rr{rr}'] = {'target_mode': 'adr', 'target_min_rr': rr}
    for rr in (2.0, 2.5):
        space[f'rngend_rr{rr}'] = {'target_mode': 'range_end', 'target_min_rr': rr}
    for p in (0.6, 0.75, 1.0):
        space[f'adrcap{p}'] = {'adr_target_pct': p}
    space['no_adr_cap'] = {'cap_target_at_adr': False}

    # -- entries -------------------------------------------------------------
    space['market'] = {'entry_style': 'market'}
    space['limit_edge'] = {'fvg_entry_depth': 'edge'}
    space['fvg_only'] = {'entry_arrays': ['fvg']}
    space['ob_only'] = {'entry_arrays': ['ob']}
    space['nokz_fill'] = {'limit_fill_to_killzone': False}
    for m in (0.0, 0.15, 0.4):
        space[f'minfvg{m}'] = {'min_fvg_atr_mult': m}
    for f in (60, 240):
        space[f'fill{f}m'] = {'entry_fill_minutes': f}

    # -- gates / trade count -------------------------------------------------
    for b in ('order_flow', 'draw', 'week_open'):
        space[f'bias_{b}'] = {'bias_mode': b}
    space['no_bias'] = {'use_daily_bias': False}
    space['multi_per_day'] = {'one_trade_per_day': False}
    space['no_veto_both'] = {'veto_both_sides_swept': False}
    space['no_prior_deliv'] = {'skip_if_prior_session_delivered': 2.0}
    space['no_range_cap'] = {'max_range_used_pct': 2.0}
    space['deep_judas_allow'] = {'deep_judas_action': 'allow'}
    space['rev_90m'] = {'max_minutes_to_reversal': 90}
    space['conf_cisd'] = {'confirmation': 'cisd'}
    space['conf_mss'] = {'confirmation': 'mss'}
    space['no_disp_qual'] = {'require_displacement_quality': False}
    space['closeback_8'] = {'close_back_within_bars': 8}
    space['skip_friday'] = {'skip_weekdays': [4]}
    space['stop_adr_0p5'] = {'max_stop_adr_pct': 0.5}

    # -- stacks (the combinations that produced the reported winner) ---------
    OF = {'bias_mode': 'order_flow'}
    MF0 = {'min_fvg_atr_mult': 0.0}
    R90 = {'max_minutes_to_reversal': 90}
    EDGE = {'fvg_entry_depth': 'edge'}
    MSS = {'confirmation': 'mss'}
    FRI = {'skip_weekdays': [4]}
    stacks = {
        'OF': OF,
        'OF+mf0': {**OF, **MF0},
        'OF+mf0+r90': {**OF, **MF0, **R90},
        'OF+mf0+edge': {**OF, **MF0, **EDGE},
        'OF+mf0+r90+edge': {**OF, **MF0, **R90, **EDGE},
        'OF+mf0+r90+edge+Fri': {**OF, **MF0, **R90, **EDGE, **FRI},
        'OF+mss': {**OF, **MSS},
        'OF+mf0+mss': {**OF, **MF0, **MSS},
        'mf0+edge': {**MF0, **EDGE},
        'mf0+edge+Fri': {**MF0, **EDGE, **FRI},
    }
    exit_variants = {
        '': {},
        '|be1.25': {'be_trigger_r': 1.25},
        '|be1.5': {'be_trigger_r': 1.5},
        '|p25@1.5+be': {'partials': [[0.25, 1.5]], 'partial_be': True},
    }
    for sname, sov in stacks.items():
        for ename, eov in exit_variants.items():
            space[sname + ename] = {**sov, **eov}

    return space


CONFIG_SPACE: dict[str, dict] = _build()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/test_walkforward.py::test_config_space_is_well_formed -v`
Expected: PASS

Then confirm the size matches what was searched:

Run: `.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'models/ict/judas_swing/backtests'); from config_space import CONFIG_SPACE, signal_key; print(len(CONFIG_SPACE), 'configs;', len({signal_key(v) for v in CONFIG_SPACE.values()}), 'signal groups')"`
Expected: roughly `113 configs; 56 signal groups` (exact counts may differ by a few; record the actual numbers in the final report)

- [ ] **Step 5: Commit**

```bash
git add models/ict/judas_swing/backtests/config_space.py infrastructure/tests/test_walkforward.py
git commit -m "Add the Judas configuration search space as committed data"
```

---

## Task 7: Sweep runner — regenerate every configuration's trade log

**Files:**
- Create: `models/ict/judas_swing/backtests/sweep_configs.py`

- [ ] **Step 1: Write the runner**

Create `models/ict/judas_swing/backtests/sweep_configs.py`:

```python
#!/usr/bin/env python3
"""Generate one full-period trade log per configuration in CONFIG_SPACE.

Signals are generated once per signal group and reused across that group's exit
variants, because exit management cannot change which setups fire. That is what
makes a 113-configuration sweep affordable.

Outputs: results/configs/<config name>.csv  (gitignored)

Usage:
  .venv/Scripts/python.exe models/ict/judas_swing/backtests/sweep_configs.py
"""
from __future__ import annotations

import copy
import pathlib
import sys
import time
import multiprocessing as mp

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

FX = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']
OUT = _HERE / 'results' / 'configs'
COSTS = {'spread_mode': 'historical', 'gap_fill': 'open'}

_W: dict = {}


def _init():
    import warnings, yaml
    warnings.filterwarnings('ignore')
    from grid import load_snapshots
    from judas_backtest import DOLLARS_PER_POINT
    base = yaml.safe_load(open(_HERE / 'configs' / 'judas_fx.yaml'))
    base['costs'] = COSTS
    _W.update(load_snapshots=load_snapshots, dpp=DOLLARS_PER_POINT,
              base=base, snaps={})


def _snaps(inst, tf):
    if (inst, tf) not in _W['snaps']:
        _W['snaps'][(inst, tf)] = _W['load_snapshots'](
            inst, _W['base']['start_date'], None, tf, verbose=False)
    return _W['snaps'][(inst, tf)]


def _signals(args):
    inst, overrides = args
    from grid import BACKTEST_KEYS
    from judas_backtest import JudasSwingModel, apply_pip_scaling
    cfg = {**copy.deepcopy(_W['base']), **overrides}
    tf = cfg.get('entry_timeframe', '5min')
    model = JudasSwingModel(config=apply_pip_scaling(
        {k: v for k, v in cfg.items() if k not in BACKTEST_KEYS}, inst))
    directions = (None,) if cfg.get('use_daily_bias', True) else ('bullish', 'bearish')
    one_per_day = bool(cfg.get('one_trade_per_day', True))
    out = []
    for day, snap in _snaps(inst, tf):
        traded = False
        for session in model.config['sessions']:
            if traded and one_per_day:
                break
            for forced in directions:
                try:
                    sig = model.generate_signal(snap, session=session,
                                                force_direction=forced)
                except Exception:
                    continue
                if not sig['actionable']:
                    continue
                out.append((day, sig))
                traded = True
                break
    return inst, out


def _execute(args):
    inst, sigs, overrides = args
    import pandas as pd
    from judas_backtest import execute_signal, apply_pip_scaling_costs
    from infrastructure.execution import cost_model_from_config
    cfg = {**copy.deepcopy(_W['base']), **overrides}
    tf = cfg.get('entry_timeframe', '5min')
    cost = cost_model_from_config(apply_pip_scaling_costs(cfg, inst))
    eod_h = cfg.get('eod_exit_hour')
    snapmap = dict(_snaps(inst, tf))
    rows = []
    for day, sig in sigs:
        snap = snapmap.get(day)
        if snap is None:
            continue
        r = execute_signal(snap, sig, pd.Timestamp(day), cost=cost,
                           usd_per_point=_W['dpp'].get(inst, 1.0),
                           eod_exit_hour=int(eod_h) if eod_h is not None else None,
                           be_trigger_r=float(cfg.get('be_trigger_r', 0.0)),
                           partials=cfg.get('partials') or None,
                           partial_be=bool(cfg.get('partial_be', True)))
        if r is None:
            continue
        r.pop('_fill', None)
        r.pop('_exit', None)
        r['date'] = day
        r['instrument'] = inst
        rows.append(r)
    return inst, rows


def main() -> int:
    import pandas as pd
    from config_space import CONFIG_SPACE, signal_key

    OUT.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[str]] = {}
    for name, cfg in CONFIG_SPACE.items():
        groups.setdefault(signal_key(cfg), []).append(name)
    print(f"{len(CONFIG_SPACE)} configs in {len(groups)} signal groups", flush=True)

    t0 = time.time()
    pool = mp.Pool(processes=min(6, mp.cpu_count()), initializer=_init)
    done = 0
    for gi, (_, names) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0]), 1):
        sig_overrides = {k: v for k, v in CONFIG_SPACE[names[0]].items()
                         if k not in ('be_trigger_r', 'partials', 'partial_be')}
        sigs = dict(pool.map(_signals, [(i, sig_overrides) for i in FX]))
        for name in names:
            res = pool.map(_execute, [(i, sigs[i], CONFIG_SPACE[name]) for i in FX])
            df = pd.DataFrame([r for _, rr in res for r in rr])
            df.to_csv(OUT / f'{name}.csv', index=False)
            done += 1
        print(f"[{time.time()-t0:6.0f}s] group {gi}/{len(groups)}  "
              f"{done}/{len(CONFIG_SPACE)} configs written", flush=True)
    pool.close()
    pool.join()
    print(f"[{time.time()-t0:6.0f}s] done -> {OUT}")
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe models/ict/judas_swing/backtests/sweep_configs.py`
Expected: prints group progress; finishes in roughly 2–3 hours; writes ~113 CSVs into `models/ict/judas_swing/backtests/results/configs/`

- [ ] **Step 3: Verify the output**

Run: `.venv/Scripts/python.exe -c "import pathlib,pandas as pd; d=pathlib.Path('models/ict/judas_swing/backtests/results/configs'); f=sorted(d.glob('*.csv')); print(len(f),'logs'); b=pd.read_csv(d/'baseline.csv'); print('baseline trades:',len(b))"`
Expected: `113 logs` (or the count Task 6 reported) and `baseline trades: 73`

The baseline count is the regression check: if it is not 73, the regenerated logs disagree with today's run and nothing downstream can be trusted.

- [ ] **Step 4: Commit**

```bash
git add models/ict/judas_swing/backtests/sweep_configs.py
git commit -m "Add reproducible configuration sweep runner"
```

---

## Task 8: Verify the no-lookahead assumption

The entire shortcut rests on this. If it fails, the design is void.

**Files:**
- Create: `models/ict/judas_swing/backtests/verify_no_lookahead.py`

- [ ] **Step 1: Write the verification**

Create `models/ict/judas_swing/backtests/verify_no_lookahead.py`:

```python
#!/usr/bin/env python3
"""Prove that slicing a full-period trade log by date is equivalent to running
that period alone.

Walk-forward here reuses full-period logs and slices them per window, which is
only legitimate if the model never uses future data. It should not - bias comes
from prior days and ADR from a trailing five-day window - but "should not" is
not evidence. This runs two windows both ways and compares.

Exit code 0 means the shortcut is sound. Non-zero means the walk-forward result
is void and every window must be re-run through the model.

Usage:
  .venv/Scripts/python.exe models/ict/judas_swing/backtests/verify_no_lookahead.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from infrastructure.walkforward import make_windows, normalize_log, slice_dates  # noqa: E402

CONFIGS = ['baseline', 'OF+mf0+r90|be1.25']
LOGS = _HERE / 'results' / 'configs'
COMPARE = ['date', 'instrument', 'direction', 'entry_time', 'pnl_r']


def _run_period(config_name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Run one configuration over a restricted period, from scratch."""
    import copy, yaml, warnings
    warnings.filterwarnings('ignore')
    from grid import load_snapshots, evaluate, BACKTEST_KEYS  # noqa: F401
    from judas_backtest import DOLLARS_PER_POINT
    from config_space import CONFIG_SPACE

    base = yaml.safe_load(open(_HERE / 'configs' / 'judas_fx.yaml'))
    base['costs'] = {'spread_mode': 'historical', 'gap_fill': 'open'}
    cfg = {**copy.deepcopy(base), **CONFIG_SPACE[config_name]}
    cfg['start_date'] = start.date().isoformat()

    frames = []
    for inst in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD']:
        snaps = load_snapshots(inst, cfg['start_date'], end.date().isoformat(),
                               cfg.get('entry_timeframe', '5min'), verbose=False)
        t = evaluate(snaps, cfg, DOLLARS_PER_POINT.get(inst, 1.0), inst)
        if len(t):
            t = t.copy()
            t['instrument'] = inst
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    windows = make_windows('2024-04-05', '2026-09-04')
    checks = [windows[0], windows[-1]]
    failures = 0

    for config_name in CONFIGS:
        path = LOGS / f'{config_name}.csv'
        if not path.exists():
            print(f"MISSING {path} - run sweep_configs.py first")
            return 2
        full = normalize_log(pd.read_csv(path))

        for w in checks:
            sliced = slice_dates(full, w.test_start, w.test_end)
            fresh = _run_period(config_name, w.test_start, w.test_end)
            fresh = normalize_log(fresh) if len(fresh) else fresh

            n_s, n_f = len(sliced), len(fresh)
            if n_s != n_f:
                print(f"FAIL {config_name} window {w.index}: "
                      f"sliced {n_s} trades, fresh run {n_f}")
                failures += 1
                continue

            if n_s:
                cols = [c for c in COMPARE if c in sliced.columns and c in fresh.columns]
                a = sliced[cols].sort_values(cols).reset_index(drop=True)
                b = fresh[cols].sort_values(cols).reset_index(drop=True)
                if not a.equals(b):
                    diff = a.compare(b)
                    print(f"FAIL {config_name} window {w.index}: "
                          f"{len(diff)} differing rows")
                    print(diff.head(10).to_string())
                    failures += 1
                    continue

            print(f"OK   {config_name} window {w.index}: {n_s} trades match")

    if failures:
        print(f"\nVERIFICATION FAILED ({failures}) - the walk-forward shortcut is "
              f"NOT valid. Re-run each window through the model instead.")
        return 1
    print("\nVERIFICATION PASSED - date-slicing is equivalent to running the "
          "period alone.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run the verification**

Run: `.venv/Scripts/python.exe models/ict/judas_swing/backtests/verify_no_lookahead.py`
Expected: four `OK` lines and `VERIFICATION PASSED`, exit code 0

If it fails, **stop**. Do not continue to Task 9. Report the mismatch and re-plan: the fallback is re-running every configuration inside every window (~10 hours), which needs its own decision from the user.

Note: a restricted run starts with less warm-up history than the full-period run, so the first few sessions of the sliced window may legitimately differ. If failures are confined to the first two session-days of the window, re-run comparing only trades dated at least five sessions after `test_start`, and record that carve-out in the final report rather than hiding it.

- [ ] **Step 3: Commit**

```bash
git add models/ict/judas_swing/backtests/verify_no_lookahead.py
git commit -m "Add no-lookahead verification for the walk-forward shortcut"
```

---

## Task 9: Walk-forward runner and report

**Files:**
- Create: `models/ict/judas_swing/backtests/walkforward_run.py`

- [ ] **Step 1: Write the runner**

Create `models/ict/judas_swing/backtests/walkforward_run.py`:

```python
#!/usr/bin/env python3
"""Run walk-forward validation over the configuration logs and report.

Reports the stitched result as-is. There is deliberately no option to drop a
window, change the window size, or re-rank after seeing the answer - those are
exactly the moves that would turn this back into the in-sample number it exists
to replace.

Usage:
  .venv/Scripts/python.exe models/ict/judas_swing/backtests/walkforward_run.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents
             if (p / 'pyproject.toml').exists())
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import infrastructure.stats as stats_mod                       # noqa: E402
from infrastructure.walkforward import (dedupe_logs, normalize_log,  # noqa: E402
                                        run_walkforward)

LOGS = _HERE / 'results' / 'configs'
OUT = _HERE / 'results' / 'walkforward'
FIRST, LAST = '2024-04-05', '2026-09-04'


def _fmt(stats: dict) -> str:
    return (f"n={stats['total_trades']:>4}  win={stats['win_pct']:>5.1f}%  "
            f"PF={stats['profit_factor']:>6.3f}  expR={stats['expectancy_r']:>+7.3f}  "
            f"totR={stats['total_r']:>+8.2f}  maxDD={stats['max_dd_r']:>6.2f}")


def main() -> int:
    paths = sorted(LOGS.glob('*.csv'))
    if not paths:
        print(f"no logs in {LOGS} - run sweep_configs.py first")
        return 2
    logs = {p.stem: pd.read_csv(p) for p in paths}
    print(f"loaded {len(logs)} configuration logs")

    kept, dropped = dedupe_logs(logs)
    print(f"distinct configurations: {len(kept)}  (collapsed {len(dropped)} duplicates)")

    stitched, selections = run_walkforward(kept, FIRST, LAST, min_trades=40)

    print("\n### PER-WINDOW SELECTIONS (chosen on training data only)")
    print(f"  {'#':>2} {'train end':<12} {'test':<24} {'chosen config':<28} "
          f"{'trainN':>6} {'trainExp':>9} {'elig':>5} {'testN':>6} {'testR':>8}")
    rows = []
    for sel in selections:
        w = sel.window
        part = stitched[stitched['window'] == w.index]
        flag = '  [FLOOR RELAXED]' if sel.floor_relaxed else ''
        print(f"  {w.index:>2} {w.train_end.date()!s:<12} "
              f"{str(w.test_start.date()) + '..' + str(w.test_end.date()):<24} "
              f"{sel.config:<28} {sel.train_trades:>6} {sel.train_expectancy:>+9.3f} "
              f"{sel.n_eligible:>5} {len(part):>6} {part['pnl_r'].sum():>+8.2f}{flag}")
        rows.append({'window': w.index, 'train_end': w.train_end.date(),
                     'test_start': w.test_start.date(), 'test_end': w.test_end.date(),
                     'config': sel.config, 'train_trades': sel.train_trades,
                     'train_expectancy': sel.train_expectancy,
                     'n_eligible': sel.n_eligible, 'floor_used': sel.floor_used,
                     'floor_relaxed': sel.floor_relaxed,
                     'test_trades': len(part), 'test_r': float(part['pnl_r'].sum())})

    distinct = len({s.config for s in selections})
    print(f"\n  distinct winners across {len(selections)} windows: {distinct}"
          f"  ({'stable' if distinct <= 2 else 'THRASHING'})")

    print("\n### HONEST WALK-FORWARD RESULT")
    wf = stats_mod.compute_stats(stitched)
    print("  " + _fmt(wf))

    print("\n### COMPARISON")
    print(f"  {'walk-forward (honest)':<34} " + _fmt(wf))
    for label, name in [('in-sample winner', 'OF+mf0+r90|be1.25'),
                        ('current production config', 'baseline')]:
        if name in kept:
            print(f"  {label + ' (' + name + ')':<34} "
                  + _fmt(stats_mod.compute_stats(normalize_log(kept[name]))))

    pf = wf['profit_factor']
    verdict = ('EDGE SURVIVES honest selection' if pf > 1.3 else
               'WEAK BUT REAL - partly selection bias' if pf >= 1.0 else
               'SELECTION BIAS - the FX result does not survive')
    print(f"\n### VERDICT (thresholds fixed in the spec before this run)")
    print(f"  walk-forward PF = {pf:.3f}  ->  {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    stitched.to_csv(OUT / 'walkforward_trades.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'walkforward_windows.csv', index=False)
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe models/ict/judas_swing/backtests/walkforward_run.py`
Expected: per-window table, the honest result, the comparison, and a verdict line. Runs in seconds.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest infrastructure/tests/ -v`
Expected: all pass, including the pre-existing `test_execution.py` and `test_registry_coverage.py`

- [ ] **Step 4: Commit**

```bash
git add models/ict/judas_swing/backtests/walkforward_run.py
git commit -m "Add walk-forward runner and report"
```

---

## Task 10: Write the analysis document

**Files:**
- Create: `models/ict/judas_swing/analysis/judas-swing-walkforward-2026-09-06.md`
- Modify: `analysis/README.md` (add a row to the index table)

- [ ] **Step 1: Write the analysis doc**

Follow `analysis/README.md` exactly. Required frontmatter block:

```markdown
# Judas Swing — Walk-Forward Validation

**Model**: judas-swing
**Concept**: entries
**Instrument**: multi (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD)
**Analysis date**: 2026-09-06
**Dataset**: 1m cleaned CSVs, 2024-04-01 → 2026-09-04, 5m entry timeframe
**Scripts**: `models/ict/judas_swing/backtests/sweep_configs.py`,
`models/ict/judas_swing/backtests/walkforward_run.py`,
`models/ict/judas_swing/backtests/verify_no_lookahead.py`,
`infrastructure/walkforward.py`
**Raw results**: `models/ict/judas_swing/backtests/results/walkforward/`
**Configs**: `models/ict/judas_swing/backtests/config_space.py`
```

Then the required sections: **What Was Tested**, **Results**, **Key Findings** (numbered), **Recommendations** (with priority), **Open Questions**.

Content that must appear:
- The selection-bias problem this exists to measure, stated plainly
- The verification result from Task 8, including any warm-up carve-out
- The per-window selection table, including whether winners thrashed
- The honest walk-forward number against the in-sample 1.487 and contaminated 1.418
- The verdict against the spec's pre-registered thresholds
- The index-CFD holdout failure (PF 0.834, gross 0.976) as context
- If the result is negative, say so in the first paragraph

- [ ] **Step 2: Add the index row**

Modify `analysis/README.md`, appending to the index table:

```markdown
| [judas-swing-walkforward-2026-09-06.md](../models/ict/judas_swing/analysis/judas-swing-walkforward-2026-09-06.md) | judas-swing | entries | 2026-09-06 | Walk-forward validation of the 113-config search; <one-line result> |
```

- [ ] **Step 3: Commit**

```bash
git add models/ict/judas_swing/analysis/judas-swing-walkforward-2026-09-06.md analysis/README.md
git commit -m "Add walk-forward validation analysis"
```

---

## Self-Review Notes

**Spec coverage check:**

| spec requirement | task |
|---|---|
| Full 113-config re-search per window | 6, 7 |
| Expanding train, quarterly test, min 4 quarters | 1 |
| Expectancy R with 40-trade floor | 3 |
| Floor relaxation to 25, recorded not silent | 3, 9 |
| Tie-break by trade count then name | 3 |
| Zero-test-trade window counted not dropped | 5 |
| Duplicate collapsing by trade-log hash | 2 |
| Fail loudly on missing/corrupt log | 7, 8, 9 |
| No-lookahead verification | 8 |
| Leak test | 4 |
| Determinism test | 5 |
| Duplicate report | 9 |
| Stitched result output | 5, 9 |
| Per-window winners output | 9 |
| Comparison table | 9 |
| Pre-registered verdict thresholds | 9 |
| Honest-reporting rule | 9 (no window-drop option exists in the code) |

**Known risk:** Task 8 may fail on warm-up differences at window starts. The mitigation and its disclosure requirement are written into Task 8 Step 2 rather than left to be discovered.
