# Walk-Forward Validation — Design

**Date**: 2026-09-06
**Model**: judas-swing (FX basket: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD)
**Status**: design approved, not implemented

---

## Problem

Today's parameter study tested ~130 configurations of the Judas Swing model and
selected the best one by its out-of-sample score. That makes the out-of-sample
number dishonest: the held-out period was consulted 130 times, so it functioned
as a second training set.

The current claims are therefore unvalidated:

| metric | value | status |
|---|---|---|
| in-sample PF | 1.487 (195 trades) | selection-contaminated |
| "out-of-sample" PF | 1.418 (60 trades) | selection-contaminated |
| index-CFD holdout PF | 0.834 (80 trades) | genuinely clean — **failed** |

The index holdout failed even at zero execution cost (gross PF 0.976), so the
strategy has no edge on index CFDs. That test could not rescue the FX result,
and no test run so far can distinguish "the FX edge is real" from "I fitted 130
configurations to 195 trades".

Walk-forward validation is the test that can, because it never lets a
configuration be scored on data that was used to choose it.

## Goal

Produce one number: **what the full selection process would have earned in real
time**, and report it as-is, including if it is negative.

Non-goal: producing a tight confidence interval. ~109 forward trades cannot do
that. This answers whether the edge survives honest selection, not its size.

---

## Design decisions

All three were chosen deliberately; each rejected alternative is recorded.

### 1. Re-selection scope — full search per window

Every window re-runs the **entire 113-configuration search**, then trades the
winner forward.

*Rejected*: re-selecting only the four winning axes (bias_mode, min_fvg_atr_mult,
max_minutes_to_reversal, be_trigger_r). It understates the bias, because it
pretends the other 109 configurations were never considered. The purpose of this
exercise is to measure the bias actually incurred, so the search must match what
was actually searched.

*Rejected*: fixing the winner and testing it forward. That measures stability
over time, not selection bias, and the configuration would still have been chosen
using the whole dataset.

### 2. Window geometry — expanding train, quarterly test

| # | train | test | est. test trades |
|---|---|---|---|
| 1 | 2024Q2 – 2025Q1 | 2025Q2 | ~17 |
| 2 | 2024Q2 – 2025Q2 | 2025Q3 | ~23 |
| 3 | 2024Q2 – 2025Q3 | 2025Q4 | ~24 |
| 4 | 2024Q2 – 2025Q4 | 2026Q1 | ~19 |
| 5 | 2024Q2 – 2026Q1 | 2026Q2 | ~13 |
| 6 | 2024Q2 – 2026Q2 | 2026Q3 | ~13 |

Minimum training length is four quarters. Six selections, ~109 forward trades.

The trade counts above are estimates taken from one configuration
(`OF+mf0+r90|be1.25`, 195 trades total). Each window selects a different
configuration, so actual counts will differ — a window that picks a
low-frequency configuration will contribute fewer forward trades. The estimates
size the exercise; they are not a commitment.

*Rejected*: half-year test windows (only 3 selections — too few to judge whether
the selection process works). *Rejected*: rolling fixed 12-month train (each
selection capped at ~80 trades and never improves; adaptation to regime change
is not what is being measured here).

### 3. Selection metric — expectancy R, minimum 40 training trades

Within a training window, eligible configurations are those with at least 40
trades; among those, the highest mean R per trade wins.

The trade-count floor is not optional. Without it the search selects a
configuration that took six trades and won five, which is luck and collapses
forward.

*Rejected*: profit factor (leans toward low-frequency configurations, which
works against the stated goal of increasing trade count). *Rejected*: total R
(rewards trade count but ignores drawdown).

---

## Architecture

```
walkforward.py
├── load_config_logs()   113 saved trade CSVs -> {config_name: DataFrame}
├── dedupe()             collapse configs with identical trade logs
├── make_windows()       emit (train_start, train_end, test_start, test_end)
├── select(window)       filter to train dates, apply >=40-trade floor,
│                        rank by expectancy R, return winning config name
├── collect(window, cfg) return that config's TEST-period trades only
├── stitch()             concatenate all test slices -> honest equity curve
└── verify()             re-run 2 windows through the real model, assert equality
```

**Data flow**: 113 logs -> 6 windows -> 6 independent selections -> 6 disjoint
test slices -> one stitched ~109-trade result.

**Invariant**: a trade enters the final result only if its configuration was
chosen using strictly earlier data. No trade is ever scored by a configuration
that saw it.

### The load-bearing assumption

Slicing a configuration's full-period trade log by date must give exactly the
trades that configuration would have produced in that period alone. This holds
only if the model never looks forward. It uses prior-day bias and trailing 5-day
ADR, and every configuration was run from 2024-04-01 with warm-up before it, so
it should hold.

**This is verified, not assumed** — see Testing. If verification fails, the
shortcut is void and the fallback is to re-run the model inside each window
(~10 hours).

---

## Edge cases

| case | handling |
|---|---|
| No configuration clears the 40-trade floor | lower the floor to 25 for that window and **record it in the output** — never silently |
| Tie on expectancy | break by trade count (more wins), then alphabetically — deterministic |
| Winning configuration has 0 test-period trades | window contributes nothing but is still counted as a window, not dropped |
| Duplicate configurations | deduplicate by trade-log hash first, so the space is not 90 configurations wearing 113 names |
| Missing or corrupt log | fail loudly; never skip |

**Honest-reporting rule**: the stitched result is reported as-is. No post-hoc
window exclusion, no "best 4 of 6", no re-running with different windows if the
answer is disappointing.

---

## Testing

1. **No-lookahead verification** (critical) — re-run two windows through the
   actual model restricted to that date range; assert the trade list matches the
   sliced log exactly. Failure voids the design.
2. **Leak test** — assert every trade in the final result has a date strictly
   after its window's `train_end`.
3. **Determinism** — identical inputs produce identical winners across two runs.
4. **Duplicate report** — state how many of the 113 configurations are distinct.

---

## Outputs

1. **The stitched forward result** — trades, win %, PF, expectancy, max drawdown.
   This is the honest number.
2. **Per-window winners** — shows whether selection is stable or thrashing
   between configurations. Thrashing is itself evidence the edge is noise.
3. **Comparison** — walk-forward vs in-sample 1.487 vs contaminated-OOS 1.418.

## Interpretation, fixed in advance

Stated before seeing the result, so it cannot be rationalised afterwards:

- **PF > 1.3 forward** — the edge survives honest selection. Proceed to
  live-readiness work.
- **PF 1.0 – 1.3** — weak but real. The FX result was partly selection bias.
  Worth continuing only with reduced expectations.
- **PF < 1.0** — the FX result was selection bias, the same verdict the index
  holdout gave. Stop optimising this model and reconsider the strategy.

## Open questions

- Whether per-window winner thrashing should itself disqualify the model, and at
  what threshold. Deferred until the thrashing is observed.
- Whether the six windows should be pooled trade-by-trade (current plan) or
  averaged per window. Pooling is planned; per-window figures are reported
  alongside so both readings are available.
