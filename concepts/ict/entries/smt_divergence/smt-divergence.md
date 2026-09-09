---
name: smt-divergence
aliases: [SMT, smart money technique, smart money divergence, crack in correlation, index SMT, correlated pair SMT]
category: entries
related: [liquidity-sweep-stop-hunt, market-structure-shift, entry-sequence, swing-points, daily-bias]
parameters:
  groups: "indices = USATECHIDXUSD/USA500IDXUSD/USA30IDXUSD (NQ/ES/YM proxies); fx_dollar = EURUSD/GBPUSD/DOLLARIDXUSD (DXY inverse)"
  am_window: "05:00-09:30 New York (DST-aware)"
  pm_window: "12:00-15:00 New York (DST-aware)"
  comparison_timeframe: "5m for execution; the same timeframe and the same bar index on every instrument"
  swing_lookback: 2
  pivot_dedupe_bars: 2
  match_tolerance_bars: 1
  min_window_bars: 12
  smt_mode: "pivot (two confirmed swings) | breach (live run through a prior pool)"
  min_pool_bars: 6
  pool_gap_bars: 1
  corr_lookback: 288
  min_abs_corr: 0.0
detection: implemented
---

# SMT Divergence (Smart Money Technique)

A **crack in correlation** between correlated instruments: one makes a new swing
extreme while the other(s) fail to confirm. At a low, one index prints a lower
low (sweeping sell stops) while another holds a higher low — the sweep was
engineered, not genuine weakness. At a high, the mirror.

SMT is a **confirmation layer, never a standalone signal**. It grades a sweep
(sweep + SMT > sweep alone) and is strongest at a HTF PD array inside a killzone
alongside an MSS.

## Correlation groups

Measured 5m log-return correlation on this repo's data:

| Group | Members | Correlation |
|---|---|---|
| `indices` | USATECHIDXUSD, USA500IDXUSD, USA30IDXUSD | NQ/ES +0.92, ES/YM +0.95, NQ/YM +0.83 (2015-2023) |
| `fx_dollar` | EURUSD, GBPUSD, **DOLLARIDXUSD (inverse)** | EU/GU +0.77; DXY −0.88 vs EU, −0.78 vs GU |

An **inversely** correlated member confirms a group *low* by making its own
*high*. Comparing its low there inverts the reading and reports tandem sessions
as divergences — pinned by
`tests/test_smt_divergence.py::test_omitting_the_inverse_flag_manufactures_a_false_signal`.

## Detection Rules

1. **Bracket a comparison window**, in New York local time, DST-aware:
   - **AM: 05:00 → 09:30** — London traders returning from lunch through the
     equities open.
   - **PM: 12:00 → 15:00** — the lunch-hour and post-lunch relative extremes.

   The window is a span, not a pair of goalposts: the extremes compared need not
   sit at its edges. A low at 07:00 compared against one at 09:30 is the normal
   case.

2. **Find the relative extremes inside the window.** Pivots are collected from
   the *union* of all instruments (a swing only one index printed is still a
   comparison moment) and deduped within `pivot_dedupe_bars`.

3. **Compare at the same timestamps.** Every instrument is read in a
   ±`match_tolerance_bars` neighbourhood of both pivot times. This is the
   knowledge base's criterion c2 and its most-cited failure mode — never compare
   one instrument's 09:00 candle with another's 10:00. It requires a **shared
   bar index**; `DataLoader.read_smt_group` intersects the indices, and
   `find_smt_events` raises if the frames disagree.

4. **Classify.** With `n` instruments and `k` of them making the new extreme:
   `k == 0` or `k == n` is tandem — no crack. `0 < k < n` is an SMT, with the
   `k` as `leader` (swept) and the rest as `diverging` (refused).

5. **Score it.**
   - `sweep_depth` — how far the deepest leader ran past its prior level.
   - `clarity` — how decisively the *least* decisive diverging instrument
     refused. Taking the minimum stops one lukewarm member inflating the reading.
   - Both normalised by a caller-supplied scale (the instrument's ADR), so they
     are comparable across ES at 5,000 and NQ at 20,000.

6. **Expire it.** `negated_at` returns the first time a diverging instrument
   takes **its own** prior compared level — the level it refused. A negated SMT
   must not fill an order.

## Two readings: pivot and breach

`find_smt_events` compares **two confirmed swing pivots**. It matches the
"relative highs and lows should be compared" phrasing directly, but a pivot does
not exist until `swing_lookback` bars have printed to its right. The reading is
therefore late by construction, and every event carries a `confirmed_time`
marking the first bar on which it was knowable. Acting on `time` instead is
lookahead: it buys the exact low with hindsight. Measured on a deliberately
naive arm, 75 of 75 signals were armed 0 minutes after a pivot that needed 10
minutes to confirm, turning a losing rule into an apparent Sharpe of 3.1.

`find_smt_breach_events` reads the same idea **causally**. The pool is history —
the running extreme of the window over bars that have already closed — and the
signal is price running through it *now* while the correlated instruments do
not:

> "that run under the **5 a.m. low** on NASDAQ, that's a stop run" — M10-AM `[14:54]`

Nothing from the future is used, so `confirmed_time == time` and the event is
actionable on the bar it prints. This is the only reading a turtle-soup entry
can be filled on: by the time a pivot confirms, price has left the sweep zone.

## Correlation must be live

SMT presumes the instruments are correlated *right now*. `realised_correlation`
returns the weakest pairwise log-return correlation across the group over a
trailing window, sign-flipping inverse members so a healthy EURUSD/DXY pairing
scores near +1. Below `min_abs_corr` a "divergence" is two markets doing
unrelated things.

This is the principled form of the clarity band: the quartile with the *largest*
divergence measured the worst outcomes and the highest adverse excursion, which
is the signature of decorrelation, not of an engineered sweep.

## Direction is chosen by bias, not by the divergence

> "When institutional order flow is bullish … we have to be comparing relative
> **lows** across the three indices." — Month 10, Index Futures AM Trend `[08:05]`

Compare lows only when HTF order flow is bullish; highs only when bearish. A
divergence read on the wrong side is not a weaker signal — it is a different
claim about the market, and it measures worse than no signal at all.

> "When one diverges, it doesn't necessarily mean that there's a trade. You have
> to have the analysis on your dollar index because that's going to set the
> tone." — Month 10, Relative Strength Analysis `[07:15]`

## Frequency, and why a filter is mandatory

> "Typically, there will be an index SMT divergence to qualify the AM trend
> setups a few times a week… there's been every single trading day some measure
> of an index SMT divergence. **It does not equate to a large opportunity with
> low risk.**" — Month 10 AM `[16:59]`–`[17:20]`
>
> "When it's not obvious, assume it's not there." — `[17:38]`

Measured on this repo's data (806,715 aligned 5m bars, 3,645 sessions,
2015-2026): **5.3 SMT events per week**, matching the source. Detection is
therefore not the hard part — selection is.

## Which instrument to trade — unresolved in the source

The corpus gives three incompatible answers:

1. **Turtle-soup the leader** — "the one that does go lower, that's a turtle
   soup … buy it below the old low" (M10-AM `[16:22]`, `[18:30]`).
2. **No preference** — "there's not one over the other in terms of advantage"
   (M10-AM `[16:46]`).
3. **Always one fixed instrument** — "if I see that crack among all three my
   trade's going to be taken in the S&P" (M10-PM `[10:10]`).

Earlier revisions of this file asserted "trade the instrument that refused to
confirm" as settled doctrine, citing the 2020 forex notes. It is not settled.
The detector reports `leader` and `diverging` and takes no position; selection
is a model-level parameter. Exploratory measurement over 11.7 years favours
*diverging* (+0.058 ADR, 58.4% positive) and *fixed instrument* (+0.059, 57.3%)
about equally, with *turtle-soup the leader* worst (+0.046, 56.9%) — a real but
small separation. See `models/ict/smt_model/`.

## Common Mistakes

- **Trading SMT alone.** It is a confluence factor. Unfiltered, it measures
  *worse* than the no-SMT baseline in half the window/side cells tested.
- **Ignoring bias.** Bias-aligned and counter-bias readings differ by more than
  the SMT signal itself is worth.
- **Assuming a bigger divergence is a stronger signal.** It is not monotone. The
  clearest quartile performs *worst*, with the highest adverse excursion — a very
  large divergence is two instruments decorrelating, not an engineered sweep.
  Use a band (`min_clarity`, `max_clarity`), not a floor.
- **Comparing weakly correlated instruments.** EURUSD vs USDJPY is unreliable,
  not inverted. Inverse logic applies only to strong-negative pairs like
  EURUSD/DXY.
- **Misaligned bars.** Comparing different timestamps invents divergences.
- **Random-bar SMT.** The compared extreme must be a liquidity pool.

## Implementation

- `find_smt_events(correlated, session_date, window, side, scale, inverse, ...)`
  → `list[SMTEvent]`. Default returns only the most recent comparison, which is
  what a live trader sees; `all_pairs=True` returns every consecutive pivot pair
  for research.
- `negated_at(...)` / `smt_is_live(...)` implement `smt-failure`.
- `detect_smt` / `smt_confirmed` are **deprecated** rolling-window functions
  retained unchanged only because `models/ict/sniper_model` is built on them.
  They have no swing anchor, no window, no cross-instrument alignment and no
  expiry. Do not use them in new work.

## Sources

- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-INDEX-FUTURES-AM` — AM window,
  bias-gated side selection, frequency, "when it's not obvious".
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-INDEX-FUTURES-PM` — PM window,
  fixed-instrument selection.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-RELATIVE-STRENGTH-ANALYSIS` —
  divergence alone is not a trade.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-02-SECRETS-SELECTING-HIGH` — SMT as
  one of three required confluence areas.
- `_refs/ict-knowledge-library/concepts/16-smt-divergence/` — the five KB pages.
- [ICT SMT divergence — innercircletrader.net](https://innercircletrader.net/tutorials/ict-smt-divergence-smart-money-technique/)
- Stronger/weaker selection rule (now contested): `strategies/notes/ict-forex-notes-mmari-2020.md` §6
