---
name: smt-model
aliases: [SMT model, index SMT model, crack-in-correlation model]
category: models
related: [smt-divergence, daily-bias, liquidity-sweep-stop-hunt, market-structure-shift, change-in-state-of-delivery, fair-value-gap, order-block, targets-and-exits]
parameters:
  group: "indices (NQ/ES/YM CFDs) | fx_dollar (EURUSD/GBPUSD/DXY, DXY inverse)"
  windows: "am (05:00-09:30) + pm (12:00-15:00) NY; london (02:00-05:00) available but NOT from the index lectures"
  smt_mode: "pivot | breach"
  entry_style: "turtle_soup | buy_stop | fvg | market"
  entry_timeframe: "5min"
  adr_days: 14
  target_rr: 2.0
  min_stop_adr_pct: 0.02
  min_stop_spread_mult: 10.0
  max_stop_adr_pct: 0.40
detection: implemented
---

# SMT Divergence Model

`bias picks the side -> a relative extreme inside the comparison window is
raided -> the correlated instruments refuse to confirm -> reversal -> deliver`

SMT is a **confirmation layer, not a trigger**. This model is the stack that
surrounds it, built so the divergence gate can be removed (`require_smt: false`)
and the identical machinery re-run. The difference between those two runs is the
only honest measure of what SMT contributes, and it is the model's primary
output.

## Why the surrounding stack exists

Measured on 806,715 aligned 5m bars, 3,645 sessions, 2015-01-01 → 2026-09-04:

- Unfiltered SMT performs **worse than the no-SMT baseline in 2 of 4**
  window/side cells.
- Bias-aligned PM SMT returns +0.082 ADR against +0.021 counter-bias and +0.049
  for the bias-aligned tandem baseline.
- Divergence frequency is 5.3/week, matching ICT's "a few times a week". The
  hard part is selection, not detection.

Full method and the F1-F8 findings: `analysis/` in this folder and
`docs/superpowers/specs/2026-09-06-smt-divergence-model-design.md`.

## Checklist (rule of exclusion — every item must pass)

Bullish shown; bearish is the mirror.

1. **Bias** — `daily_bias` on the group's reference instrument is non-neutral.
   Bullish order flow means we compare **lows**. This is not a confluence
   bonus: a divergence read on the wrong side is a different claim about the
   market and measures worse than no signal.
2. **Window** — inside a stated comparison window, NY clock, DST-aware.
   AM 05:00-09:30, PM 12:00-15:00.
3. **Pool** — the earlier relative low being compared was still resting when the
   raid arrived (`require_pool_live`). Optionally it must also coincide with a
   standing PDH/PDL (`require_named_pool`).
4. **Raid** — the leader trades through that low and is **rejected**: a close
   back above it inside `close_back_within_bars`. Without this the pattern is a
   run (continuation), not a sweep.
5. **SMT** — at the same timestamps, at least one correlated instrument refused
   to make its own new low, with `clarity` inside a **band**
   (`min_clarity_adr` … `max_clarity_adr`) and `sweep_depth` above a floor.
   The band matters: the clearest divergence quartile measured the *worst*
   outcome with the *highest* adverse excursion, because a very large
   divergence is decorrelation rather than an engineered sweep.
6. **Reversal** — CISD or MSS within `max_minutes_to_reversal` of the raid.
7. **Entry** — one of four, and the choice is measured rather than assumed:
   - `turtle_soup` — a limit inside the sweep zone, between the raided level
     and the sweep extreme. *"identify that as turtle soup… buy it below the
     old low"* (M10-AM `[16:27]`).
   - `buy_stop` — a stop entry beyond the high of the last down-close candle of
     the raid leg. *"buying on a stop on a down-closed candle, whatever that
     high is"* (M10-AM `[15:56]`).
   - `fvg` — limit into the FVG/OB left by the displacement, armed only once
     the array was visible.
   - `market` — at the open after confirmation.

   The first two act **at** the extreme, which is the point: the delayed
   retrace entry is what made the stop wide enough that only 29% of trades
   reached 1.5R, with median adverse excursion (1.03R) exceeding median
   favourable (0.76R). `turtle_soup` requires `smt_mode: breach` to be fillable
   at all — by the time a pivot confirms, price has left the zone. Measured:
   zero fills in pivot mode.
8. **Stop** — at the level whose breach negates the SMT (and beyond the entry
   array, whichever is further), plus a buffer. Floored by **both**
   `min_stop_adr_pct × ADR` and `min_stop_spread_mult × spread`, capped by
   `max_stop_adr_pct × ADR`.
9. **Target** — fixed R, opposing liquidity, or a pure time-of-day exit.
10. **Negation** — an unfilled order is cancelled if the divergence closes
    before entry.

## Two design decisions worth stating

**The stop is the invalidation level.** `stop_mode: smt_invalidation` places the
stop at `min(v1, v2)` for a long — for a *leader* that is its own sweep extreme,
for a *diverging* instrument it is the level it refused. One rule covers both
roles without branching, and the trade dies exactly when its premise does.

**Two stop floors, not one.** `min_stop_adr_pct` keeps the stop clear of market
noise; `min_stop_spread_mult` keeps execution cost a small fraction of risk
(`risk >= 10 x spread` caps round-turn spread at ~10% of risk). They guard
different failures and a single number conflates them: the original 5%-of-ADR
floor, inherited from the Judas config, silently rejected **every** turtle-soup
entry, because a tight stop is that technique's entire purpose. The spread
floor is derived from the Judas post-mortem, where a 0.3-pip spread against 5-6
pip stops took 43% of the edge and flipped out-of-sample expectancy negative.

## Frequency

The strict daily-bias gate rejected **68.6%** of sessions on its own and was the
main reason the model produced roughly one trade a month. `bias_mode: off`
removes the veto and records bias on every signal instead, so it can be used as
a ranking feature later rather than a blunt filter. Measured on 46 months of
index data that took the model from ~1 to **4.7 fills per month**.

## Which instrument is traded

`instrument_selection` is an experiment axis, not a doctrine, because the corpus
teaches three incompatible rules — turtle-soup the leader, no preference, and
always trade one fixed instrument. See the "Which instrument to trade" section
of `concepts/ict/entries/smt_divergence/smt-divergence.md` for citations and the
exploratory measurement.

## Timing of exits

Doctrine ties the exit to time of day rather than to price alone: AM setups are
held "minimum 10:30 to 11 o'clock", PM setups toward the 15:00 bond close and
the final hour. `WINDOW_EXIT` encodes that (AM → 11:00, PM → 16:00) and
`target_mode: time` tests the pure time-of-day exit on its own.

## Running it

```bash
# one configuration
python models/ict/smt_model/backtests/smt_backtest.py \
    --config models/ict/smt_model/backtests/configs/smt_indices.yaml

# the full experiment suite (arms, windows, selection, exits, costs, OOS)
python models/ict/smt_model/backtests/experiments.py --tag indices
python models/ict/smt_model/backtests/experiments.py --group fx_dollar --tag fx
```

## Non-goals

- No per-instrument parameter tuning; no dropping losing instruments from a
  group. A negative instrument is evidence against the model.
- No optimisation before the out-of-sample split is drawn (2022-01-01).
- If arm A ≈ arm B, the finding is "SMT adds nothing measurable here".

## Sources

- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-INDEX-FUTURES-AM` — AM window,
  bias-gated side selection, frequency, "when it's not obvious, assume it's not
  there", turtle-soup entry.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-INDEX-FUTURES-PM` — PM window,
  fixed-instrument selection, holding to the bond close.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-INDEX-FUTURES-INDEX` — session
  profiles, hold-for-time-of-day.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-10-RELATIVE-STRENGTH-ANALYSIS` —
  a divergence alone is not a trade.
- `ICT-2022-MENTORSHIP-CORE-CONTENT-MONTH-02-SECRETS-SELECTING-HIGH` — SMT as
  one of three required confluence areas.
