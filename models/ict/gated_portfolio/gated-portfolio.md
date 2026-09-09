---
name: gated-portfolio
aliases: [gated queue, portfolio queue, admission-gated portfolio, GPQ]
category: models
related:
  - judas-swing
  - model-2022
  - silver-bullet
  - turtle-soup
  - market-maker-models
  - targets-and-exits
  - stop-placement
  - liquidity-sweep-stop-hunt
  - killzones
parameters:
  instruments: [ES, NQ, YM]
  target_trades_per_week: [2, 3]
  min_reward_risk: 1.5
  target_profit_factor: 2.0
  partial_exit_r: 1.5
  partial_exit_fraction: 0.5
  max_concurrent_positions: 2
  max_correlated_positions: 1
detection: not-implemented
---

# Gated Portfolio Queue

A meta-model. It does not find setups — the existing models do that. It decides
**which of them get taken**, and it is built so that raising selectivity does not
cost trade count.

## Why this model exists

Four constraints were set for it, and two of them pull against each other:

| # | Constraint |
|---|---|
| 1 | Win rate must go **up** |
| 2 | Trade count must **not** go down — minimum 2–3 trades/week |
| 3 | Losing trades minimised |
| 4 | Profit factor **> 2.0**, with a dynamic reward-risk floor of 1.5 R |

Constraints 1 and 2 are in direct conflict for any single model: every filter that
raises win rate does so by deleting trades. `judas-swing` today takes 73 trades in
2.4 years across six pairs — about **30 a year**, against a requirement of
**104–156**. Filtering it harder moves it further from the target, not closer.

The resolution is not a better filter. It is a **wider funnel with a harder gate**:
generate many more candidates than we intend to trade, then admit only the top
slice. Selectivity and trade count stop competing, because they are now controlled
by two different knobs.

## The arithmetic that sets the target

`PF = (WR × R) / (1 − WR)`, so `WR = PF / (PF + R)`. What win rate PF 2.0 actually
demands depends entirely on the payoff:

| Payoff R | WR needed for PF 2.0 | Breakeven WR |
|---|---|---|
| 1.5 R | 57.1% | 40.0% |
| 2.0 R | 50.0% | 33.3% |
| 2.5 R | 44.4% | 28.6% |
| **2.70 R** (judas today) | **42.6%** | 27.0% |
| 3.0 R | 40.0% | 25.0% |
| 4.0 R | 33.3% | 20.0% |

`judas-swing` currently realises **31.5% WR at a 2.70 R payoff** (avg win +3.12 R,
avg loss −1.15 R, spread-costed). At that payoff, **PF 2.0 needs 42.6% win rate,
not 70–80%.** That is a +35% relative improvement — the same order as the single
stop-sizing fix that moved `fallenpheonix23` from 24.8% to 40.3%.

> **Corrected 2026-09-07.** This section originally concluded "hold R high and let
> win rate be the smaller lift", and treated the 70%-win-rate route as
> arithmetically equivalent but practically worse. The only dataset that has
> actually swept this axis says otherwise. Across 13 XAUUSD configurations,
> `corr(win rate, payoff) = −0.945` — they are one number, not two — but
> `corr(win rate, PF) = +0.667`, with mean PF **1.497** above 70% win rate against
> **1.251** below it, and the best PF observed (1.76) at the highest win rate
> (76.7%). PF 2.0 at 76.7% needs a payoff of only **0.608 R**; that configuration
> reached 0.54 R, a 13% shortfall. See
> [`cross-model-winrate-payoff-frontier`](../../../analysis/entries/cross-model-winrate-payoff-frontier-2026-09-07.md).
>
> **The revised design decision:** the exit policy chooses a point on a frontier
> it cannot move, and the high-win-rate end is the side worth testing for a PF-2
> target. What survives from the original reasoning is the cost warning — a
> smaller payoff is taxed harder by spread, which at our stop sizes is already 43%
> of gross edge. So the two must be swept together, with a cost-per-R budget
> attached to each exit variant.
>
> The 1.5 R floor in constraint 4 is **incompatible** with constraints 1 and 4
> held jointly: 70% win rate at a 1:2 payoff implies PF 4.67, and 80% implies
> 8.00. The three constraints are satisfiable in pairs, never as a triple. For the
> PF-2 variant the floor is dropped and replaced by a payoff *target* of
> 0.6–0.9 R.

## Architecture

Six layers. Each is separately measurable, and the model is only as good as the
weakest one.

### L1 — Candidate generation (wide)

Every model emits **candidates**, not trades. A candidate is a fully specified
would-be trade: entry, stop, target ladder, direction, and the context needed to
grade it. Nothing is suppressed here; the one-trade-per-day rule is lifted at this
layer and reimposed at L4 where it can be calibrated.

Width comes from four multipliers:

- **instruments** — ES, NQ, YM (15 years of 1m data, already in `Data/`)
- **models** — `judas-swing`, `model-2022`, `silver-bullet`, `turtle-soup`,
  `market-maker-models`
- **sessions** — London KZ, NY AM KZ, NY PM
- **lifting the one-per-day cap** where setups are genuinely independent

### L2 — Hard rejection gates (rule of exclusion)

Binary, no scoring. A candidate failing any of these is discarded and logged with
the reason:

- **dynamic RR floor** — target is the nearest *unswept* structural level (the
  MMXM engineered-liquidity ladder where available, otherwise the swing pivot);
  `R = (target − entry)/(entry − stop)`; **reject if `R < 1.5`**
- **regime gate** — trend/volatility filter (ADX-style); counter-trend candidates
  additionally require price on the correct side of the long SMA
- **staleness** — zone age cap and MSS expiry
- **structure quality** — minimum displacement, minimum gap size in ATR terms

### L3 — Quality grade

A 0–10 score over HTF alignment, model clarity, session, confluence count and
computed RR. **The weights are fitted in-sample and validated out-of-sample — they
are not asserted.** A grade that does not separate winners from losers out-of-sample
is deleted, not tuned.

### L4 — Admission policy — the knob that protects trade count

Candidates surviving L2 are ranked by L3 grade. Admission takes the top slice, and
**the slice size is set by the trade-rate target, not by a score threshold**:

```
admit the highest-graded candidates such that the realised rate is 2–3 per week
```

This is what makes constraint 2 a design constraint rather than a hope. It also
makes the funnel-width requirement explicit and measurable: at a 20% admission
rate, hitting 2–3 trades/week needs **520–780 raw candidates a year** (2–3 per
trading day); at 10%, twice that. **Measuring the actual raw candidate count is
the first build step**, because if the funnel is not wide enough, no amount of
grading will satisfy constraints 1 and 2 together.

### L5 — Portfolio risk

ES, NQ and YM are ~0.9 correlated; three simultaneous index longs are one position
with three times the risk. So:

- `max_concurrent_positions: 2`
- `max_correlated_positions: 1` — one index-direction exposure at a time; the
  higher-graded candidate wins the slot
- fixed 1 contract per trade (project convention), risk reported in R

### L6 — Exit — where "minimum losing trades" is actually won

- **partial exit of 50% at 1.5 R, then stop to breakeven** on the remainder
- runner targets the next rungs of the MMXM ladder

This is the direct lever on constraint 3. Once the partial fills, the worst case on
the remainder is 0 R, not −1 R, so a trade that reaches 1.5 R and then reverses
books **+0.75 R and counts as a win** instead of a full loss. Full −1 R losses
survive only on trades that never reach 1.5 R.

**It must be reported honestly.** This raises realised win rate and cuts variance;
it does **not** raise profit factor, and the backtest must publish the
single-exit and partial-exit variants side by side so the effect is visible rather
than absorbed into a headline.

## Verification — non-negotiable

Both gates now exist in `infrastructure/` and every run passes through them:

- `infrastructure/promotion.py` — `evaluate_gate(is_trades, oos_trades)`. PROMOTE
  requires ≥30 OOS trades, positive OOS expectancy after costs, OOS PF ≥ 1.15,
  <50% expectancy decay, and no losing streak exceeding 15% of the sample.
- `infrastructure/tripwire.py` — `run_tripwire(run_fn, df)`. Re-runs the model on
  bar-shuffled series. If the shuffled runs still earn, the edge is in the engine.

Applied to `judas-swing` as it stands today, the gate returns **REJECT on all five
criteria**. That is the baseline this model has to beat, and it is why the funnel
work is sequenced behind the measurement work.

## Build order

Each step ends in a number, and no step starts until the previous one has one.

| # | Step | Ends with |
|---|---|---|
| 1 | **Measure funnel width.** Run existing detectors over ES/NQ/YM, 15y, all sessions. Count raw candidates/week. | The admission rate needed for 2–3/week — or proof the funnel is too narrow |
| 2 | **Stop-sizing study.** Sweep `max(k×ATR, floor)` vs the current structural stop. Report WR, PF, expectancy *and* cost-as-%-of-risk. | The payoff R the rest of the design is built on |
| 3 | **Dynamic RR gate + rejection log.** Implement L2's RR floor; log every rejection with its computed R. | Whether 1.5 R is the right floor, measured on the rejected population |
| 4 | **L6 exits.** Partial-at-1.5R + BE trail, published against the single-exit variant. | The honest win-rate and PF delta |
| 5 | **L3 grade.** Fit weights in-sample, validate out-of-sample. | A grade that separates OOS, or a deleted layer |
| 6 | **L4 + L5.** Admission policy calibrated to 2–3/week, correlation cap. | The first end-to-end run through `evaluate_gate` |

## Honest status

Nothing in the reference audit reached PF 2.0 out-of-sample. The best validated
result anywhere in that pile was PF 1.52 at 52.1% win rate on 121 trades over 15
years; our own best costed number is PF 1.24 in-sample and 0.90 out-of-sample.
**PF > 2.0 sustained out-of-sample would be better than every result surveyed.**

That is not a reason to aim lower — the arithmetic above shows the required win
rate is 42.6%, not 70%, which is a normal-sized improvement rather than an
extraordinary one. It is the reason the promotion gate and the leakage tripwire
were built first: if a run does come back at PF 2.0, we need to already know
whether to believe it.

## Sources

- Arithmetic and evidence base: [`analysis/entries/cross-model-repo-edge-audit-2026-09-07.md`](../../../analysis/entries/cross-model-repo-edge-audit-2026-09-07.md)
- Cost and payoff baseline: [`models/ict/judas_swing/analysis/judas-swing-execution-baseline-2026-09-06.md`](../judas_swing/analysis/judas-swing-execution-baseline-2026-09-06.md)
- Target ladder concept: [`models/ict/market_maker_models/market-maker-models.md`](../market_maker_models/market-maker-models.md)
