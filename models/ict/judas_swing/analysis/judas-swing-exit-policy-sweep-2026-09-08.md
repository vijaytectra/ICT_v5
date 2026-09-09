# Judas Swing — Exit Policy Sweep

**Model**: judas-swing
**Concept**: entries (exit policy; targets-and-exits)
**Instrument**: multi (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD)
**Analysis date**: 2026-09-08
**Dataset**: the 73 trades of the costed judas baseline, 2024-04-08 → 2026-09-04,
replayed bar-by-bar against 5m data. Entries, stops and instruments frozen; only
the exit rule varies. 36 policies.
**Scripts**: `models/ict/judas_swing/backtests/exit_sweep.py`,
`infrastructure/exits.py`, `infrastructure/promotion.py`
**Raw results**: `models/ict/judas_swing/backtests/results/exit_sweep/exit_sweep.csv`
**Configs**: none — the grid is built in `exit_sweep.build_grid`

*Extends: [judas-swing-execution-baseline-2026-09-06.md](judas-swing-execution-baseline-2026-09-06.md).
**Corrects** finding 3 of
[cross-model-winrate-payoff-frontier](../../../../analysis/entries/cross-model-winrate-payoff-frontier-2026-09-07.md).*

---

## What Was Tested

The frontier study measured, across 13 XAUUSD configurations,
`corr(win rate, payoff) = −0.945` and `corr(win rate, PF) = +0.667`, and
concluded that the high-win-rate end of the exit frontier was the side worth
testing for a PF-2 target. This is that test, on our own trades.

Every entry, stop, direction and instrument is taken verbatim from the costed
baseline log. Only the exit rule changes: fixed R targets from 0.6 to 3.0, the
structural target, partial fractions of 0.25/0.5/0.75 at 0.6/0.8/1.0/1.5 R,
and breakeven triggers at 0.6/1.0/1.5 R. Each variant is replayed through
`infrastructure.exits.simulate_exit` and scored full-sample, in-sample,
out-of-sample (cut 2025-11-01) and through `evaluate_gate`.

### The replay had to be made honest first

The reproduction control — baseline policy must return the log it came from —
failed three times, each on a different defect in the *runner*, not the engine:

| # | Defect | Effect |
|---|---|---|
| 1 | Structural target rebuilt from `planned_rr` | `planned_rr` is computed on mids while `entry_price` carries the spread. All 73 trades disagreed, by up to **1.49 R**. |
| 2 | R measured against `entry_price − stop` | The log's `risk_pts` reconciles to `\|entry_mid − stop\|` **exactly** (0.00000000). Every R in the replay was rescaled. |
| 3 | `cost_r` subtracted on top of spread-inclusive fills | `exec_cost_pts` is a decomposition of the spread already inside `entry_price`, not a further charge. Double-counted **0.07 R/trade**. |

Expectancy drift fell **0.1461 → 0.0001 R/trade (0.1%)**. The first run's drift
was 76.5% of the baseline expectancy and would have been presented as a result.

One difference remains by choice: **1 of 73 trades** has a bar holding both stop
and target. The replay books it a loss, the logged engine a win. Without tick
data neither is knowable, and the pessimistic reading is the one that cannot
flatter the strategy.

A partial exit is a second trip to the market that the logged convention never
charges for, so each partial pays an extra half-turn on the slice taken off.

---

## Results

### Reproduction control

| | n | WR% | payoff | PF | expR | totalR |
|---|---:|---:|---:|---:|---:|---:|
| logged | 73 | 31.5 | 2.699 | 1.242 | 0.1911 | 13.95 |
| replayed | 73 | 30.1 | 2.906 | 1.254 | **0.1912** | 13.96 |

### The frontier on our trades, against XAUUSD

| | our FX (36 policies, 73 trades) | XAUUSD (13 configs) |
|---|---:|---:|
| `corr(win rate, payoff)` | **−0.953** | −0.945 |
| `corr(win rate, PF)` | **−0.568** | **+0.667** |
| `corr(win rate, OOS PF)` | −0.369 | not reported |
| mean PF, WR > 60% | **1.024** (n=10) | 1.497 (WR > 70%) |
| mean PF, WR < 40% | **1.331** (n=6) | 1.251 (WR < 70%) |

### Best five by out-of-sample expectancy

| policy | WR% | payoff | PF | expR | oosWR | oosPF | oosExpR |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed 2.5R | 35.6 | 2.258 | 1.249 | 0.173 | 38.9 | **1.277** | **+0.188** |
| fixed 3.0R | 32.9 | 2.700 | 1.323 | 0.233 | 33.3 | 1.174 | +0.128 |
| fixed 1.5R | 47.9 | 1.392 | 1.282 | 0.158 | 44.4 | 1.071 | +0.044 |
| structural + BE1.5 | 27.4 | 3.066 | **1.614** | **0.344** | 22.2 | 1.041 | +0.026 |
| structural + 0.25@1.5R + BE1.5 | 47.9 | 1.646 | 1.516 | 0.290 | 44.4 | 1.037 | +0.023 |
| *baseline (structural, no partial, no BE)* | *30.1* | *2.906* | *1.254* | *0.191* | *27.8* | *0.975* | *−0.021* |

### The high-win-rate end

Every variant reaching 64.4% win rate ran a payoff of 0.52–0.63 R and a PF of
0.95–1.14. The best of them (`structural + 0.25@0.6R + BE0.6`, PF 1.137) has an
out-of-sample PF of 0.298.

### Promotion gate

**0 of 36 promote** — but the binding criterion is sample size: the gate requires
≥30 out-of-sample trades and this split has **18**. The rejections are not a
verdict on the policies; they are the gate correctly refusing to rule on 18
trades.

---

## Key Findings

1. **The payoff trade-off is a law; the PF gradient is not.**
   `corr(win rate, payoff)` is −0.953 here against −0.945 on XAUUSD — the same
   number on a different instrument, different timeframe, different strategy.
   Win rate and payoff are one quantity. But `corr(win rate, PF)` is **−0.568**
   here against **+0.667** there: the direction reverses.

2. **The frontier recommendation was wrong for our instruments.** The frontier
   study concluded the high-win-rate end was the side worth testing. On our
   trades it is the worst side — mean PF 1.024 above 60% win rate against 1.331
   below 40%. That study's own open question 3 asked whether the slope was
   instrument-dependent. It is, and the answer inverts the advice.

3. **The mechanism is cost, and the earlier cost warning was the part that held.**
   Judas stops average 5–6 pips, so spread is ~5% of risk per trade. At a 3 R
   payoff that is a rounding error; at the 0.55 R payoff a 64% win rate requires,
   it is a third of the average win. XAUUSD M15 stops are wide enough that the
   same policy shift is nearly free. The frontier's slope is set by cost as a
   share of payoff, which is why it points one way on gold and the other way here.

4. **The exit axis does carry a real improvement — on the patient side.**
   `structural + BE1.5` lifts full-sample PF **1.254 → 1.614** and expectancy
   **+0.191 → +0.344 R**, and moves out-of-sample expectancy from **−0.021 to
   +0.026 R**. Four policies flip the out-of-sample sign. A breakeven stop at
   1.5 R is the single highest-value change found.

5. **Nothing here can be promoted, and the best number is the least trustworthy.**
   18 out-of-sample trades against 36 policies: the top OOS result is the maximum
   of 36 draws on a sample of 18. That is the textbook selection-bias setup, and
   it is why `fixed 2.5R` at OOS PF 1.277 should be read as a hypothesis, not a
   finding.

6. **PF 2.0 was not reached on any axis.** Best full-sample PF across 36 exit
   policies is 1.614, on 73 in-sample trades. The exit policy alone does not
   close the gap from 1.25 to 2.0 on this model.

---

## Recommendations

| # | Action | Rationale | Priority |
|---|---|---|---|
| 1 | **Run this sweep on the 12-year PF 1.54 system.** The machinery is model-agnostic: it needs a trade log with entry/stop/direction/instrument plus bars. At ~2 trades/week over 12 years the out-of-sample split is ~1,200 trades, where the promotion gate can actually rule. | Finding 5 — the sweep is sound but this sample cannot decide. That system can. | **high** |
| 2 | **Test breakeven-at-1.5R there first**, as a single change against the incumbent. | Finding 4 — the largest and most robust effect found, and a one-parameter change. | **high** |
| 3 | **Measure cost as a share of payoff before choosing a frontier point.** For any candidate exit policy, report `cost_r / payoff`. Reject any policy where it exceeds ~10%. | Finding 3 — this ratio, not the win rate, is what decides which way the PF gradient points. | **high** |
| 4 | **Do not pursue the high-win-rate end on instruments with tight structural stops.** | Findings 2 and 3. It is not a universal route; it is a route that needs cheap execution relative to the payoff. | **medium** |
| 5 | **Keep the reproduction control on every future sweep**, at the 0.01 R/trade threshold. | It caught three independent defects that would each have shipped a wrong answer. The first threshold, 0.15, was loose enough to pass a 76% error. | **medium** |

---

## Open Questions

1. **Does breakeven-at-1.5R survive on a large sample?** It is the best candidate
   found, on 73 trades with 18 out-of-sample. Recommendation 1 answers it.
2. **Where is the cost/payoff break-even point?** Finding 3 says the PF gradient
   flips somewhere between XAUUSD's cost share and ours. Locating that threshold
   would turn "test the exit axis" into a prediction rather than an experiment.
3. **Is the 1 same-bar conflict representative?** One trade in 73 here, worth
   5.17 R. On a 1,200-trade log the same convention could matter far more, and
   it should be counted before it is argued about.
4. **Would a trailing stop beat a fixed breakeven?** Not tested — the grid covers
   partials, breakeven and fixed targets only.
