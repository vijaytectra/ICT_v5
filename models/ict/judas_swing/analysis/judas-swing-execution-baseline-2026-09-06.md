# Judas Swing — Execution Cost Baseline

**Model**: judas-swing
**Concept**: judas-swing
**Instrument**: multi (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD)
**Analysis date**: 2026-09-06
**Dataset**: 1m cleaned CSVs, 2024-04-01 → 2026-09-04, resampled to 5m for entry.
887 session-days per pair; 73 filled trades across the basket.
**Scripts**: `models/ict/judas_swing/backtests/portfolio.py`,
`models/ict/judas_swing/backtests/f2_exposure.py`,
`infrastructure/execution.py`, `infrastructure/stats.py`
**Raw results**: `models/ict/judas_swing/backtests/results/baseline/fx_zerocost.csv`,
`.../fx_spread.csv`, `.../baselines.txt`
**Configs**: `models/ict/judas_swing/backtests/configs/judas_fx.yaml`

---

## What Was Tested

Three additive infrastructure changes, with the strategy's entry, liquidity, FVG
and session rules **frozen**:

1. **Execution cost model** (`infrastructure/execution.py`) — historical per-bar
   spread from the cleaned CSVs, applied directionally as `mid ± spread/2`;
   configurable slippage (entry / exit / stop separately) and commission; a
   `gap_fill` mode for stops.
2. **Configurable EOD exit** — `eod_exit_hour` in config, built explicitly in
   `America/New_York` and clamped to the session day. Default unchanged at 16.
3. **Enriched statistics** (`infrastructure/stats.py`) — R-based, with the exit
   reason enum `TP / SL / EOD_PROFIT / EOD_LOSS / BREAKEVEN / OTHER`.

Then two baselines over the same 73 trades:

- **A — legacy zero-cost**: mid prices, no spread, no slippage, commission 0,
  legacy stop fills. This is the reproduction control.
- **B — historical spread**: `spread_mode: historical`, `gap_fill: open`,
  slippage 0, commission 0.

**Neither baseline is validated.** Every parameter in `judas_fx.yaml` was chosen
on this same 2024-2026 window, so both are in-sample by construction.

---

## Results

### Regression control — zero-cost reproduces the previous engine

| | previous engine | new engine (zero-cost) |
|---|---|---|
| trades | 10 (EURUSD) | 10 |
| win rate | 40.0% | 40.0% |
| expectancy | +0.886 R | +0.886 R |
| profit factor (points) | 2.71 | 2.709 |
| execution cost | n/a | 0.00000000 |

Reproduction passes. The only headline number that moved is profit factor,
because it is now computed in **R** rather than price points (2.477 vs 2.709).
With different stop sizes per trade those are genuinely different quantities and
R is the correct one; the points figure is reproduced exactly.

### A vs B — basket, 73 trades

| metric | A zero-cost | B historical spread | delta |
|---|---|---|---|
| trades | 73 | 73 | 0 |
| profitable % | 31.51 | 31.51 | 0 |
| target-hit % | 28.77 | 28.77 | 0 |
| **expectancy R** | **+0.337** | **+0.191** | **−0.146** |
| median R | −1.000 | −1.076 | −0.075 |
| total R | +24.62 | +13.95 | −10.67 |
| gross profit R | 74.62 | 71.70 | −2.93 |
| gross loss R | 50.00 | 57.75 | +7.75 |
| **profit factor** | **1.492** | **1.242** | **−0.250** |
| **max drawdown** | **11.00 R** | **12.49 R** | **+1.49 R** |
| DD duration | 15 trades / 151 days | 16 trades / 185 days | +34 days |
| Sharpe | 2.510 | 1.415 | −1.095 |
| Sortino | 5.355 | 2.615 | −2.740 |
| max losing streak | 11 | 11 | 0 |
| avg / median hold | 110 / 55 min | 110 / 55 min | 0 |

Costs change no trade selection — the same 73 setups fire, at the same times,
with the same win rate. Only the P&L moves.

### Per instrument

| pair | n | win% | A expR | B expR | B PF |
|---|---|---|---|---|---|
| EURUSD | 10 | 40.0 | +0.886 | +0.826 | 2.283 |
| USDJPY | 17 | 41.2 | +0.639 | +0.591 | 1.954 |
| AUDUSD | 13 | 30.8 | +0.396 | +0.178 | 1.216 |
| GBPUSD | 12 | 25.0 | +0.001 | −0.152 | 0.825 |
| USDCHF | 10 | 20.0 | +0.024 | −0.148 | 0.842 |
| USDCAD | 11 | 27.3 | −0.045 | −0.306 | 0.671 |

### Out-of-sample split (cut 2025-11-01)

| | n | win% | PF | expR |
|---|---|---|---|---|
| A in-sample | 55 | 32.7 | 1.619 | +0.416 |
| A out-of-sample | 18 | 27.8 | 1.133 | +0.096 |
| B in-sample | 55 | 32.7 | 1.372 | +0.283 |
| **B out-of-sample** | **18** | **27.8** | **0.897** | **−0.091** |

### By year (B)

| year | n | win% | total R | expR | PF |
|---|---|---|---|---|---|
| 2024 | 24 | 37.5 | +13.15 | +0.548 | 1.78 |
| 2025 | 36 | 30.6 | +3.56 | +0.099 | 1.12 |
| 2026 | 13 | 23.1 | −2.76 | −0.212 | 0.78 |

### Exit reasons (B)

| reason | n | share | total R | expR |
|---|---|---|---|---|
| TP | 21 | 28.8% | +70.32 | +3.348 |
| SL | 50 | 68.5% | −57.74 | −1.155 |
| EOD_PROFIT | 2 | 2.7% | +1.38 | +0.688 |
| EOD_LOSS | 0 | 0.0% | 0.00 | — |
| BREAKEVEN | 0 | 0.0% | 0.00 | — |
| OTHER | 0 | 0.0% | 0.00 | — |

EOD exits contribute **+1.38 R, 9.9% of net R, from 2 trades**. There are **zero
EOD losses**. The forced flat is not carrying the result, but it is not neutral
either — at 9.9% of a thin edge it deserves the exit study before it is trusted.

### Cost attribution (B)

- spread: **10.673 R total, 0.146 R per trade** ($514.31 at the configured notionals)
- slippage: 0.000 R (not modelled)
- commission: **$0.00 — explicitly zero, not modelled**
- pre-cost total R 24.62 → post-cost 13.95

The average stop is ~5.6 pips on EURUSD, so a ~0.3 pip round-turn spread is
~5% of risk per trade. **Spread consumes 43% of the zero-cost edge.**

---

## Key Findings

1. **A −20R trade was manufactured by my own gap rule, not by the market.**
   On the 2026-09-04 NFP candle, USDCAD's 12:30 bar opens continuous at 1.37928
   and reaches 1.38722 *inside the bar*. The limit filled on that bar and the stop
   was breached on that same bar, but the exit scan starts at the next bar, whose
   open sat 60 pips through the stop. `gap_fill='open'` filled there, booking
   −19.95 R for a −1 R trade and dragging USDCAD to −1.990 R/trade.
   The rule now requires that the previous bar did **not** already trade through
   the level. USDCAD: −1.990 → **−0.306 R/trade**, matching its measured 0.261
   R/trade cost. Pinned by `test_same_bar_breach_is_not_a_gap`.
   The legacy engine hid this by never checking the entry bar for a stop breach —
   realistic gap fills turned a silent optimism into a loud fabrication.

2. **Costs do not change which trades fire, only what they pay.** Identical 73
   trades, identical win rate and hold times in A and B. That is the expected
   signature of a post-trade cost layer and is evidence the layer is additive.

3. **Spread takes 43% of the edge.** +0.337 → +0.191 R/trade. The cause is stop
   size: this model's structural stops average 5-6 pips, so a 0.3 pip spread is a
   large fraction of risk. Any future work that tightens stops further will be
   taxed harder, and any work that widens them gets cost relief for free.

4. **Out-of-sample expectancy is negative once costs are charged.** −0.091 R over
   18 trades, PF 0.897. Zero-cost OOS was +0.096. Costs are what flip the sign.

5. **Performance decays monotonically by year**: +0.548 → +0.099 → −0.212 R.
   With 24/36/13 trades per year this is not statistically decisive, but it points
   the same way as the OOS split rather than against it.

6. **Half the basket is negative after costs.** EURUSD and USDJPY carry the whole
   result (+18.3 R of +13.95 R net); GBPUSD, USDCHF and USDCAD are all negative.
   That is poor cross-instrument stability for a model whose rules contain no
   instrument branching.

7. **F2 exposure is real but immaterial, and now measured.** `find_fvgs`
   timestamps a gap to its middle candle, so the earliest bar a limit can fill on
   is the bar that confirmed the gap. **1 of 73 trades (1.4%)** filled there, and
   it was a loss (−1.10 R) — removing it *raises* expectancy to +0.209 R. The
   optimism is in the conservative direction. No detector change made.

8. **Killzone terminology is internally inconsistent.** The NY session anchors at
   07:00 (2017 mentorship / forex set) but ends at 11:00 (public 2016/2022 set),
   producing a 4-hour window belonging to neither published set. Not changed —
   session definitions were frozen — but it must be resolved before any session
   study is meaningful.

9. **The sample is too small to conclude anything.** 73 trades, 21 winners, max
   losing streak 11. A single 11-trade streak is 15% of the sample.

---

## Recommendations

| # | Action | Priority |
|---|---|---|
| 1 | Do **not** start the 16-step optimisation. The cost-adjusted OOS expectancy is negative; optimising now fits the in-sample half | high |
| 2 | Run the exit / MFE-MAE study next (step 3/4). TP expectancy is +3.35 R against SL −1.16 R; whether the 2.5 R:R floor is right is the highest-value open question | high |
| 3 | Resolve the killzone set inconsistency (finding 8) before any session-level analysis | high |
| 4 | Consolidate the two `CostModel` implementations — `infrastructure/execution.py` and `plot_trades.py` — onto the infrastructure one, once no session is editing model_2022 | medium |
| 5 | Add commission as a real number rather than 0. At 5-6 pip stops, a $3-7 round turn is another 5-10% of risk | medium |
| 6 | Treat GBPUSD / USDCHF / USDCAD as evidence against the model, not as pairs to exclude. No arbitrary instrument exclusions | high |

---

## Open Questions

- Is the 5-6 pip structural stop the real problem? It makes the model
  spread-sensitive by construction. Widening it is a strategy change and is
  frozen for now, but a sensitivity run would say how much of the fragility it
  explains.
- Does `entry_style: market` (no fill lag, wider effective stop) survive costs
  better than the limit entry? Untested — it is a parameter axis, and axes stay
  frozen until the exit study is done.
- 73 trades over 2.4 years is ~30/year across six pairs. Is that consistent with
  "about one setup per session across a basket of four or five pairs", or is the
  gate stack over-filtering? The rejection funnel says the bias gate alone removes
  ~70% of days.
- Limit fills assume zero queue latency. What does a conservative
  fill-only-if-traded-through rule do to the 21 TP exits?

---

*Baselines A and B are both in-sample. Neither is validated.*
