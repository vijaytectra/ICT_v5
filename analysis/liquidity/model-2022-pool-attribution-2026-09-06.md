# Model 2022 — Liquidity Pool Attribution and Out-of-Sample Failure

**Model**: model-2022
**Concept**: liquidity-pools
**Instrument**: multi — USATECHIDXUSD / USA500IDXUSD / USA30IDXUSD (Dukascopy index CFDs), plus 6 spot FX pairs
**Analysis date**: 2026-09-06
**Dataset**:
- in-sample: 2024-05-01 → 2026-09-04, 3 indices, ~600 sessions each, 918,785 1m bars per instrument
- out-of-sample: 2015-05-01 → 2023-12-29, same 3 indices, ~2,250 sessions each, 3,114,741 1m bars per instrument
- FX: 2024-05-01 → 2026-09-04, EURUSD / GBPUSD / USDJPY / USDCHF / USDCAD / AUDUSD, 1,001,998 1m bars each

**Scripts**:
- `models/ict/model_2022/backtests/fvg_sweep_backtest.py` — the runner
- `models/ict/model_2022/backtests/pool_attribution.py` — pool-set sweep (C0–C5)
- `models/ict/model_2022/backtests/oos_validation.py` — the 2015-2023 and FX validation
- `Data/prep/forex_1m_to_cleaned.py` — bid/ask → mid conversion, EET→UTC

**Raw results**: `models/ict/model_2022/backtests/results/runs/idx_{nasdaq,sp500,dow}{,_spread,_v1}/` (gitignored)
**Configs**: `models/ict/model_2022/backtests/configs/idx_{nasdaq,sp500,dow}.yaml`

> **Headline**: the configuration this study arrived at earns **+0.351 R/trade over 139
> in-sample trades** and **−0.051 R/trade over 506 out-of-sample trades**. The edge does
> not survive. Verdict: **REJECT**.

---

## What Was Tested

Roughly 25 configurations were searched on the 2024-2026 index data, in four stages:

1. **Execution realism** — a cost model was added first, charging half of each bar's
   *recorded* ask-bid spread on entry and exit in the direction that costs the trade, plus
   gap-aware stop fills. Commission and slippage exist as config but are zero throughout.
   Every number below is spread-charged.
2. **Pool attribution** — the 148-trade baseline was decomposed by swept pool.
3. **Pool sets** (C0–C5) — dropping the pools the attribution showed to be losing.
4. **Design axes** (D0–D9) and **target sensitivity** (E0–E8) — target R, entry depth,
   entry arrays, stop mode, MSS promptness.

The result of that search is referred to below as **E6**: ONS + REH/REL pools only, 2.0R
target floor, FVG/OB/breaker entry arrays, near-edge entry, wick stop with structural
buffer, daily bias gate.

---

## Results

### 1. Baseline decomposition — where the P&L actually comes from

148 trades, 2024-2026, all pools, spread charged. Total **+5.71 R**.

| Swept pool | Total R |
|---|---|
| ONS (overnight 05:00–09:15 NY) | **+25.4** |
| REH | +5.1 |
| REL | +1.9 |
| Chicago pit (prior + same-day) | **−12.1** |
| Opening range 09:30 | **−8.3** |
| PDL | −3.1 |
| PDH | −3.0 |

One pool family carries **+25.4 R**; every other pool combined is **−20.4 R**.

### 2. Pool timing is mechanically forced, not incidental

A pool cannot be swept before its own `formed_at`, so each family occupies a fixed slot:

| Pool family | Earliest observed sweep | Total R |
|---|---|---|
| PDL / REH / REL / Chicago-prior | 02:00–02:40 NY | −4.7 |
| PDH | 03:18 | −3.1 |
| **ONS** | **09:16** (formed 09:15) | **+25.3** |
| Opening range | 10:00 (formed 10:00) | −8.3 |
| Chicago same-day | 13:46 (formed 12:00) | −3.5 |

By sweep time, independent of pool:

| Sweep window (NY) | n | Profitable % | Total R |
|---|---|---|---|
| < 08:30 | 22 | 27.3 | −10.7 |
| 08:30–09:15 | 8 | 50.0 | +3.9 |
| **09:15–10:00** | **90** | 40.0 | **+21.1** |
| 10:00–11:30 | 21 | 33.3 | −9.2 |
| 13:30–16:00 | 7 | 28.6 | +0.7 |

`_detect_sweep` takes the *earliest* valid sweep and the model takes one setup per session,
so an early pool consumes the session before the ONS setup at 09:16 can be evaluated.
Removing the early pools raised ONS trades from 85 to 135 — the freed sessions were real.
Those 50 additional ONS setups were, however, worth **−10.6 R** collectively: on days where
an early pool was raided, that day's later ONS setup was of lower quality.

### 3. Loss anatomy (92 losing trades)

| MFE reached before exit | Count | Share |
|---|---|---|
| < 0.25 R — entry wrong immediately | 31 | 34% |
| 0.25–1 R — worked briefly, reversed | 39 | 42% |
| ≥ 1 R — worked, gave it back | 22 | 24% |

The dominant failure is entry selection, not exit management. Only 22.6% of stopped trades
ever reached 1R, which is why a breakeven-stop rule tested earlier lost money (−$954 on
EURUSD): it had little to rescue and killed half the winners.

### 4. Exit route vs trade outcome

| Exit | n | Net pts | Avg R |
|---|---|---|---|
| TP | 20 | +2,616 | 3.09 |
| **EOD_PROFIT** | 35 | **+2,129** | 0.95 |
| SL | 84 | −3,933 | −1.05 |
| EOD_LOSS | 9 | −77 | −0.18 |

The forced 16:00 flat supplies **44.9% of gross profit**. Removing those exits leaves the
book at **−1,393 pts**: TP income does not cover stop losses. Target-hit rate (13.5%) and
profitable-trade rate (37.2%) are different measurements and were being conflated before
`classify_exit` was added.

### 5. Execution cost

| | Zero cost | Historical spread |
|---|---|---|
| Net (3 indices) | +961.7 pts | +735.8 pts |
| Total R | +12.1 | +5.7 |
| Expectancy | 0.094 R | 0.051 R |
| Mean PF | 1.268 | 1.195 |
| Worst PF | 1.157 | 1.065 |

Spread alone removes **23.6%** of net. Break-even total cost is ≈ **6.5 pts/trade**, a level
realistic slippage would reach. Commission is zero in all runs.

### 6. Configuration search on 2024-2026

| Stage | Best variant | Trades | Total R | Mean PF | Worst PF |
|---|---|---|---|---|---|
| C0 baseline, all pools | — | 148 | +5.71 | 1.20 | 1.07 |
| C-series, pool sets | C2 ONS + REH/REL | 137 | +25.30 | 1.21 | 1.03 |
| D-series, design axes | D3 2.0R target | 137 | +41.04 | 1.49 | 1.07 |
| E-series, target × arrays | **E6 2.0R + arrays** | **139** | **+48.82** | **1.55** | **1.25** |

Target sensitivity (arrays on/off) formed a smooth inverted-U peaking at 2.0R with 2.5R
second — a plateau rather than a spike:

| Target floor | 1.0R | 1.5R | 2.0R | 2.5R | 3.0R |
|---|---|---|---|---|---|
| Total R (no arrays) | 24.6 | 30.1 | **41.0** | 36.1 | 25.3 |
| Total R (arrays) | — | 38.3 | **48.8** | 43.8 | — |

OB/breaker arrays changed **only Nasdaq** (+8R at every target level); Dow and S&P were
bit-identical with and without, because an FVG is always found there and the fallback never
fires. Failed cleanly: Asian range as a pool (−11.7R), ATR stops (33.5 vs 48.8R), a tighter
15-bar MSS budget (12.8R, not positive on all three).

### 7. Out-of-sample — the decisive test

E6 applied unchanged to 2015-2023, same three instruments, price params re-measured on that
period by the same volatility formula (Nasdaq traded near 7,600 then against 29,500 now, so
frozen point values would fail arithmetically).

| Period | Trades | Prof % | Expectancy | Total R | Mean PF | Worst PF | Positive |
|---|---|---|---|---|---|---|---|
| in-sample 2024-26 | 139 | 46.2 | **+0.351 R** | **+48.8** | 1.55 | 1.25 | 3/3 |
| **OOS 2015-2023** | **506** | 33.4 | **−0.051 R** | **−25.8** | **0.91** | **0.77** | **1/3** |

Per instrument out-of-sample: Dow **+11.87 R** (PF 1.13), Nasdaq **−16.42 R** (PF 0.77),
S&P **−21.27 R** (PF 0.82).

### 8. FX, 2024-2026 — E6 applied without modification

| Pair | Trades | Total R | PF |
|---|---|---|---|
| EURUSD | 35 | +20.56 | 1.71 |
| USDJPY | 30 | +12.02 | 1.38 |
| AUDUSD | 29 | +5.14 | 1.49 |
| GBPUSD | 30 | +0.78 | 0.94 |
| USDCHF | 40 | −8.97 | 0.68 |
| USDCAD | 26 | −13.86 | 0.45 |

190 trades, +15.7 R, expectancy 0.082 R, 4/6 positive, worst PF 0.45. This is only partly
out-of-sample: different instruments, but the same 2024-2026 period and market regime.

---

## Key Findings

1. **The configuration does not survive out-of-sample.** +0.351 R/trade in-sample becomes
   −0.051 R/trade on 3.6× more data. Every improvement found in this study — pool selection,
   the 2.0R target, the entry arrays — was fitted to a single 2.3-year window.

2. **"All three instruments agree" was a false safeguard, and it was leaned on repeatedly.**
   Nasdaq, S&P and Dow over the same period are >0.9 correlated: they are one sample, not
   three. Roughly 25 configurations were searched against what was effectively n=1. The
   plateau in target sensitivity was likewise a plateau *within that one period*.

3. **Pool attribution is the substantive finding and may outlive the verdict.** ONS is
   +25.4 R while every other pool combined is −20.4 R, and the split is mechanically
   explained: `formed_at` locks each pool into a time slot, and only the 09:15–10:00 slot is
   profitable. The Chicago pit range is a futures-session concept applied to a CFD tracking a
   cash index; the 09:30 opening range cannot be swept until 10:00, after the profitable
   window has closed. Whether ONS itself holds out-of-sample was not isolated.

4. **The edge such as it is comes from the forced session flat, not from the ICT target
   rule.** EOD exits supply 44.9% of gross profit; on TP and SL alone the book is −1,393 pts.
   The 16:00 flat is a backtest convention, not a rule from any ICT source.

5. **Losses are an entry-quality problem.** 34% of losing trades never reached 0.25R. Exit
   management (breakeven stops, partial takes) was tested and could not fix this, because
   there is little to rescue.

6. **Execution cost is material and was previously invisible.** Spread alone takes 23.6% of
   net; break-even total cost is ~6.5 pts/trade. Before this study the engine used mid prices,
   filled stops at the exact stop level, and charged nothing.

7. **A prior "win rate" of 7.5% was a reporting artefact.** Target-hit and profitable-trade
   rates were conflated; profit factor was computed over TP/SL trades only, silently dropping
   EOD exits. Both are fixed (`classify_exit`, PF over all trades by P&L sign).

---

## Recommendations

1. **Do not trade this configuration.** (priority: high) The out-of-sample expectancy is
   negative on 506 trades.

2. **Invert the design/holdout split before any further work.** (high) Use 2015-2023
   (506 trades, 9 years, multiple regimes) to design, and hold 2024-2026 back entirely. This
   study did the reverse.

3. **Test ONS in isolation, out-of-sample.** (high) It is the only component with a
   structural story and a large in-sample effect. A single pre-registered hypothesis — "ONS
   high/low swept in 09:15–10:00 NY carries positive expectancy" — tested once on 2015-2023
   is worth more than another 25-config search.

4. **Cap the configuration search.** (high) Three or four candidates, each with a stated
   mechanism written down *before* the run. Twenty-five candidates against one period is how
   this study produced a +48.8R illusion.

5. **Stop counting correlated instruments as independent evidence.** (high) Report the
   number of independent *periods*, not the number of symbols.

6. **Add commission and slippage to the standard baseline.** (medium) Spread alone leaves
   only ~2.7 pts/trade of headroom before break-even.

---

## Open Questions

- Does ONS alone hold up on 2015-2023? The OOS run tested the full E6 stack; the pool
  component was never isolated out-of-sample.
- Is the 2015-2023 failure uniform across the nine years, or concentrated in particular
  regimes? Trade logs were not saved for those runs (`save_outputs=False`).
- Would the 16:00 flat — responsible for 45% of gross profit — behave differently at another
  hour? `eod_exit_hour` is now configurable but was never swept.
- Is the FX result (4/6 positive, expectancy 0.082 R) anything more than the same 2024-2026
  regime showing up again? An FX test on 2015-2023 would settle it; that data has not been
  obtained.
- Does the model's one-setup-per-session rule cost more than it saves, now that the earliest
  sweep is known to often be the worst one?
