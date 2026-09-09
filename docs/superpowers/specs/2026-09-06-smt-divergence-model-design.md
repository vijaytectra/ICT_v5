# SMT Divergence Model — Design

**Date**: 2026-09-06
**Status**: design, awaiting review
**Layer**: concept rewrite (`concepts/ict/entries/smt_divergence/`) + new model (`models/ict/smt_model/`)
**Instruments**: ES / NQ / YM index CFDs (2015-01-01 → 2026-09-04) and
EURUSD / GBPUSD / DXY (2024-01-02 → 2026-09-04, extending to 2015 when the
download lands)

---

## 1. Why this document exists

The repo already has an `smt-divergence` concept with `detection: implemented`.
It is not usable. `detect_smt` compares `df['high'].iloc[-3:].max()` against
`df['high'].iloc[-6:-3].max()` — a rolling-window comparison with no swing
anchor, no liquidity pool, no time alignment between instruments, and no
expiry. It will fire on noise at every bar. The spec beside it already
describes the correct behaviour; the code never implemented it.

This design replaces the detector, records the doctrine the spec is missing,
and defines the experiment that decides whether SMT earns its place in a model
at all.

## 2. What the corpus actually specifies

Read from `_refs/ict-knowledge-library`: the five `16-smt-divergence` pages plus
the primary transcripts (Month 10 Index Futures AM / PM / Relative Strength,
Month 02 High Reward Setups, Month 05 Qualifying Trade Conditions).

| Rule | Citation | In repo spec? |
|---|---|---|
| AM comparison window **05:00 → 09:30 NY** | M10-AM `[07:57]`, `[13:41]` | no |
| PM comparison window **12:00 → 15:00 NY** | M10-PM `[07:34]` | no |
| Bullish order flow → compare **lows**; bearish → compare **highs** | M10-AM `[08:05]`, `[10:08]` | no |
| Compare **relative extremes inside** the window, not the window bounds | M10-AM `[10:42]`–`[12:59]` | no |
| Compare the **same time-of-day candle** on both charts | KB `correlated-pairs-smt` | partially |
| SMT appears near-daily; only "a few times a week" are tradeable | M10-AM `[16:59]`–`[17:20]` | no |
| "When it's not obvious, assume it's not there" | M10-AM `[17:38]` | no |
| Divergence alone is not a trade — HTF order flow sets the tone | M05 `[07:15]`, M10-RS `[07:15]` | no |
| Hold for time of day: AM → 10:30–11:00 min; PM → 15:00 bond close | M10-INDEX `[02:41]`, `[03:00]` | no |
| SMT is confluence, never a standalone trigger | KB `smt-confirmation`, M02 `[43:48]` | yes |

### 2.1 An unresolved contradiction in the source

Which instrument to trade after a divergence has **three** answers in the corpus:

1. **Turtle-soup the leader** — "the one that does go lower, that's a turtle
   soup … buy it below the old low" (M10-AM `[16:22]`, `[18:30]`).
2. **No preference** — "there's not one over the other in terms of advantage.
   They're all three going to move in tandem" (M10-AM `[16:46]`).
3. **Always one fixed instrument** — "if I see that crack among all three my
   trade's going to be taken in the S&P. I'm not trying to trade the Dow or the
   Nasdaq" (M10-PM `[10:10]`).

The repo spec asserts (1)'s opposite — "trade the instrument that refused to
confirm" — citing the 2020 forex notes. All three are carried as a tested
parameter (`instrument_selection`), not assumed. Section 6 records what the
exploratory data already says.

## 3. Exploratory findings that shape this design

Measured on 806,715 aligned 5m bars, 3,645 sessions, 2015-01-01 → 2026-09-04.
Scripts are throwaway (scratchpad); the numbers are reproduced as a first task
of implementation into `models/ict/smt_model/analysis/`.

**Method**: within each window, detect relative swing extremes (5m, k=2) on any
of the three indices, dedupe within 2 bars, take the last two pivots, and read
every instrument's local extreme at those same two timestamps. A mixed reading
(1 or 2 of 3 made the new extreme) is an SMT; 0 or 3 is tandem. Forward outcome
measured from the signal bar close to 16:00 NY, normalised by each instrument's
own 14-day ADR.

**F1 — frequency matches doctrine.** 3,161 SMT events over 3,005 sessions =
1.05/session = **5.3/week**, against ICT's "a few times a week". The detector is
faithful.

**F2 — raw SMT has no edge.** Against the tandem baseline, SMT is *worse* in 2
of 4 window/side cells (AM-high +0.006 vs +0.025; PM-high +0.034 vs +0.060 ADR).
Unconditional SMT is not a signal.

**F3 — the bias gate is the load-bearing filter.** Bias-aligned PM SMT returns
+0.082 ADR (MFE/MAE 1.73, 62.1% positive) against +0.021 for counter-bias SMT
and +0.049 for bias-aligned tandem.

**F4 — the PM window carries the signal; the AM window does not.** AM
bias-aligned SMT (+0.040) is *worse* than AM bias-aligned tandem (+0.044). The
famous 05:00–09:30 window does not separate in this dataset. PM (12:00–15:00)
also has the lowest MAE of any group measured (0.148 ADR), which is what makes a
tight stop survivable there.

**F5 — sweep depth is monotone.** Deepest quartile 62.1% positive / +0.086 ADR
versus shallowest 53.9% / +0.052. A deeper raid is a better setup.

**F6 — clarity is a band, not a floor.** Divergence magnitude quartiles run
+0.043 / +0.077 / **+0.089** / +0.028, with MAE *highest* in the clearest
quartile (0.391). The largest divergences are decorrelation events, not
engineered sweeps. `min_clarity_adr` **and** `max_clarity_adr` are both
parameters. This contradicts the universal retail assumption that a bigger
divergence is a stronger signal.

**F7 — instrument selection.** diverging +0.058 (58.4% positive) ≈ always-ES
+0.059 (57.3%) > leader/turtle-soup +0.046 (56.9%). The repo's rule survives but
is statistically tied with a fixed instrument; turtle-soup is worst.

**F8 — the edge decays.** +0.093 (2015-2018) → +0.057 (2019-2021) → +0.033
(2022-2026). This dictates the out-of-sample split.

> Caveat carried into every downstream document: F1-F8 measure **raw forward
> drift from the signal bar** with no entry, stop, target, pool requirement or
> reversal confirmation. They locate where to point the model. They are not a
> verdict on it.

## 4. Architecture

### 4.1 Concept layer — rewrite `smt_divergence.py`

Replaces the rolling-window detector. Public API:

```python
@concept("smt-divergence", depends_on=["swing-points", "liquidity-pools"])
def find_smt_events(
    correlated: dict[str, pd.DataFrame],   # aligned OHLCV, same index
    window: tuple[str, str],               # NY-local ('12:00','15:00')
    session_date: pd.Timestamp,
    side: str,                             # 'low' (bullish) | 'high' (bearish)
    swing_lookback: int = 2,
    pivot_dedupe_bars: int = 2,
    match_tolerance_bars: int = 1,
) -> list[SMTEvent]
```

`SMTEvent` carries `{time, side, leader[], diverging[], pivot_times,
levels_by_instrument, clarity_adr, sweep_depth_adr, pool_ref}`.

Two rules the old detector had no concept of:

- **Same-timestamp comparison.** Pivots are found on the union of instruments,
  deduped, and every instrument is then read at those same timestamps
  (±`match_tolerance_bars`). Valid here because the index CFDs and FX pairs are
  100% bar-aligned (verified: 918,785/918,785 and 3,114,741/3,114,741).
- **Negation (`smt-failure`).** `negated_at(df, event)` returns the first time
  the diverging instrument takes its own compared level. A negated event must
  not fill an order.

`smt-divergence.md` and `smt-failure` guidance are updated in the same commit —
windows, bias-gating, the clarity band, the negation rule and the unresolved
instrument-selection question all become part of the spec, per the CLAUDE.md
code/semantic sync rule.

### 4.2 Model layer — `models/ict/smt_model/`

Follows `models/README.md`: `MarketSnapshot` / `generate_signal`, rule of
exclusion, config-driven. Every price bound is expressed as a fraction of the
instrument's own ADR so one config runs unchanged across index CFDs and FX —
the mechanism `judas_fx.yaml` uses to avoid per-instrument curve-fitting.

**Checklist (all must pass, else `actionable=False`):**

1. **Bias** — `daily_bias` (intermediate) returns non-neutral. Chooses the side:
   bullish → compare lows, bearish → compare highs. Not optional (F3).
2. **Window** — inside AM (05:00–09:30 NY) or PM (12:00–15:00 NY), DST-aware via
   `ny_index`. Configurable per run; PM is the default (F4).
3. **Pool** — the compared extreme is a real liquidity pool: prior-session
   extreme, PDH/PDL, or a relative-equal cluster from
   `find_relative_equal_levels`, and still live per `pool_is_live`.
4. **Sweep** — the leader trades through the pool and is rejected
   (`detect_liquidity_sweep` → `classification == 'sweep'`), with
   `sweep_depth_adr >= min_sweep_depth_adr` (F5).
5. **SMT** — a non-negated `SMTEvent` with
   `min_clarity_adr <= clarity_adr <= max_clarity_adr` (F6).
6. **Reversal** — `detect_cisd` or `detect_mss` within
   `max_minutes_to_reversal` of the sweep, with displacement quality.
7. **Entry** — limit into the FVG/OB left by the displacement (also testing
   market-on-confirmation and ICT's buy-stop-above-the-down-close-candle).
8. **Stop** — beyond the swept extreme + buffer, clamped to
   `[min_stop_adr_pct, max_stop_adr_pct]`. The floor is mandatory: the Judas
   post-mortem showed 5-6 pip stops surrendered 43% of the edge to spread.
9. **Target** — resolved by the exit study (§5).
10. **Negation** — cancel an unfilled order the moment the SMT is negated.

## 5. The experiment

This is the deliverable, not a by-product.

### 5.1 Does SMT add anything? (the primary question)

| Arm | Description |
|---|---|
| **A** | Full model, SMT gate ON |
| **B** | Byte-identical model, SMT gate OFF (bias + pool + sweep + reversal only) |
| **C** | SMT alone, no sweep/reversal — deliberate strawman |

`A − B` is the honest measure of SMT's contribution. Given F2, B matching A is a
live possibility and will be reported as such.

### 5.2 Parameter axes (frozen until §5.1 resolves)

- `window`: PM / AM / both
- `instrument_selection`: diverging / leader / fixed
- `min_sweep_depth_adr`, `min_clarity_adr`, `max_clarity_adr`
- `entry_style`: limit / market / buy-stop
- `max_minutes_to_reversal`, `confirmation`: cisd / mss / either

### 5.3 Exit study — run on one frozen trade set

- Fixed RR: 1R, 1.5R, **2R (the stated anchor)**, 2.5R, 3R, 4R
- Liquidity target (opposing pool), with and without an ADR cap
- **Time-of-day exit** — AM held to 10:30/11:00, PM to 15:00 bond close.
  Doctrine-specific (M10-INDEX `[02:41]`) and, as far as I can tell, never
  backtested anywhere.
- Partials 50% at 1R / 1.5R / 2R with remainder to target, ± breakeven
- Trailing: swing-based and ATR-based
- Breakeven trigger at 0.5R / 1R / 1.5R
- MFE/MAE distribution, so the RR choice is read off where price actually goes

### 5.4 Cost regimes

Zero-cost control (reproduction check) · historical per-bar spread ·
1.5× stressed · 2× stressed · futures-commission variant. Using
`infrastructure/execution.py` with `gap_fill='open'`.

### 5.5 Validation

- **Indices**: in-sample 2015-01-01 → 2021-12-31; **out-of-sample 2022-01-01 →
  2026-09-04**, chosen because F8 shows the edge is weakest there.
- **FX**: same config, 2024-2026 now, extending to 2015 on download. Cross-market
  agreement is the robustness evidence.
- Per-year and per-instrument breakdowns reported for every arm.

## 6. Deliverables

1. Rewritten `concepts/ict/entries/smt_divergence/smt_divergence.py` + updated
   `smt-divergence.md`, with tests in that folder's `tests/`.
2. `models/ict/smt_model/{smt_model.py, smt-model.md, __init__.py}`.
3. `models/ict/smt_model/backtests/` — single-instrument runner, portfolio
   runner, rejection funnel, exit study, cost-regime runner, configs.
4. `models/ict/smt_model/analysis/smt-model-baseline-<completion date>.md`
   following `analysis/README.md`, with the index-table row added. A second doc
   records the F1-F8 exploratory results reproduced as committed scripts.
5. Registry lineage updated (`infrastructure/lineage.py --update-readme`).

## 7. Non-goals

- No per-instrument parameter tuning, and no dropping losing instruments from
  the basket. A negative instrument is evidence against the model.
- No optimisation before the out-of-sample split is drawn.
- No claim of validation from in-sample numbers.
- If arm A ≈ arm B, the finding is reported as "SMT adds nothing measurable
  here", not tuned until it looks better.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Gate stack over-filters to too few trades | Rejection funnel reports the survivor count at every checklist step |
| Clarity band is fitted to the exploratory sample | Band bounds are set in-sample only and held fixed through OOS |
| 2015-2018 index CFD data quality differs from 2024+ | Per-era spread and bar-count reported before any result is trusted |
| Multiple-comparison risk across many arms | Arm count fixed in advance in this document; OOS is touched once |
