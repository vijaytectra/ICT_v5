# Cross-Model — Reference Repo Edge Audit

**Model**: cross-model (audit input for judas-swing, model-2022, market-maker-models)
**Concept**: entries (trade selection + reward-risk gating)
**Instrument**: multi (evidence spans ES, SPY/QQQ/GLD/TLT, FX majors, crypto)
**Analysis date**: 2026-09-07
**Dataset**: no new backtest. Forensic read of 8 external repos in `_refs/` plus this
project's own two costed baselines.
**Scripts**: none — source audit only. Numbers are as published by each repo, or
recomputed from that repo's own published components where the two disagree.
**Raw results**: `_refs/*/README.md`, `_refs/fallenpheonix23_ict-trading-bot/DEVLOG.md`,
`_refs/OPKYEI_ICT-Trading/ICT_ml_trading/ICT_ml_trading/results/*.json`,
`models/ict/judas_swing/analysis/judas-swing-execution-baseline-2026-09-06.md`,
`analysis/liquidity/model-2022-pool-attribution-2026-09-06.md`
**Configs**: none

---

## What Was Tested

Nine repos were nominated for review. This audit answers three questions:

1. Which of them contain **validated** edge, as opposed to claimed edge?
2. Which specific mechanisms measurably moved win rate / profit factor, and by how much?
3. What is reusable for this project's stated goal — **high win rate, high profit
   factor, dynamic (non-fixed) reward-risk with a hard 1.5 R floor**?

**The nine are eight.** `dextergsm/ict-trading` is a zero-star fork of
`OPKYEI/ICT-Trading`; `CHALLENGES.md` and `README.md` are byte-identical
(md5 `30b15130…` and `a31c6ec8…`) and the GitHub API confirms
`fork: true, parent: OPKYEI/ICT-Trading`. Ranking them separately at #5 and #8
double-counts one codebase.

---

## Results

### Evidence table — every performance claim in the pile

Sorted by credibility, not by claim size.

| Source | Market | Trades | Win rate | PF | Avg RR | Costs? | OOS? | Verdict |
|---|---|---|---|---|---|---|---|---|
| **this repo** — judas-swing (costed) | FX basket | 73 | 31.5% | **1.24** | 3.35R win / 1.16R loss | spread yes, commission no | yes | **credible** |
| **this repo** — judas-swing OOS slice | FX basket | 18 | 27.8% | **0.90** | — | yes | is the OOS | **credible, negative** |
| **this repo** — model-2022 pool study | index CFDs | 506 OOS | — | — | — | yes | yes | **credible, negative** (−0.051 R/trade OOS) |
| fallenpheonix23 v3 | SPY+QQQ+GLD+TLT daily, 15y | 121 | **52.1%** | **1.52** | +0.38 avg R | no slippage (admitted) | no walk-forward | **best validated in pile** |
| fallenpheonix23 v2 | SPY 1h, 2y | 67 | 40.3% | 1.43 | — | no | no | credible |
| fallenpheonix23 v1 | SPY 1h, 2y | 202–333 | 24.8–27.3% | 0.55–0.64 | — | no | no | credible **failure** (−88% DD) |
| islero ICT-NT | **ES futures**, 10mo | 79 | 35.4% | 1.34 (see note) | **2.29** | not stated | no | medium — real engine, in-sample |
| GdotAiM ICT-Options-Agent | SPY options | — | — | gate PF ≥ 1.05 | — | — | has OOS gate | **self-declared synthetic** |
| sixscripts train-ict | FX journal | ~7 | 57.1% | — | — | live discretionary | no | not systematic |
| silotabs ict-trading-bot | Bybit crypto | — | — | — | — | — | — | **no published results at all** |
| OPKYEI / dextergsm | FX 60m | — | 68% claimed | — | — | claims fixed | claims embargo | **not credible — see below** |

### The OPKYEI number is leakage, and its own files prove it

`CHALLENGES.md` advertises XGBoost at 91% hold-out accuracy, Sharpe 2.14, 68% win
rate, and closes with *"the journey from 99% accuracy (with leakage) to 91%
(without leakage) represents moving from fantasy to reality."*

The saved result files in the same repo say otherwise:

| Pair | Sharpe | Sortino | Max DD | Calmar |
|---|---|---|---|---|
| AUDCHF 60m | 28.9 | 58.3 | 0.82% | 1,082 |
| EURCZK 60m | 31.8 | 66.9 | 0.31% | 1,053 |
| GBPJPY 60m | **41.5** | 98.1 | 0.95% | **9,576** |
| NZDUSD 60m | 31.9 | 74.4 | 0.36% | 3,856 |

A Sharpe of 41 and a Calmar of 9,576 are not a strategy. Renaissance Medallion ran
roughly Sharpe 2–3. These are the arithmetic signature of a model that can see the
bar it is trading. The repo documented the *right fixes* — embargo, shuffle test,
next-bar execution — and then shipped results showing the fixes did not land.

**Consequence for the ranking:** OPKYEI's ML layer is not a source of edge. Its
`CHALLENGES.md` anti-leakage checklist is genuinely valuable — as a *guard we run
against ourselves*, made more convincing, not less, by the fact that its own author
failed it.

### A second forensic note — islero's PF does not reconcile

islero publishes: 79 positions, 35.44% win rate, avg winner $713.21, avg loser
$312.04, total PnL $4,055.50, PF 1.3416.

Expectancy reconciles exactly: 79 × $51.34 = $4,055.86 ✓, and solving
`W×713.21 − (79−W)×312.04 = 4055.50` gives W = 28.0, i.e. 28/79 = 35.44% ✓.

But PF from those same components is `28×713.21 / 51×312.04` = **1.255**, not 1.3416.
The published PF is ~7% higher than its own inputs support — most likely computed
over the 246 *orders* rather than the 79 *positions*. Treat islero as
**PF ≈ 1.25, in-sample, costs unstated**, not 1.34.

### The win-rate arithmetic

Requested: 70–80% win rate at a minimum 1.5 R. That is not two goals, it is one
number. `PF = (WR × R) / (1 − WR)`:

| Win rate | at 1.5 R | Implied PF |
|---|---|---|
| 70% | 1.5 R | **3.50** |
| 80% | 1.5 R | **6.00** |
| 52.1% (best in pile) | 1.5 R | 1.63 |
| 35.4% (islero, ES) | 2.29 R | 1.25 |
| 31.5% (our judas, costed) | ~2.9 R | 1.24 |

Breakeven win rate at a given R is `1/(1+R)`: **40.0%** at 1.5 R, 33.3% at 2.0 R,
30.4% at 2.29 R, 28.6% at 2.5 R.

Nothing in eight repos plus this project's own two studies clears 52.1% on a
validated, cost-aware basis — and that 52.1% is a 121-trade, 15-year, daily-bar,
no-slippage result. The requested target is **2.3× to 4× the best number anyone in
this pile has produced.**

There is one honest way the *reported* win rate rises without inventing edge, and
fallenpheonix23 already uses it: **partial exit at 1.5 R, trail to breakeven**. A
trade that banks half at 1.5 R and stops the rest at BE books +0.75 R and counts as
a win. It converts variance into win-rate optics. It does not raise PF, and this
document will not present it as if it does.

### What measurably moved the numbers — ranked by observed effect

Every row is a *measured* before/after from a repo that published both sides.

| # | Change | Measured effect | Source |
|---|---|---|---|
| 1 | **Stop sizing**: fixed 0.1% → `max(0.5×ATR, 0.3%)` | WR **24.8% → 40.3%**, max DD **−88% → −16.4%** | fallenpheonix v1→v2 |
| 2 | **Short-side confirmation**: require price < SMA200 | bear-FVG WR **14.8% → 36.4%** | fallenpheonix v3 |
| 3 | **MSS quality**: 5-bar/1.0×ATR → 10-bar/1.5×ATR | 94 → **17** signals; direction stops flipping weekly | fallenpheonix v2 |
| 4 | **Regime gate**: ADX > 18 long, > 25 short | "significantly reduced false entries in choppy markets" (not quantified) | fallenpheonix v3 |
| 5 | **Drop breaker blocks** | +54 trades but DD **−16% → −36%** for only +$2.8k | fallenpheonix v2 |
| 6 | **Min FVG size ≥ 0.15×ATR** | removes micro-gaps that are bid-ask artefacts | fallenpheonix v2 |
| 7 | **Zone/MSS staleness expiry** (100 bars / 45–250 bars) | stops a months-old MSS gating every entry forever | fallenpheonix v2/v3 |
| 8 | **Realistic spread** | expectancy **+0.337 → +0.191 R**; OOS sign **flips negative** | **this repo**, judas baseline |
| 9 | **Same-bar breach ≠ gap** (fill-logic bug) | USDCAD **−1.990 → −0.306 R/trade**; killed a fabricated −20 R trade | **this repo**, judas baseline |

Row 1 is the largest single effect anywhere in the audit, and rows 8–9 are ours —
this project's execution layer is already more honest than six of the eight repos.

### What is worth taking, per repo

| Repo | Take | Leave |
|---|---|---|
| **fallenpheonix23** (ranked #4 → **should be #1**) | The whole improvement ledger, rows 1–7 above. The only repo that publishes a failure and the fix that cured it. | The engine (yfinance, daily bars, equities). |
| **islero ICT-NT** (#3 → **#2**) | `RewardRiskRatioRule` + `ExpectedTargetRule` — a structural target with a hard RR gate that *rejects* the trade. Exactly the shape you asked for, and the only ES-futures evidence in the pile. | Its published PF (use 1.25). NautilusTrader — we have our own event loop. |
| **sixscripts train-ict** (#6 → **#3, narrowly**) | `market_maker_model.py` — a working 682-line MMXM detector with consolidation, engineered-liquidity ladder (−1/−2/−3), MSS, entry zone and target ladder. This is the code our MMXM spec has been missing. Also `setup_grader.py`'s 10-point pre-trade score. | Its swing test reads `df.high.iloc[i+j]` — **forward-looking**; must be rewritten before use. Its `max_range = avg_atr × 1.5 × 10` is an unexplained magic number. No systematic backtest exists. |
| **GdotAiM** (#2 → **#4**) | The OOS **promotion gate**: ≥20 OOS trades, positive OOS expectancy, PF ≥ 1.05, drawdown cap, no severe OOS decay — deterministic, and the AI layer cannot override it. | The options layer. Its own README says the backtester uses synthetic bars and "is not evidence of live options profitability." |
| **OPKYEI/dextergsm** (#5 + #8 → **one entry, #5**) | The anti-leakage checklist only: embargo between folds, label-shuffle test must return ~50%, next-bar-open execution, ATR-based slippage, dedupe timestamps. | **Every performance number.** |
| **silotabs** (#1 → **#6**) | Operational hardening: kill switch, loss-streak halt, exposure caps, cancel-on-disconnect, market-data freshness. Useful the day we go live. | It is a Bybit crypto paper-trading system with **zero published results**. 45k LOC of infrastructure is not evidence of edge. Ranking it #1 ranks engineering effort as if it were alpha. |
| **futureteck** (#7) | Nothing. 548 LOC skeleton. | — |
| **zakariab0/ozo** (#9) | Nothing beyond a compact Judas structure we already have. | — |
| **OpenMobius-skill** (unranked, present locally) | A 2,008-card ICT knowledge base — a lookup peer to `_refs/ict-knowledge-library`. | Not a trading system. |

---

## Key Findings

1. **The nominated list has a duplicate and an inverted top.** `dextergsm` is a
   byte-identical fork of `OPKYEI`. And the two repos ranked #1 and #2 publish no
   validated performance at all, while the repo ranked #4 is the only one in the
   pile that shows a measured before/after for every change it made. Ranked by
   *evidence produced* rather than lines of code: fallenpheonix23, islero,
   sixscripts, GdotAiM, OPKYEI, silotabs.

2. **No validated source anywhere clears 52.1% win rate.** Across eight repos and
   this project's own two costed studies, the best cost-aware, out-of-sample-aware
   result is 52.1% WR / PF 1.52 on 121 trades over 15 years. The single claim above
   that (68%) sits in a repo whose own saved metrics show Sharpe 41 and Calmar 9,576.

3. **70–80% WR at ≥1.5 R is a request for PF 3.5–6.0.** That is 2.3–4× the best
   validated number in the audit. It is not reachable by combining these repos, and
   any backtest that appears to reach it should be treated as a leakage alarm and
   investigated before it is celebrated. This is not a reason to lower the ambition —
   it is the reason to make the *test* trustworthy first, so that a real improvement
   is distinguishable from a bug.

4. **Stop sizing is the highest-leverage parameter in the entire audit.** One
   change — a fixed 0.1% stop replaced by `max(0.5×ATR, 0.3%)` — moved win rate
   24.8% → 40.3% and max drawdown −88% → −16.4%. No signal filter in any repo comes
   close to that effect size. Our stored note that FX needs a 2–5 pip structural
   buffer is the same finding arrived at from the other direction.

5. **Our stops are small enough that costs are a first-order term.** Judas structural
   stops average 5–6 pips, so a 0.3 pip spread is ~5% of risk and consumes **43% of
   the edge**, flipping OOS expectancy negative. Any change that tightens stops to
   manufacture a higher R-multiple gets taxed harder. **A high nominal RR bought by a
   tight stop is not free — this is the specific trap the 1.5 R floor could walk us
   into.**

6. **The one ES-futures data point in the pile has the low-WR/high-RR shape, not the
   high-WR shape.** islero: 35.4% WR at 2.29 avg RR. Our judas: 31.5% at ~2.9 R. Two
   independent implementations of ICT logic on different markets land in the same
   region — ~30–35% WR carried by a 2.3–2.9 R payoff. That is what ICT edge appears
   to look like when it is measured honestly, and it argues the 1.5 R floor should be
   read as *a floor, not a target*.

7. **Dynamic RR is already solved in islero, and MMXM is the better target source.**
   `ExpectedTargetRule` takes the latest structural pivot as the target;
   `RewardRiskRatioRule` computes `reward/risk` and **rejects** below threshold.
   MMXM improves on it: the left side of the curve is an explicit *target ladder*
   (each engineered −1/−2/−3 level is a right-side objective), so RR is computed
   against a ranked series of real structural levels rather than a single pivot.

8. **The MMXM code we need already exists and is 60% right.** sixscripts'
   `MarketMakerModelDetector` implements consolidation detection, the engineered
   liquidity ladder, PD array, MSS, entry zone and a target ladder — the exact
   decomposition our spec calls for. Its swing test inspects `df.high.iloc[i+j]`,
   which is **forward-looking**, and its consolidation width uses an unexplained
   `×1.5×10` constant. It is a skeleton to port and fix, not to copy.

9. **Two repos independently converged on a promotion gate; we have none.**
   GdotAiM gates policy promotion on ≥20 OOS trades + positive OOS expectancy +
   PF ≥ 1.05 + no severe OOS decay. OPKYEI gates deployment on walk-forward
   Sharpe ≥ 0.8 + shuffled accuracy ≈ 50%. This project has walk-forward machinery
   (commit `a74ab57`) but no written pass/fail gate, so "is this good enough?" is
   currently decided per-conversation.

---

## Recommendations

Ordered so that each step's result is trustworthy before the next one is taken.

| # | Action | Rationale | Priority |
|---|---|---|---|
| 1 | **Write the promotion gate down as code before changing any model.** A config-level `PromotionGate`: ≥30 OOS trades, OOS expectancy > 0 after costs, OOS PF ≥ 1.15, OOS/IS expectancy decay < 50%, max losing streak < 15% of sample. A model that fails is not "close" — it is rejected. | Both our costed studies already produced negative OOS. Without a written gate we keep re-deciding what counts as success while the target moves. | **high** |
| 2 | **Add a leakage tripwire to the backtest engine.** A label-shuffle / date-shuffle run must collapse expectancy to ≈0. Wire it into the runner so it executes on every backtest, not on request. | The one repo claiming 68% WR is the one that skipped this. Given the 70–80% ambition, this tripwire is what will tell us a big number is real. | **high** |
| 3 | **Do the stop-sizing study before any signal work.** Sweep `max(k×ATR, floor)` against the current structural stop on judas + model_2022, reporting WR, PF, expectancy *and* cost-as-%-of-risk per variant. | Largest measured effect in the audit (finding 4), and it directly attacks our 43%-cost problem (finding 5) — a wider stop dilutes the spread tax for free. | **high** |
| 4 | **Implement dynamic RR as a rejection gate, not a fixed multiple.** Target = nearest *unswept* structural level (MMXM ladder where available, else swing pivot); `R = (target − entry)/(entry − stop)`; if `R < 1.5` → `actionable=False`, and log the rejection with its computed R. Then measure the rejected population — if rejected trades would have won, the gate is miscalibrated. | This is exactly what you asked for, it is proven in islero, and logging rejections turns the floor from a guess into a measurement. | **high** |
| 5 | **Build `models/ict/market_maker_models/market_maker_models.py`** by porting sixscripts' detector: fix the `i+j` forward look to a causal swing test, replace the `×1.5×10` constant with an explicit ATR-multiple parameter, and reuse our existing `SwingPointScanner`, `detect_mss`, `find_fvgs` and `liquidity_sweep`. Build the **retrospective labeler first** (how often does the curve complete?) before any prospective signal. | The spec's own Detection Rules already prescribe this decomposition, the ladder feeds recommendation 4, and MMXM is currently `detection: not-implemented`. | **medium** |
| 6 | **Port fallenpheonix's filter ledger as testable gates**, each behind a config flag and each measured alone: min-FVG-size ≥ k×ATR, zone staleness cap, MSS expiry, MSS displacement threshold, regime filter, counter-trend price confirmation. | Six pre-validated hypotheses with published effect sizes. Cheap to test, and one of them (counter-trend confirmation) tripled a losing sub-population's win rate. | **medium** |
| 7 | **Add partial-exit-at-1.5R + BE trail as a measured variant — and report it honestly.** Publish WR *and* PF side by side with the single-exit variant. | It is the legitimate route to a higher reported win rate, and it genuinely cuts variance. It must never be reported as a PF improvement, because it is not one. | **medium** |
| 8 | **Add commission as a real number.** At 5–6 pip stops a $3–7 round turn is another 5–10% of risk, currently modelled as zero. | Recommendation 5 of the judas baseline, still open. Any RR floor computed on uncosted R is optimistic by roughly one filter's worth. | **medium** |
| 9 | **Lift silotabs' operational risk controls only when we go live.** Kill switch, loss-streak halt, exposure cap, cancel-on-disconnect. | Correctly built and genuinely useful — but it is live-trading hygiene, not edge, and building it now delays every item above it. | **low** |

**Explicitly not recommended:** porting anything from OPKYEI/dextergsm's ML layer,
adopting silotabs' architecture wholesale, or treating any number in this audit
above 52.1% win rate as a target to reproduce.

---

## Open Questions

1. **Is 1.5 R the right floor for *our* markets?** islero and our judas both realise
   2.3–2.9 R on winners. A 1.5 R floor may admit trades well below the payoff the
   model actually delivers — the rejection log from recommendation 4 answers this
   with data rather than assumption.
2. **Does the RR floor interact with the stop-sizing study?** A wider stop lowers
   computed R and rejects more trades; a tighter stop raises R and pays more spread.
   These must be swept **jointly**, not one after the other.
3. **Does MMXM's target ladder actually resolve in sequence?** The spec asserts each
   left-side level becomes a right-side objective. Untested here. The retrospective
   labeler in recommendation 5 is what would establish it.
4. **What is the honest ceiling on ICT win rate for ES/NQ/YM 5m?** Two independent
   implementations converged on 30–35%. Whether that is the method's ceiling or two
   implementations sharing a weakness is unresolved — and it is the question that
   decides whether the 70–80% goal is reachable by any route.
5. **Are index futures even the right vehicle for the stated goal?** fallenpheonix's
   best result came from a *4-instrument daily portfolio* — diversification, not a
   better signal, produced the 52.1% win rate and the −5.12% max drawdown.
