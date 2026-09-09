# Cross-Model — Win-Rate / Payoff Frontier

**Model**: cross-model (audit input for the PF 1.54 production system)
**Concept**: entries (exit policy and its effect on win rate, payoff and profit factor)
**Instrument**: multi (XAUUSD M15, FX majors 1H, ES/NQ 1H)
**Analysis date**: 2026-09-07
**Dataset**: no new backtest. Forensic read of 7 further reference repos, of which
two carry usable evidence: 13 distinct backtest configurations in
`GifariKemal/xaubot-ai` (XAUUSD M15, 2025-08-01 → 2026-02-07) and a 10-upgrade
ablation ledger in `MoAbdullahi/ICT_Strategy_Video_Extraction`.
**Scripts**: none — source audit plus arithmetic on published components.
**Raw results**: `_refs/GifariKemal_xaubot-ai/backtests/*/*.log`,
`_refs/MoAbdullahi_ICT_Strategy_Video_Extraction/FINDINGS.md`,
`_refs/Musyimi97_phase404/README.md`
**Configs**: none

*Extends: [cross-model-repo-edge-audit-2026-09-07.md](cross-model-repo-edge-audit-2026-09-07.md).
**Corrects** that document's central design recommendation — see finding 3.*

---

## What Was Tested

Eight further repos were nominated. `Semordzi-Eric/ICT-Trading-Bot` returns
**HTTP 404** — it does not exist publicly. The remaining seven were cloned and
read against one question: does anything here move a validated system from
**PF 1.54** to **PF > 2.0** at **70–80% win rate**?

Five of the seven contribute no usable evidence:

| Repo | Stars | What it is | Usable evidence |
|---|---|---|---|
| `rcook0/SMC-ICT` | 9 | C# engine + MT4/MT5 EAs + Pine, 5 parallel versions | none published |
| `SiyabongaDlamini/SmartMoney.MQ5` | 7 | Single 1,555-line MQL5 EA, XAUUSD M1 | PF 1.20, 434 trades, ~50% WR |
| `Musyimi97/phase404` | 12 | MQL5 EA + SMT filter | thresholds only, no results |
| `Aditya18487/mt5-trading-bot` | 13 | Python SMC bot; "Monte Carlo" is a plotting function, not a gate | none |
| `lordgaruda/XAU-60` | 49 | CRT+TBS: Asia range → killzone → sweep → close back inside | none published |

`lordgaruda`'s `SweepQuality` enum (WEAK/MODERATE/STRONG/PERFECT) is the only
mechanism worth noting, and this repo already has an equivalent in
`_sweep_quality_ok`. **No new mechanism was found in any of the seven.**

The two that matter carry evidence rather than code.

---

## Results

### 1. The frontier — 13 configurations, one instrument, one window

`GifariKemal/xaubot-ai` (72 stars, 94k LOC, XAUUSD M15) ran 30+ numbered
ablations. Thirteen produced parseable performance logs. Win rate, profit factor
and payoff (avg win ÷ avg loss) recomputed from each log's own components:

| Configuration | n | Win rate | PF | payoff |
|---|---:|---:|---:|---:|
| 08_stoch_sell | 416 | **76.7%** | **1.76** | 0.54 |
| 06_stochastic | 580 | 75.9% | 1.55 | 0.49 |
| 02_earlycut_improved | 665 | 73.7% | 1.40 | 0.50 |
| 05_sellfilter_only | 495 | 73.3% | 1.53 | 0.55 |
| 01_smc_only (synced) | 686 | 72.2% | 1.42 | 0.55 |
| 03_sellfilter_pullback | 452 | 70.4% | 1.32 | 0.56 |
| 07_ema_stack | 640 | 69.5% | 1.38 | 0.60 |
| 04_pullback_only | 635 | 69.0% | 1.09 | 0.49 |
| 11_broker_sl | 421 | 65.1% | 1.36 | 0.73 |
| 12_stoch_sell_broker_sl | 294 | 62.9% | 1.20 | 0.70 |
| 14_stoch_sell_patient | 375 | 61.9% | 1.26 | 0.78 |
| 13_patient_exit | 602 | 61.6% | 1.26 | 0.78 |
| 01_smc_only (raw, no position manager) | 464 | 47.6% | 1.21 | **1.33** |

- **corr(win rate, payoff) = −0.945.** Not a correlation — a constraint. Every
  configuration that bought win rate paid for it in payoff, at close to a fixed
  exchange rate.
- **corr(win rate, PF) = +0.667.** Mean PF above 70% win rate is **1.497**;
  below, **1.251**.
- The single change from the bottom row to the fifth row is the
  SmartRiskManager / SmartPositionManager exit layer: same signals, 47.6% → 72.2%
  win rate, payoff 1.33R → 0.55R, PF 1.21 → 1.42.

### 2. The arithmetic the frontier implies

`PF = WR × payoff / (1 − WR)`, so `payoff_required = PF(1 − WR) / WR`:

| Win rate | payoff needed for PF 2.0 | payoff needed for PF 2.0, vs. observed |
|---|---:|---|
| 60% | 1.333 R | far above the 0.78 R seen at 62% |
| 70% | 0.857 R | above the ~0.56 R seen at 70% |
| **76.7%** | **0.608 R** | **they achieved 0.54 R — short by 13%** |
| 80% | 0.500 R | at or below what the high-WR end already delivers |

And the same win rates at a **1:2 payoff**:

| | |
|---|---|
| 70% win rate @ 2.0 R | PF **4.67** |
| 80% win rate @ 2.0 R | PF **8.00** |

### 3. `MoAbdullahi` — ten upgrades, eight rejected

A full ablation ledger with committed trade logs, and the most rigorous negative
result across all fifteen repos surveyed:

| Upgrade | Verdict | Evidence |
|---|---|---|
| FVG quality scoring | **kept** | +4.7 R vs baseline, stable across halves |
| Partial targets + breakeven | **kept** | max DD −15.5% → −8.5%, win rate → 40% |
| DXY SMT divergence | opt-in | PF 1.21/1.29, halves DD — but hurt GBPUSD |
| Killzone hours | rejected | −0.5 R; "profitable hours" flip sign across halves |
| Displacement CISD | rejected | −8.8 R |
| Day-of-week filter | rejected | mined from the backtest's own output |
| News avoidance | untested | no reliable free calendar |
| Power of Three | rejected | −0.07 R refined, −3.32 R alone |
| Dynamic quality sizing | rejected | same R, worse equity — amplifies variance |
| **Meta-labeling (Random Forest)** | **clean null** | **CV AUC 0.394** (worse than a coin flip); the gatekeeper's confident picks lost −1.37 R against +2.79 R unfiltered |

Their cross-pair result on identical parameters: EURUSD PF 1.02, GBPUSD 1.01,
USDJPY 1.01, **AUDUSD 0.68**. And on ES/NQ their apparent PF ~1.25 decomposed to
longs +11.5 R / shorts −3.0 R (ES) and +13.2 R / −2.8 R (NQ) across 2024–2026 —
directional beta in a bull market, not alpha.

### 4. A practitioner's own acceptance bar

`Musyimi97/phase404`, unprompted, in its README:

> "SMC strategies typically produce 40–50% win rates with high RR ratios (1:2 to
> 1:5). **Do not optimise for win rate alone** — focus on profit factor and drawdown."

Its own grading: PF > 1.2 minimum / > 1.5 good / > 2.0 excellent; win rate > 35%
minimum / > 45% good / **> 55% excellent**.

---

## Key Findings

1. **Win rate and payoff are one number, not two.** At −0.945 correlation across
   13 configurations, they are not independent knobs. The exit policy chooses a
   point on the curve; it does not move the curve. The user's system at PF 1.54
   and the 76.7% / 0.54 R configuration are plausibly the same edge read at two
   different points.

2. **70–80% win rate is real and reachable — and it does not mean what it sounds
   like.** Six of thirteen configurations exceeded 70%. Every one of them ran a
   payoff between 0.49 R and 0.56 R. A 70%+ win rate in this family of strategies
   is produced by early and partial exits, which is the same thing as selling the
   right tail.

3. **The earlier audit's central recommendation pointed the wrong way for this
   goal.** [`cross-model-repo-edge-audit`](cross-model-repo-edge-audit-2026-09-07.md)
   argued "hold R high and let win rate be the smaller lift", on the reasoning
   that a smaller R is taxed harder by costs. That reasoning still holds for
   costs, but on the only dataset that has actually swept this axis, **PF was
   higher at the high-win-rate end** (mean 1.497 above 70% vs 1.251 below), and
   the best PF observed sat at the highest win rate. For a PF-2 target the
   high-win-rate end is the side worth testing, not the side to avoid.

4. **The impossible constraint is the 1:2 payoff, not the win rate.** PF 2.0 at
   76.7% win rate requires a payoff of only **0.608 R**, and the best observed
   configuration reached 0.54 R — a 13% shortfall. But 70–80% win rate *at a 1:2
   payoff* implies **PF 4.67 to 8.00**, which nothing in fifteen repos approaches
   by a factor of three. The three constraints are satisfiable in pairs, never as
   a triple.

5. **The best number in this batch is weaker evidence than the system we already
   have.** PF 1.76 comes from a 6-month window (2025-08 → 2026-02), one
   instrument, no out-of-sample split, and it is the **maximum of 13
   configurations tried on that same window** — the textbook setup for selection
   bias. The production system's PF 1.54 carries 12 years, a 1200/1200 random
   control and P(edge) 99.8%. On evidence quality these are not close.

6. **Meta-labeling has now been tested and failed.** It was the strongest
   remaining candidate for raising precision on an existing signal without
   rebuilding it — a secondary model that predicts whether the primary signal
   wins. `MoAbdullahi` implemented it and got CV AUC 0.394, worse than random,
   with the filtered subset losing money the unfiltered set made. One negative
   result is not proof, but it removes this from the "obvious untried idea" list.

7. **Eight of ten plausible upgrades failed ablation, and the two that survived
   are already in our system.** FVG quality scoring and partial-target-plus-
   breakeven were kept; killzone hours, CISD displacement, day-of-week, PO3 and
   dynamic quality sizing were all rejected — several of them ideas this project
   has recommended or scheduled.

---

## Recommendations

| # | Action | Rationale | Priority |
|---|---|---|---|
| 1 | **Sweep the exit policy along the frontier, not the signal.** Take the existing PF 1.54 system, hold entries frozen, and vary only the exit: partial fraction, partial trigger R, breakeven trigger, time stop. Record win rate, payoff and PF for each point. | This is the one axis with published evidence of a PF gradient (finding 3), it costs no new alpha, and the system's own validation machinery can score it. | **high** |
| 2 | **Target payoff 0.6–0.9 R at 70–80% win rate as the explicit PF-2 route**, and drop the 1:2 RR floor for that variant. | Finding 4: the floor and the win rate cannot both hold. At 76.7% the requirement is 0.608 R, which is 13% above what has already been demonstrated in-sample elsewhere. | **high** |
| 3 | **Attack cost per trade before attacking signal.** PF is a post-cost number; the judas baseline measured spread at 43% of gross edge. A high-win-rate/low-payoff policy makes this worse, so pair recommendation 1 with an explicit cost-per-R budget per exit variant. | The only lever that raises PF without requiring new alpha — and the one the "every PF-2 attempt failed" history has probably not covered, since those were signal-side attempts. | **high** |
| 4 | **Do not implement meta-labeling, killzone-hour filters, day-of-week filters, PO3 gating or confidence-weighted sizing** without new evidence. | All five were ablation-tested and rejected in `MoAbdullahi`, with logs. Re-deriving them costs weeks and the prior is now against them. | **medium** |
| 5 | **Keep the 12-year system as the incumbent and require any challenger to beat it on the same protocol.** PF 1.76 on 6 months in-sample does not qualify. | Finding 5. The `evaluate_gate` thresholds in `infrastructure/promotion.py` already encode this. | **medium** |
| 6 | **Stop surveying repos.** Fifteen reviewed; the last seven produced one usable dataset, one usable ablation ledger and zero new mechanisms. | Diminishing returns are now unambiguous. | **medium** |

---

## Open Questions

1. **Where on the frontier does the PF 1.54 system currently sit?** Its win rate
   and payoff decomposition were not supplied. If it is already at the high-win-rate
   end, recommendation 1 has little room; if it is at the 40–50% end, the
   untested span is large.
2. **Does the +0.667 WR/PF correlation survive out-of-sample?** It is measured
   across 13 configurations on one 6-month window of one instrument. It may be an
   artefact of that window's trend persistence — gold trended strongly through
   2025-08 → 2026-02, which flatters early-exit policies.
3. **Is the frontier's slope instrument-dependent?** All 13 points are XAUUSD.
   Index futures have different intraday mean-reversion characteristics, and the
   exchange rate between win rate and payoff may not be the same.
4. **How much of the 43% cost share is recoverable?** If spread is a fixed
   fraction of a shrinking payoff, the high-win-rate end of the frontier may be
   unreachable net of costs even where it is reachable gross.
