# Analysis

Post-backtest research documents — one file per study, for cross-cutting studies; per-concept findings live in that concept's own `analysis/` folder under `concepts/`.

---

## File naming

```
analysis/{concept}/{model}-{concept}-{YYYY-MM-DD}.md
```

Examples:
```
analysis/daily-bias/model-2022-daily-bias-2026-06-13.md
analysis/entries/model-2022-fvg-entry-2026-07-01.md
analysis/killzones/model-2022-killzone-timing-2026-07-15.md
```

- **`{concept}`** — matches a concept category under `concepts/ict/` (e.g. `daily-bias`, `entries`, `killzones`, `pd-arrays`)
- **`{model}`** — the model being studied (e.g. `model-2022`, `romeo-crt`)
- **`{YYYY-MM-DD}`** — date the analysis was run, not the backtest date range

If a study spans multiple concepts, use the concept that is the primary focus.

---

## Required sections

Every analysis document must include:

```markdown
# {Model} — {Concept} Study

**Model**: model-name
**Concept**: concept-slug
**Instrument**: NQ | ES | YM | multi
**Analysis date**: YYYY-MM-DD
**Dataset**: description of date range and bar count
**Scripts**: scripts that produced the results
**Raw results**: paths to CSV outputs in results/
**Configs**: paths to YAML configs used (if any)
```

Then at minimum:
- **What Was Tested** — parameters, gate combos, or config variations
- **Results** — tables, key numbers
- **Key Findings** — numbered, one insight per finding
- **Recommendations** — concrete next actions with priority
- **Open Questions** — things not answered by this study

---

## What belongs here vs elsewhere

| Content | Goes in |
|---|---|
| Concept definitions, detection rules, ICT semantics | each concept's `<slug>.md` under `concepts/` |
| Human strategy write-ups from video transcripts | `strategies/notes/` |
| Backtest runner scripts | `backtests/` |
| Named experiment configs | `configs/{model}/` |
| Raw trade logs and equity curves | `results/` (gitignored) |
| **Quantitative findings from backtests** | **`analysis/`** ← here |
| Parameter grid search results and interpretation | `analysis/` |
| Session/killzone breakdowns | `analysis/` |
| Recommendations for model changes based on data | `analysis/` |

---

## Updating existing documents

Do not edit an existing analysis file to reflect new results. Instead, create a new file with the current date. The old file is a historical record of what was known at that point. Link between files using relative markdown links when a new study supersedes or extends an old one:

```markdown
*Supersedes: [model-2022-daily-bias-2026-06-13.md](model-2022-daily-bias-2026-06-13.md)*
```

---

## Index

| File | Model | Concept | Date | Summary |
|---|---|---|---|---|
| [model-2022-daily-bias-2026-06-13.md](daily-bias/model-2022-daily-bias-2026-06-13.md) | model-2022 | daily-bias | 2026-06-13 | Grid search across 12 gate combos; hard 4H gate best; London 0 trades; NY AM longs 0-for-7; 10-13:30 UTC window untapped |
| [currency-merchant-dxy-weekly-bias-plan-2026-06-14.md](dxy-weekly-bias/currency-merchant-dxy-weekly-bias-plan-2026-06-14.md) | currency-merchant | dxy-weekly-bias | 2026-06-14 | Build plan: weekly DXY pool reaction (MSS/CISD) → NQ weekly bias; in-progress |
| [judas-swing-execution-baseline-2026-09-06.md](../models/ict/judas_swing/analysis/judas-swing-execution-baseline-2026-09-06.md) | judas-swing | judas-swing | 2026-09-06 | Execution cost baseline: spread takes 43% of the edge (+0.337 -> +0.191 R); cost-adjusted OOS expectancy negative (-0.091 R); gap-fill bug that manufactured a -20R trade found and fixed; verdict RESEARCH |
| [model-2022-pool-attribution-2026-09-06.md](liquidity/model-2022-pool-attribution-2026-09-06.md) | model-2022 | liquidity-pools | 2026-09-06 | Pool attribution on index CFDs: ONS +25.4R vs every other pool -20.4R, timing mechanically forced by formed_at; ~25-config search reached +0.351 R/trade in-sample but -0.051 R/trade on 506 out-of-sample trades (2015-2023); verdict REJECT |
| [cross-model-repo-edge-audit-2026-09-07.md](entries/cross-model-repo-edge-audit-2026-09-07.md) | cross-model | entries | 2026-09-07 | Forensic audit of 8 reference repos (dextergsm = OPKYEI fork, byte-identical): best validated result anywhere is 52.1% WR / PF 1.52; OPKYEI's 68% WR claim contradicted by its own saved Sharpe 41 / Calmar 9576; islero PF 1.34 recomputes to 1.25; stop sizing is the largest measured lever (WR 24.8->40.3%); 9 ranked recommendations led by a written promotion gate and a leakage tripwire |
| [cross-model-winrate-payoff-frontier-2026-09-07.md](entries/cross-model-winrate-payoff-frontier-2026-09-07.md) | cross-model | entries | 2026-09-07 | 7 more repos (1 is a 404); 13 XAUUSD configs give corr(WR,payoff) = -0.945 and corr(WR,PF) = +0.667; 70-80% WR is real but always at 0.49-0.56R payoff, best PF 1.76; PF 2 at 76.7% WR needs only 0.608R (13% short); the impossible constraint is the 1:2 payoff, not the win rate; meta-labeling ablation-tested to a clean null (AUC 0.394) |
| [judas-swing-exit-policy-sweep-2026-09-08.md](../models/ict/judas_swing/analysis/judas-swing-exit-policy-sweep-2026-09-08.md) | judas-swing | entries | 2026-09-08 | 36 exit policies, entries frozen, replayed bar-by-bar; reproduction control caught 3 runner defects (drift 0.1461 -> 0.0001 R/trade); corr(WR,payoff) -0.953 replicates XAUUSD's -0.945 but corr(WR,PF) is -0.568 here vs +0.667 there - the PF gradient REVERSES because cost is a third of the payoff at high win rates; best change is breakeven-at-1.5R (PF 1.254 -> 1.614, OOS expectancy -0.021 -> +0.026); 0/36 promote on 18 OOS trades |
