---
name: judas-swing
aliases: [judas, judas swing, opening judas, false-direction open, market protraction, london trap]
category: models
related:
  - power-of-three
  - liquidity-sweep-stop-hunt
  - change-in-state-of-delivery
  - market-structure-shift
  - fair-value-gap
  - killzones
  - sessions-and-ranges
  - true-day-midnight-open
  - daily-bias
  - targets-and-exits
  - turtle-soup
parameters:
  sessions: [london, ny]
  london_anchor: "00:00 NY"
  london_judas_minutes: 210
  london_kz_end: "05:00 NY"
  ny_anchor: "07:00 NY"
  ny_judas_minutes: 150
  ny_kz_end: "11:00 NY"
  asia_anchor: "20:00 NY"
  bias_mode: strict
  close_back_within_bars: 3
  min_sweep_depth_atr: 0.05
  deep_judas_pct: 0.40
  max_minutes_to_reversal: 45
  confirmation: cisd_or_mss
  entry_style: limit
  fvg_entry_depth: ce
  target_min_rr: 2.5
  adr_days: 5
  adr_target_pct: 0.90
  skip_if_prior_session_delivered: 0.80
  max_range_used_pct: 0.85
detection: implemented
---

# Judas Swing

The **manipulation leg of the session, personified**: the engineered false move
shortly after a session opens, running against the day's true direction to sweep
liquidity and trap traders before the real move. Named for the betrayal — the
session's first directional move "betrays" whoever follows it.

The pattern is ICT's *market protraction* under its more famous name: "a small,
counter-directional impulse price swing anchored to a specific time of day",
whose "design is to manipulate the sentiment ... of traders wanting to be
participants". The time anchor is the whole definition — an impulse swing without
a clock is just a leg.

**Four sessions have one, not two.** ICT enumerates London open, the CME open
(New York), Asia at 20:00 NY / 0 GMT, and London close. Three of the four fake
*against* the coming delivery; the London-close instance is the exception — it
*extends* the day's direction, prints the day's extreme, and is then faded. This
model implements the three counter-directional ones and defaults to the two ICT
tells students to focus on: London and New York. Asia and London close are, in
his words, "the two smallest tiny little windows of opportunity" that "[don't]
pay out enough ... to take on the risk".

**Frequency is about one setup per session across a basket of four or five
pairs — not one per pair per day.** A model that fires daily on a single
instrument is not finding Judas swings.

## Detection Rules

The rule of exclusion applies: every item must pass simultaneously, or
`actionable=False`.

### 1. Bias — the direction the Judas opposes

The reversal must agree with the higher-timeframe read; "a Judas swing without
HTF bias confirmation is just chop". Which read supplies it is a named choice
(`bias_mode`), because the corpus offers several:

| mode | rule |
|---|---|
| `strict` | the project's daily-bias gate — daily order flow **and** draw-on-liquidity must agree, 4H may veto |
| `order_flow` | daily order flow alone over `daily_lookback` bars |
| `draw` | the nearest unswept daily pool |
| `week_open` | the ICT day-trading model's own filter: this week's opening price projected forward — above it = buy days, below it = sell days |

Bias bullish ⇒ trade direction long ⇒ **the Judas is a bearish leg raiding
sell-side liquidity**, and vice versa.

### 2. Anchor — the protraction window

The counter-move must *begin* inside the session's Judas window, measured on the
New York clock (the GMT equivalences drift with DST; the NY column governs).

| session | anchor | window | killzone ends | positions collapsed |
|---|---|---|---|---|
| London | **00:00 NY** | +210 min (to 03:30) | 05:00 | 12:00 |
| New York | **07:00 NY** | +150 min (to 09:30) | 11:00 | 16:00 |
| Asia | 20:00 NY | +60 min | 23:59 | 23:59 |

The London anchor is **midnight, not 02:00**. The 2016 protraction lecture puts
the London window "after midnight New York time"; the 2017 intraday-profiles
lecture adds the *delayed* case — "draw a vertical line at 2 a.m. ... between 2
o'clock and 3 a.m., that Judas swing should occur". A 00:00 anchor with a 3.5-hour
window spans both readings. A 02:00 anchor sees only the delayed one, and worse,
treats an on-time protraction as a pool that was already mitigated — which is why
the first implementation of this model fired seven times in two years.

### 3. Sweep — which pool, and was it a raid or a breakout

The pool set **rotates with the session**: London raids Asia's stops, New York
raids London's, Asia raids the prior late-session swing.

| session | pools |
|---|---|
| London | Asian range (20:00–00:00 NY, prior evening) high/low, PDH/PDL |
| New York | London range (02:00→anchor) high/low, Asian range, PDH/PDL |
| Asia | prior late-session (13:00→anchor) swing high/low, PDH/PDL |

A pool only counts if the **first** trade through it since it formed lands inside
the Judas window — a level taken earlier is already mitigated, and revisiting it
is not a fresh liquidity grab.

Telling a stop hunt from a breakout at that level is the single most important
filter in the model, and the corpus gives the test outright: **candle 2 or
candle 3 must close back inside the range**. Implemented as
`close_back_within_bars` (default 3), on the entry timeframe. Demanding the
close-back of the *breach bar alone* throws away most genuine raids, because the
rejection normally completes on the next bar or two.

Two depth guards frame it: `min_sweep_depth_atr` (a one-tick graze is not a
raid) and the optional `max_sweep_depth_atr` (an excursion far past the level is
delivery, not manipulation).

### 4. Depth — the protraction must be *small*

Protraction is "explicitly a *small* swing relative to the delivery". The
operational form comes from the Mobius card: a **deep** counter-move — one that
consumes roughly half the period's range — caps the expansion, and the target
must then be cut back toward the opening price rather than projected to a full
range objective.

Measured as `|anchor price − Judas extreme| / ADR5`. Past `deep_judas_pct`
(default 0.40) the setup is either skipped (`deep_judas_action: skip`) or kept
with the target reduced to the session open (`open_target`).

### 5. Reversal — or Judas failure

CISD and/or MSS in the bias direction (`confirmation`), with a displacement-quality
check on the confirming candle (body dominance, opposing wick), **inside the
killzone** and within `max_minutes_to_reversal` (default 45) of the sweep.

This is where a Judas is separated from a **Judas failure**: "by 30–45 minutes
into the killzone, if price is still extending in the initial direction ... treat
the Judas hypothesis as failed". The taught response is to stand down, not to
wait longer, so no reversal inside the deadline returns `actionable=False`.

The **Judas extreme** — the furthest point reached between the sweep bar and the
confirmation — is recorded here, not the sweep bar's own wick. That is the level
the setup is invalidated beyond.

### 6. Entry

Two styles, both in the corpus:

- `limit` (default) — the retrace into the FVG or order block the displacement
  left, taken at **consequent encroachment** (`fvg_entry_depth: ce`), the gap's
  midpoint. This is ICT's standard play, and the Mobius cards insist on it
  ("wait for the deeper discount/premium rather than entering at the order block
  or the daily open"). The limit expires at `entry_fill_minutes` **or the end of
  the killzone, whichever is first** — the institutional participation the setup
  rides ends with the window, so a fill after it is a different trade.
- `market` — the close of the confirmation candle, filled at the next bar's open.
  Worse price, no fill risk, and the stop still sits at the Judas extreme.

### 7. Stop

Beyond the **Judas extreme**, plus `stop_buffer_pts` (ICT's 2–5 pips on FX / 2–5
points on indices). A stop pinned pixel-precisely at the invalidation is a named
failure mode; so is replacing the structural anchor with a fixed or ATR stop.

Two floors and a ceiling reject setups the stop cannot express: `min_stop_pts`
(absolute), `min_stop_atr_mult` (relative — the structural level must sit outside
one bar's worth of noise), and `max_stop_pts` (a Judas deeper than this was
delivery).

### 8. Target

Nearest opposing external liquidity clearing `target_min_rr`, or the far side of
the raided range (`target_mode: range_end`). Capped at the **five-day ADR**
projected from the day's range so far, at `adr_target_pct` (default 0.90) —
"I don't mind getting out and leaving the last 25, 30 pips on the table ... every
time I've tried to be that consistent or that accurate, it's always hurt me".

The cap is only applied when it lands on the correct side of entry; a bound
already passed means the day has delivered its range, and the R:R floor is left
to throw the setup out.

### 9. Stand-aside gates

| gate | rule | source |
|---|---|---|
| `skip_if_prior_session_delivered` | no NY Judas if London already delivered ≥80% of ADR5 | ICT day-trading model — "if London printed ~80% of the five-day ADR, New York is chop or a reversal" |
| `veto_both_sides_swept` | both sides of the prior range already run ⇒ consolidation day, stand down | Mobius: when London sweeps both Asia high **and** low, NY oscillates between them |
| `max_range_used_pct` | ≥85% of ADR5 already used before the anchor ⇒ no runway for the delivery leg | ICT ADR framing |
| `skip_weekdays` | optional; FOMC/NFP are stand-aside days in the source, but this model has no economic calendar | ICT day-trading model |

## Common Mistakes

Each of these is a filter in the code, not just advice:

- **Trading the Judas direction itself.** The Judas *is* the trap. Entry is on
  the reversal. The model only ever takes the bias direction.
- **No bias filter.** Without one the pattern is chop; `bias_mode` cannot be
  switched off, only chosen.
- **Insisting on the reversal.** Once no reversal has come by ~45 minutes into
  the killzone, the hypothesis is dead. Not a wider stop, not a later entry.
- **Confusing manipulation with delivery.** A breach that never closes back
  inside the level was run-and-continue; the liquidity was fuel, not a trigger.
- **Applying the "opposes delivery" rule to London close.** That variant extends
  the day's direction and is faded — a different pattern, not implemented here.
- **Expecting a Judas every day, on every pair.** One per session across a basket
  of four or five pairs.
- **Demanding size from a session designed to be narrow.** Asia's protraction is
  small by construction; wanting it to be large bets against the premise the rest
  of the model depends on.
- **Reading the reversal on too fast a chart.** The five-minute chart is the
  stated chart for the scalping module and for the CISD confirmation. On a
  two-minute chart the Judas extreme sits one or two pips from the entry array,
  which is a coin flip wearing a checklist.

## Sources

- `ICT-2016-PROTRACTION` — the parent concept; three protraction windows per 24h
  (20:00 / 00:00 / 07:00 NY), counter-directional by construction, small by
  construction, "market protraction or a Judas swing".
- `ICT-2017-MARKET-REVERSALS` (28:07) — the four-session enumeration.
- `ICT-2017-BREAD-BUTTER-BUY` / `-SELL` — the 08:20 CME anchor, the 5-day ADR
  objective, the 10:00 partial rule, one setup per session across a basket, the
  5-minute chart, and the London-close exception.
- `ICT-2017-INTRADAY-PROFILES` — delayed protraction, and the accepted no-entry day.
- `ICT-2017-DAYTRADE-ESSENTIALS` / `-HTF` / `ICT-2017-INTRADAY-TOP-DOWN` — the
  ADR5 frame, the week-open filter, the 80%-of-ADR stand-aside, banking inside
  the projection.
- OpenMobius knowledge base — `judas_swing` (key times, close-back confirmation,
  shallow vs deep counter-move), `new_york_manipulation_profile` (London
  consolidates ⇒ NY manipulates; CISD on the 5-minute chart),
  `breakout_vs_manipulation` (the candle-2/3 close-back test),
  `accumulation_manipulation_distribution` (both-sides-swept ⇒ consolidation).
- `strategies/notes/ict-forex-notes-mmari-2020.md` §17 — session-stop rotation.
