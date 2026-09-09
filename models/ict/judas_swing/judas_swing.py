"""
ICT Judas Swing Model
=====================

``session anchor -> Judas leg against bias -> sweep of the prior-session pool
-> reversal (CISD/MSS + displacement) -> retrace into the array -> deliver``

The Judas swing is the *manipulation* phase of Power-of-Three at session-open
scale: the opening move that runs the wrong way, takes the stops resting at the
prior session's range bound, and is then reversed into the day's real delivery.
This model trades the **reversal**, never the Judas itself.

Checklist (bullish â€” mirrored for bearish):
  1. Bias        â€” daily/4H order flow says UP; the Judas must therefore go DOWN.
  2. Anchor      â€” the counter-move starts inside a session's Judas window
                   (02:00 London / 08:20 CME / 20:00 Asia, NY clock).
  3. Sweep       â€” it raids a *prior-session* pool (Asia range for London,
                   London range for NY) and is rejected: long wick, close back
                   inside the level.
  4. Depth       â€” the manipulation is *small* relative to the coming delivery.
                   A counter-move that eats half the expected day is not a
                   Judas, it is the delivery.
  5. Reversal    â€” CISD or MSS with displacement, inside the same killzone and
                   within `max_minutes_to_reversal` of the sweep. No reversal in
                   that window = Judas failure = stand down.
  6. Entry/stop  â€” limit into the FVG/OB left by the displacement; stop beyond
                   the Judas extreme plus a structural buffer.
  7. Target      â€” opposing external liquidity, ADR bound as the ceiling.

Stand-aside gates (all from the source corpus, all opt-in):
  - London already delivered most of the 5-day ADR  -> no NY Judas.
  - London swept BOTH sides of Asia                 -> consolidation day, skip.
  - The Judas is deeper than `deep_judas_pct` * ADR5 -> skip, or cut the target
    back to the session opening price.

See judas-swing.md
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from concepts import (FVG, find_fvgs, find_order_blocks, find_breaker_blocks,
                      detect_mss, detect_cisd, first_breach,
                      find_liquidity_target, session_high_low, asian_range,
                      premium_discount)
from infrastructure.registry import concept
from infrastructure.time_utils import localize_like, ny_index
from models.intermediate.daily_bias import daily_bias_components


# â”€â”€â”€ Session profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# Every session has a protractionary stage (ICT-2016-PROTRACTION 11:38), but the
# four are not equally tradeable. London and NY are the two ICT tells students to
# focus on; Asia and London-close are "the two smallest tiny little windows of
# opportunity" that "[don't] pay out enough ... to take on the risk"
# (ICT-2017-BREAD-BUTTER-BUY 28:21). They are defined here so the model can be
# run on them, but `sessions` defaults to London + NY only.
#
#   anchor          NY time the Judas is measured from.
#   judas_minutes   How long after the anchor the counter-move may start.
#   kz_end          End of the killzone; the reversal must land inside it.
#   exit_ny         Position collapse time (ICT-2017-INTRADAY-TOP-DOWN 08:05:
#                   intraday positions are collapsed from 10:00-11:00 NY).
#   pools           Which prior-session pools the Judas is allowed to raid.
SESSION_PROFILES = {
    # London. Anchored on the **midnight NY open**, not 02:00. ICT-2016-PROTRACTION
    # (11:38) puts the London protraction window "after midnight New York time",
    # and ICT-2017-INTRADAY-PROFILES adds the *delayed* case — "draw a vertical
    # line at 2 a.m. ... between 2 o'clock and 3 a.m., that Judas swing should
    # occur" [21:19]. A 00:00 anchor with a 3.5h window spans both readings; a
    # 02:00 anchor sees only the delayed one and treats the on-time protraction
    # as a pool that was already mitigated.
    'london': {
        'anchor': '00:00', 'judas_minutes': 210, 'kz_end': '05:00',
        'exit_ny': '12:00',
        'pools': ['asia', 'pdh_pdl'],
    },
    # New York. ICT-2016-PROTRACTION times this window at 07:00 NY; the 2017
    # mentorship killzone is 07:00-10:00 and the bread-and-butter lecture
    # anchors on the 08:20 CME open. 07:00 + 150 min covers all three.
    'ny': {
        'anchor': '07:00', 'judas_minutes': 150, 'kz_end': '11:00',
        'exit_ny': '16:00',
        'pools': ['london', 'asia', 'pdh_pdl'],
    },
    # Asia open / 0 GMT. Sweep target is "a previous late session swing", not the
    # Asian range - that range does not exist yet at 20:00 NY. ICT does not trade
    # this one ("doesn't pay out enough ... to take on the risk"); it is here for
    # completeness of the four-session model.
    'asia': {
        'anchor': '20:00', 'judas_minutes': 60, 'kz_end': '23:59',
        'exit_ny': '23:59',
        'pools': ['late_session_swing', 'pdh_pdl'],
    },
}

_POOL_LABEL = {
    'asia': ('Asia-high', 'Asia-low'),
    'london': ('London-high', 'London-low'),
    'pdh_pdl': ('PDH', 'PDL'),
    'midnight_open': ('Midnight-open', 'Midnight-open'),
    'late_session_swing': ('LateSwing-high', 'LateSwing-low'),
}


@dataclass
class JudasSnapshot:
    """One session's worth of bars, sliced ahead of the model call.

    Attributes:
        df_entry:     Entry-timeframe bars (1m-5m) covering the days around the
                      session; sweeps, reversals and arrays are all read here.
        df_15m:       15-minute bars, for pool context.
        df_daily:     Daily bars ending *before* the session â€” bias and ADR5.
        session_date: Calendar date being evaluated (tz-naive).
        df_4h:        4-hour bars for the bias confirmation leg (optional).
    """
    df_entry: pd.DataFrame
    df_15m: pd.DataFrame
    df_daily: pd.DataFrame
    session_date: pd.Timestamp
    df_4h: Optional[pd.DataFrame] = None


@concept("judas-swing", depends_on=[
    "daily-bias", "power-of-three", "liquidity-sweep-stop-hunt",
    "market-structure-shift", "change-in-state-of-delivery",
    "fair-value-gap", "killzones", "sessions-and-ranges",
    "true-day-midnight-open", "targets-and-exits",
])
class JudasSwingModel:
    """Session-open manipulation model: fade the Judas, ride the delivery.

    Config keys are documented inline in DEFAULT_CONFIG. Everything that could
    be a subjective call is a named parameter with the source rule beside it, so
    a config diff reads as a change of doctrine rather than a change of taste.
    """

    DEFAULT_CONFIG = {
        # â”€â”€ Which sessions to evaluate, in order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        'sessions': ['london', 'ny'],
        # Per-session overrides, e.g. {'ny': {'judas_minutes': 60}}
        'session_overrides': {},

        # â”€â”€ Bias â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The reversal direction MUST agree with HTF bias. "A Judas swing without
        # HTF bias confirmation is just chop" (judas-swing.md, Common Mistakes).
        # Which read supplies that bias is a doctrine choice, so it is named:
        #   'strict'     - the project's daily-bias gate: daily order flow AND
        #                  draw-on-liquidity must agree, 4H may veto. Highest
        #                  conviction; abstains on roughly two days in three.
        #   'order_flow' - daily order flow alone over `daily_lookback` bars.
        #   'draw'       - the nearest unswept daily pool (draw on liquidity).
        #   'week_open'  - the ICT day-trading model's own filter: this week's
        #                  opening price projected forward. Above it = buy days,
        #                  below it = sell days (ICT-2017-DAYTRADE-ESSENTIALS
        #                  23:09-29:07).
        'bias_mode': 'strict',
        'daily_lookback': 5,
        'dealing_range_days': 20,
        'swing_lookback': 1,
        'four_h_gate': 'soft',          # 'soft' = neutral 4H abstains, 'hard' = vetoes
        'require_pd_zone': False,       # gate on premium/discount agreeing with bias

        # â”€â”€ Judas leg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # A genuine protraction spikes through the pool and is REJECTED; a move
        # that breaks the level and keeps going was real delivery using the
        # liquidity as fuel. Telling those apart is the single most important
        # filter in the model, and the corpus gives the test explicitly:
        # "confirm whether candle 2 or candle 3 closes back inside the range"
        # (breakout_vs_manipulation, identification rule 3). Demanding it of the
        # breach bar alone - as a naive close-inside check does - throws away
        # most genuine raids, because the rejection usually completes on the
        # next bar or two.
        'close_back_within_bars': 3,    # 0 disables the check
        # Optional extra: rejection wick as a share of the breach bar's range.
        # Off by default - the close-back test above already excludes
        # run-and-continue, and stacking both mostly costs sample.
        'min_sweep_wick_pct': 0.0,
        # Minimum breach depth, so a one-tick graze of the level is not a raid.
        # Expressed as a multiple of the pre-session ATR.
        'min_sweep_depth_atr': 0.05,
        # Cap on how far past the level the raid may run before it stops being a
        # stop hunt and becomes a breakout. Also as a multiple of ATR; 0 = off.
        'max_sweep_depth_atr': 0.0,

        # â”€â”€ Judas depth (the "small impulse swing" criterion) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # market-protraction.md: the protraction is "explicitly a *small* swing
        # relative to the delivery". Mobius adds the operational form: a deep
        # counter-move that consumes ~50% of the period's range caps the
        # expansion, so the target must be cut back to the opening price.
        # Measured as (anchor price -> Judas extreme) / ADR5.
        'deep_judas_pct': 0.40,
        'deep_judas_action': 'skip',    # 'skip' | 'open_target' | 'allow'

        # â”€â”€ Reversal confirmation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # "by 30-45 minutes into the killzone, if price is still extending in the
        # initial direction ... treat the Judas hypothesis as failed"
        # (judas-swing-failure.md). This is the model's Judas-failure guard.
        'max_minutes_to_reversal': 45,
        'confirmation': 'cisd_or_mss',  # 'cisd' | 'mss' | 'cisd_or_mss' | 'both'
        'require_reversal_in_kz': True,
        'require_displacement_quality': True,
        'displacement_body_mult': 1.3,
        'displacement_body_pct': 0.60,
        'displacement_opp_wick_pct': 0.30,

        # â”€â”€ Entry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # How the position is taken once the reversal is confirmed:
        #   'limit'  - wait for the retrace into the FVG/OB the displacement
        #              left. ICT's standard play, and the one the Mobius cards
        #              insist on ("wait for the deeper discount/premium rather
        #              than entering at the order block or the daily open").
        #              Better price, but a fill that arrives an hour later is a
        #              different trade from the one that was signalled.
        #   'market' - take the close of the confirmation candle (filled at the
        #              next bar's open, so there is no hindsight in the price).
        #              Worse entry, no fill risk, and the stop sits a healthy
        #              distance away because it is still the Judas extreme.
        'entry_style': 'limit',
        'entry_arrays': ['fvg', 'ob'],  # tried in order
        'fvg_entry_depth': 'ce',        # 'edge' | 'ce' (consequent encroachment)
        'array_lookback_bars': 20,
        'min_fvg_atr_mult': 0.25,       # kill noise-sized gaps
        # Minutes after the reversal within which the limit must fill. A Judas
        # entry that has not filled by the end of its killzone is a missed trade,
        # not a late one.
        'entry_fill_minutes': 120,
        'limit_fill_to_killzone': True,

        # â”€â”€ Stop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Structural: beyond the Judas extreme. ICT teaches a 2-5 pip (2-5 point)
        # buffer; a stop pinned exactly at the invalidation is a named failure mode.
        'stop_buffer_pts': 0.0,
        'max_stop_pts': 20.0,
        'min_stop_pts': 0.0,
        # Instrument-agnostic noise floor: reject a setup whose stop is closer
        # than this multiple of the session ATR. A structural stop is only
        # structural if it sits outside the bar-to-bar noise; on a 2-minute
        # chart the Judas extreme is often 1-2 pips from the entry array, which
        # is a coin flip wearing a checklist.
        'min_stop_atr_mult': 0.0,
        # ADR-relative bounds. These are the ones that travel: a stop is a
        # share of what the instrument moves in a day, so one config runs on
        # EURUSD and on the Nasdaq CFD without a pip table. The absolute
        # *_pts bounds above stay available for per-instrument overrides.
        'min_stop_adr_pct': 0.0,
        'max_stop_adr_pct': 0.0,

        # â”€â”€ Target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        'target_mode': 'liquidity',     # 'liquidity' | 'range_end' | 'adr'
        'target_min_rr': 2.5,
        # Cap the target at the 5-day ADR bound projected from the day's open.
        # "I don't mind getting out and leaving the last 25, 30 pips on the table"
        # (ICT-2017-INTRADAY-TOP-DOWN 20:24) â€” holding for the projected extreme
        # is reported as costly by ICT himself.
        'adr_days': 5,
        'cap_target_at_adr': True,
        'adr_target_pct': 0.90,         # bank this far into the ADR projection

        # â”€â”€ Stand-aside gates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # "Skip New York entirely if London already delivered ~80% or more of the
        # five-day ADR" (ict-day-trading-model.md 13:55).
        'skip_if_prior_session_delivered': 0.80,
        # Mobius: when London sweeps BOTH Asia high and low, NY consolidates
        # between them rather than expanding. Low-probability day, stand down.
        'veto_both_sides_swept': True,
        # Do not open a new Judas trade once the day has already run this much of
        # its ADR â€” there is no runway left for the delivery leg.
        'max_range_used_pct': 0.85,
        'skip_weekdays': [],            # e.g. [4] to skip Fridays (0=Mon)
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.checks: list = []

    # â”€â”€ logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _log(self, msg: str):
        self.checks.append(msg)

    # â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _ny_minutes(ts: pd.Timestamp) -> int:
        ny = ts.tz_convert('America/New_York') if ts.tzinfo else ts
        return ny.hour * 60 + ny.minute

    @staticmethod
    def _hhmm(s: str) -> int:
        h, m = map(int, str(s).split(':'))
        return h * 60 + m

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        """ATR over the tail of df; 0.0 when there is not enough data."""
        if df is None or len(df) < 2:
            return 0.0
        w = df.tail(period + 1)
        pc = w['close'].shift(1)
        tr = pd.concat([w['high'] - w['low'],
                        (w['high'] - pc).abs(),
                        (w['low'] - pc).abs()], axis=1).max(axis=1)
        return float(tr.dropna().mean() or 0.0)

    def _adr(self, df_daily: pd.DataFrame) -> float:
        """Five-day average daily range â€” the model's unit of 'a day's worth'."""
        n = self.config['adr_days']
        if df_daily is None or len(df_daily) < 2:
            return 0.0
        tail = df_daily.tail(n)
        return float((tail['high'] - tail['low']).mean() or 0.0)

    def _profile(self, session: str) -> dict:
        prof = dict(SESSION_PROFILES[session])
        prof.update(self.config['session_overrides'].get(session, {}))
        return prof

    def _ny_time_on(self, snapshot: JudasSnapshot, hhmm: str) -> Optional[pd.Timestamp]:
        """`hhmm` NY on the session date, in the entry frame's own timezone."""
        try:
            ts = pd.Timestamp(f"{snapshot.session_date.date()} {hhmm}",
                              tz='America/New_York')
        except Exception:
            return None
        idx = snapshot.df_entry.index
        return ts.tz_convert(idx.tz) if idx.tz is not None else ts.tz_localize(None)

    def _day_slice(self, df: pd.DataFrame, session_date: pd.Timestamp,
                   start_min: int, end_min: int) -> pd.DataFrame:
        """Bars of `session_date` between two NY-clock minute offsets."""
        ny = ny_index(df)
        m = ny.hour * 60 + ny.minute
        mask = (ny.date == session_date.date()) & (m >= start_min) & (m < end_min)
        return df[mask]

    # â”€â”€ Step 1: bias â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _week_open_bias(self, snapshot: JudasSnapshot) -> str:
        """ICT's Sunday-open filter: which side of the week's open price we sit on.

        "Bearish weekly bias + price below the Sunday open -> sell in London,
        continuation in New York, every day" (ICT-2017-DAYTRADE-ESSENTIALS
        26:45). The weekly-bias half is supplied by the week's own candle so far;
        the filter half by the current price against the week's opening price.
        Both must agree, which is what stops it degenerating into a coin flip in
        a week that opens and immediately reverses.
        """
        daily = snapshot.df_daily
        if daily is None or len(daily) < 2:
            return 'neutral'
        date = snapshot.session_date
        week_start = pd.Timestamp(date.date()) - pd.Timedelta(days=date.weekday())
        idx = daily.index
        idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
        this_week = daily[idx_naive >= week_start]
        if len(this_week) == 0:
            return 'neutral'
        week_open = float(this_week['open'].iloc[0])
        last_close = float(daily['close'].iloc[-1])
        week_dir = 'bullish' if last_close > week_open else 'bearish'
        side = 'bullish' if last_close > week_open else 'bearish'
        return week_dir if week_dir == side else 'neutral'

    def _bias(self, snapshot: JudasSnapshot) -> str:
        mode = self.config['bias_mode']
        if mode == 'week_open':
            bias = self._week_open_bias(snapshot)
            self._log(f"Bias: {bias.upper()} (week-open filter)")
            return bias

        comp = daily_bias_components(
            snapshot.df_daily,
            lookback=self.config['daily_lookback'],
            dealing_range_days=self.config['dealing_range_days'],
            swing_lookback=self.config['swing_lookback'],
            df_4h=snapshot.df_4h,
            four_h_gate=self.config['four_h_gate'],
        )
        bias = {'strict': comp['bias'],
                'order_flow': comp['order_flow'],
                'draw': comp['draw']}.get(mode)
        if bias is None:
            raise ValueError(f"Unknown bias_mode: {mode!r} (expected 'strict', "
                             f"'order_flow', 'draw' or 'week_open')")
        self._log(f"Bias: {bias.upper()} [{mode}] (order_flow={comp['order_flow']}, "
                  f"draw={comp['draw']}, 4H={comp['four_h_structure']})")
        return bias

    # â”€â”€ Step 2: the pools the Judas is allowed to raid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_pools(self, snapshot: JudasSnapshot, session: str) -> dict:
        """Prior-session liquidity, keyed by side.

        The pool set rotates with the session: London raids Asia's stops, NY
        raids London's, Asia raids the prior late-session swing
        (judas-swing.md / asia-judas-swing.md).
        """
        prof = self._profile(session)
        df = snapshot.df_entry
        date = snapshot.session_date
        highs: list = []
        lows: list = []

        for kind in prof['pools']:
            hi_label, lo_label = _POOL_LABEL[kind]

            if kind == 'asia':
                rng = asian_range(df, date)
                if rng is None:
                    continue
                highs.append({'extreme_price': rng['high'], 'source': hi_label,
                              'formed_at': rng['end'], 'range_low': rng['low'],
                              'range_high': rng['high'], 'timestamps': [rng['end']]})
                lows.append({'extreme_price': rng['low'], 'source': lo_label,
                             'formed_at': rng['end'], 'range_low': rng['low'],
                             'range_high': rng['high'], 'timestamps': [rng['end']]})

            elif kind == 'london':
                rng = session_high_low(df, date, start_ny='02:00', end_ny=prof['anchor'])
                if rng is None:
                    continue
                highs.append({'extreme_price': rng['high'], 'source': hi_label,
                              'formed_at': rng['end'], 'range_low': rng['low'],
                              'range_high': rng['high'], 'timestamps': [rng['end']]})
                lows.append({'extreme_price': rng['low'], 'source': lo_label,
                             'formed_at': rng['end'], 'range_low': rng['low'],
                             'range_high': rng['high'], 'timestamps': [rng['end']]})

            elif kind == 'pdh_pdl':
                daily = snapshot.df_daily
                if daily is None or len(daily) == 0:
                    continue
                prior = daily.iloc[-1]
                formed = localize_like(pd.Timestamp(f"{date.date()} 00:00"), df.index)
                highs.append({'extreme_price': float(prior['high']), 'source': hi_label,
                              'formed_at': formed, 'timestamps': [daily.index[-1]]})
                lows.append({'extreme_price': float(prior['low']), 'source': lo_label,
                             'formed_at': formed, 'timestamps': [daily.index[-1]]})

            elif kind == 'midnight_open':
                # The true-day open is a reference PRICE, not a range: the Judas
                # is read as the excursion away from it before delivery.
                mo = self._day_slice(df, date, 0, 5)
                if len(mo) == 0:
                    continue
                px = float(mo['open'].iloc[0])
                highs.append({'extreme_price': px, 'source': hi_label,
                              'formed_at': mo.index[0], 'timestamps': [mo.index[0]]})
                lows.append({'extreme_price': px, 'source': lo_label,
                             'formed_at': mo.index[0], 'timestamps': [mo.index[0]]})

            elif kind == 'late_session_swing':
                # "it can go down just below a previous late session swing"
                # (ICT-2017-BREAD-BUTTER-BUY 21:06) â€” the NY PM window's extremes.
                rng = session_high_low(df, date, start_ny='13:00', end_ny=prof['anchor'])
                if rng is None:
                    continue
                highs.append({'extreme_price': rng['high'], 'source': hi_label,
                              'formed_at': rng['end'], 'timestamps': [rng['end']]})
                lows.append({'extreme_price': rng['low'], 'source': lo_label,
                             'formed_at': rng['end'], 'timestamps': [rng['end']]})

        self._log(f"Pools[{session}]: {len(highs)} buy-side / {len(lows)} sell-side")
        return {'highs': highs, 'lows': lows}

    # â”€â”€ Step 3: stand-aside gates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _prior_session_gate(self, snapshot: JudasSnapshot, session: str,
                            adr: float) -> bool:
        """True = keep going. Skip NY when London already delivered the day."""
        pct = self.config['skip_if_prior_session_delivered']
        if not pct or adr <= 0 or session != 'ny':
            return True
        prof = self._profile(session)
        london = session_high_low(snapshot.df_entry, snapshot.session_date,
                                  start_ny='02:00', end_ny=prof['anchor'])
        if london is None:
            return True
        delivered = (london['high'] - london['low']) / adr
        if delivered >= pct:
            self._log(f"Stand aside: London already delivered {delivered:.0%} of ADR5 "
                      f"(limit {pct:.0%}) â€” NY extends rather than reverses")
            return False
        return True

    def _both_sides_gate(self, snapshot: JudasSnapshot, session: str,
                         pools: dict, anchor_min: int) -> bool:
        """True = keep going. Both sides of the prior range run = consolidation."""
        if not self.config['veto_both_sides_swept'] or session == 'london':
            return True
        prior = self._day_slice(snapshot.df_entry, snapshot.session_date, 0, anchor_min)
        if len(prior) == 0:
            return True
        # Only range pools carry two sides worth testing.
        for hi in pools['highs']:
            if 'range_low' not in hi:
                continue
            took_high = float(prior['high'].max()) > hi['extreme_price']
            took_low = float(prior['low'].min()) < hi['range_low']
            if took_high and took_low:
                self._log(f"Stand aside: both sides of {hi['source'].split('-')[0]} "
                          f"already swept before the anchor â€” consolidation day")
                return False
        return True

    def _range_used_gate(self, snapshot: JudasSnapshot, anchor_min: int,
                         adr: float) -> bool:
        """True = keep going. No runway left once the ADR is mostly spent."""
        pct = self.config['max_range_used_pct']
        if not pct or adr <= 0:
            return True
        so_far = self._day_slice(snapshot.df_entry, snapshot.session_date, 0, anchor_min)
        if len(so_far) == 0:
            return True
        used = (float(so_far['high'].max()) - float(so_far['low'].min())) / adr
        if used >= pct:
            self._log(f"Stand aside: {used:.0%} of ADR5 already used before the anchor "
                      f"(limit {pct:.0%})")
            return False
        return True

    # â”€â”€ Step 4: the Judas leg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _sweep_quality_ok(self, df: pd.DataFrame, ts: pd.Timestamp, level: float,
                          side: str, atr: float) -> bool:
        """Separate a stop hunt from a breakout at the raided level.

        Three tests, ordered by how much sample they cost:
          1. depth - a one-tick graze is not a raid; an excursion far past the
                     level is not a raid either, it is delivery.
          2. wick  - optional rejection-wick share of the breach bar.
          3. close - price must close back on the origin side of the level
                     within `close_back_within_bars`. This is the corpus test
                     ("confirm whether candle 2 or candle 3 closes back inside
                     the range", breakout_vs_manipulation) and the one that does
                     the real work; demanding it of the breach bar alone throws
                     away most genuine raids, because the rejection normally
                     completes on the next bar or two.
        """
        pos = int(df.index.get_indexer([ts], method='nearest')[0])
        row = df.iloc[pos]
        o, h, l, c = (float(row['open']), float(row['high']),
                      float(row['low']), float(row['close']))
        rng = h - l
        if rng <= 0:
            return False

        if side == 'high':
            wick, depth = h - max(o, c), h - level
        else:
            wick, depth = min(o, c) - l, level - l

        lo_need = self.config['min_sweep_depth_atr'] * atr
        if lo_need > 0 and depth < lo_need:
            self._log(f"Judas rejected: breach depth {depth:.5f} < {lo_need:.5f} "
                      f"({self.config['min_sweep_depth_atr']}x ATR) - a graze, not a raid")
            return False
        hi_need = self.config['max_sweep_depth_atr'] * atr
        if hi_need > 0 and depth > hi_need:
            self._log(f"Judas rejected: breach depth {depth:.5f} > {hi_need:.5f} "
                      f"({self.config['max_sweep_depth_atr']}x ATR) - a breakout, not a raid")
            return False

        min_pct = self.config['min_sweep_wick_pct']
        if min_pct and wick / rng < min_pct:
            self._log(f"Judas rejected: rejection wick {wick / rng:.0%} of bar "
                      f"< {min_pct:.0%}")
            return False

        n = self.config['close_back_within_bars']
        if n:
            window = df.iloc[pos:pos + n]
            back = ((window['close'] < level) if side == 'high'
                    else (window['close'] > level))
            if not bool(back.any()):
                self._log(f"Judas rejected: no close back inside {level:.5f} within "
                          f"{n} bars - run-and-continue, not manipulation")
                return False
        return True

    def _find_judas(self, snapshot: JudasSnapshot, session: str, pools: list,
                    judas_dir: str, atr: float) -> Optional[dict]:
        """Earliest qualifying sweep inside the session's Judas window.

        `judas_dir` is the direction of the *fake* move: 'bearish' when the day's
        bias is bullish. It raids the sell-side (low) pools.
        """
        prof = self._profile(session)
        anchor = self._hhmm(prof['anchor'])
        window_end = anchor + prof['judas_minutes']
        window = self._day_slice(snapshot.df_entry, snapshot.session_date,
                                 anchor, window_end)
        if len(window) < 3:
            self._log(f"Rule of exclusion: no bars in the {session} Judas window")
            return None
        window_times = set(window.index)

        side = 'low' if judas_dir == 'bearish' else 'high'
        best = None
        for pool in pools:
            lvl = pool['extreme_price']
            ts, row = first_breach(snapshot.df_entry, lvl, side,
                                   after=pool.get('formed_at'))
            if ts is None or ts not in window_times:
                continue  # never raided, or first raided outside the Judas window
            if not self._sweep_quality_ok(snapshot.df_entry, ts, lvl, side, atr):
                continue
            if best is None or ts < best['sweep_time']:
                best = {'sweep_time': ts, 'pool': pool, 'level': lvl,
                        'sweep_extreme': float(row['low'] if side == 'low' else row['high'])}

        if best is None:
            self._log(f"Rule of exclusion: no qualifying Judas sweep in the "
                      f"{session} window ({prof['anchor']}+{prof['judas_minutes']}m)")
            return None

        anchor_bar = window.iloc[0]
        best['anchor_price'] = float(anchor_bar['open'])
        best['anchor_time'] = window.index[0]
        self._log(f"Judas: {judas_dir} leg raided {best['pool']['source']} "
                  f"@ {best['level']:.5f} at {best['sweep_time']} "
                  f"(extreme {best['sweep_extreme']:.5f})")
        return best

    def _depth_ok(self, judas: dict, adr: float) -> Optional[str]:
        """Classify the manipulation depth. Returns None to reject the setup.

        Returns 'shallow' or 'deep'; a deep Judas either kills the trade or caps
        the target at the session open, per `deep_judas_action`.
        """
        limit = self.config['deep_judas_pct']
        if not limit or adr <= 0:
            return 'shallow'
        depth = abs(judas['anchor_price'] - judas['sweep_extreme']) / adr
        if depth < limit:
            self._log(f"Judas depth {depth:.0%} of ADR5 â€” shallow, full expansion expected")
            return 'shallow'
        action = self.config['deep_judas_action']
        if action == 'skip':
            self._log(f"Rule of exclusion: Judas depth {depth:.0%} of ADR5 "
                      f">= {limit:.0%} â€” the counter-move IS the delivery")
            return None
        self._log(f"Judas depth {depth:.0%} of ADR5 â€” deep; expansion capped "
                  f"(action={action})")
        return 'deep'

    # â”€â”€ Step 5: reversal confirmation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _displacement_ok(self, df: pd.DataFrame, pos: int, direction: str) -> bool:
        """Body-dominance / opposing-wick test on the confirmation candle."""
        if not self.config['require_displacement_quality']:
            return True
        if pos < 0 or pos >= len(df):
            return True
        c = df.iloc[pos]
        o, h, l, cl = float(c['open']), float(c['high']), float(c['low']), float(c['close'])
        rng, body = h - l, abs(cl - o)
        if rng <= 0:
            return False
        prior = df.iloc[max(0, pos - 14):pos]
        avg_body = float((prior['close'] - prior['open']).abs().mean() or 0.0)
        if avg_body > 0 and body < self.config['displacement_body_mult'] * avg_body:
            self._log(f"Displacement rejected: body {body:.5f} < "
                      f"{self.config['displacement_body_mult']}x avg {avg_body:.5f}")
            return False
        if body / rng < self.config['displacement_body_pct']:
            self._log(f"Displacement rejected: body {body / rng:.0%} of range")
            return False
        opp = (h - max(o, cl)) if direction == 'bullish' else (min(o, cl) - l)
        if opp / rng > self.config['displacement_opp_wick_pct']:
            self._log(f"Displacement rejected: opposing wick {opp / rng:.0%} of range")
            return False
        return True

    def _confirm_reversal(self, snapshot: JudasSnapshot, session: str,
                          judas: dict, bias: str) -> Optional[dict]:
        """CISD and/or MSS in the bias direction, inside the failure deadline.

        This is where a Judas is separated from a Judas *failure*: no reversal
        inside `max_minutes_to_reversal` means the opening move was the real
        delivery, and the correct response is to stand down.
        """
        df = snapshot.df_entry
        prof = self._profile(session)
        sweep_ts = judas['sweep_time']
        deadline_min = self._ny_minutes(sweep_ts) + self.config['max_minutes_to_reversal']
        if self.config['require_reversal_in_kz']:
            deadline_min = min(deadline_min, self._hhmm(prof['kz_end']))

        bar_minutes = max(1, int(
            pd.Series(df.index).diff().dt.total_seconds().median() // 60 or 1))
        max_bars = max(3, int(self.config['max_minutes_to_reversal'] / bar_minutes) + 1)

        mode = self.config['confirmation']
        cisd = mss = None
        if mode in ('cisd', 'cisd_or_mss', 'both'):
            cisd = detect_cisd(df, direction=bias, sweep_time=sweep_ts,
                               max_bars_after_sweep=max_bars)
        if mode in ('mss', 'cisd_or_mss', 'both'):
            mss = detect_mss(df, direction=bias,
                             swing_lookback=self.config['swing_lookback'],
                             max_bars_after_sweep=max_bars, sweep_time=sweep_ts)

        if mode == 'both' and (cisd is None or mss is None):
            self._log("Rule of exclusion: needed CISD *and* MSS, only one fired")
            return None

        candidates = []
        if cisd is not None:
            candidates.append(('CISD', cisd['cisd_time'], cisd['reference_level']))
        if mss is not None:
            candidates.append(('MSS', mss['break_time'], mss['broken_level']))
        if not candidates:
            self._log(f"Rule of exclusion: JUDAS FAILURE â€” no reversal within "
                      f"{self.config['max_minutes_to_reversal']} min of the sweep")
            return None

        kind, ts, level = min(candidates, key=lambda c: c[1])
        if self._ny_minutes(ts) > deadline_min:
            self._log(f"Rule of exclusion: {kind} at {ts} past the reversal deadline "
                      f"({deadline_min // 60:02d}:{deadline_min % 60:02d} NY)")
            return None

        pos = int(df.index.get_indexer([ts], method='nearest')[0])
        if not self._displacement_ok(df, pos, bias):
            return None

        # The Judas extreme is the furthest point reached between the sweep and
        # the confirmation â€” not the sweep bar's own wick. That is the level the
        # setup is invalidated beyond.
        sweep_pos = int(df.index.get_indexer([sweep_ts], method='nearest')[0])
        leg = df.iloc[sweep_pos:pos + 1]
        extreme = float(leg['low'].min()) if bias == 'bullish' else float(leg['high'].max())

        self._log(f"Reversal: {kind} through {level:.5f} @ {ts} "
                  f"(Judas extreme {extreme:.5f})")
        return {'kind': kind, 'time': ts, 'level': level, 'pos': pos,
                'judas_extreme': extreme}

    # â”€â”€ Step 6: entry array â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_depth(self, f: FVG) -> FVG:
        depth = self.config['fvg_entry_depth']
        if depth == 'edge':
            return f
        if depth != 'ce':
            raise ValueError(f"Unknown fvg_entry_depth: {depth!r}")
        return dataclasses.replace(f, entry=(f.top + f.bottom) / 2.0)

    def _find_entry_array(self, snapshot: JudasSnapshot, reversal: dict,
                          bias: str) -> Optional[FVG]:
        """First array of the configured kinds at the reversal displacement."""
        df = snapshot.df_entry
        pos = reversal['pos']
        end = min(len(df), pos + 3)
        atr = self._atr(df.iloc[:max(1, pos)])

        for kind in self.config['entry_arrays']:
            if kind == 'fvg':
                search = df.iloc[max(0, pos - 2):end]
                if len(search) < 3:
                    continue
                found = find_fvgs(search, direction=bias)
                mult = self.config['min_fvg_atr_mult']
                if mult and atr > 0:
                    found = [g for g in found if abs(g.top - g.bottom) >= mult * atr]
                if found:
                    f = self._apply_depth(found[0])
                    self._log(f"Entry FVG {f.bottom:.5f}-{f.top:.5f} @ {f.timestamp}, "
                              f"limit {f.entry:.5f}")
                    return f
                continue

            wide = df.iloc[max(0, pos - self.config['array_lookback_bars']):end]
            if len(wide) < 5:
                continue
            if kind == 'ob':
                zones = find_order_blocks(wide, direction=bias, validated_only=False)
            elif kind == 'breaker':
                zones = find_breaker_blocks(wide, direction=bias)
            else:
                raise ValueError(f"Unknown entry array: {kind!r}")
            if not zones:
                continue
            z = zones[-1]
            f = self._apply_depth(FVG(timestamp=z.timestamp, direction=bias,
                                      top=float(z.zone_top), bottom=float(z.zone_bot),
                                      entry=float(z.proximal)))
            self._log(f"Entry {kind.upper()} {f.bottom:.5f}-{f.top:.5f} @ {f.timestamp}, "
                      f"limit {f.entry:.5f}")
            return f

        self._log("Rule of exclusion: no entry array at the reversal displacement")
        return None

    # â”€â”€ Step 7: target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _adr_bound(self, snapshot: JudasSnapshot, direction: str, adr: float,
                   as_of: pd.Timestamp) -> Optional[float]:
        """The 5-day ADR projected from the day's range *so far* - the ceiling.

        ICT uses the five-day ADR to bound how far a single day can reasonably
        deliver, and banks inside it rather than reaching for the extreme. Only
        bars up to `as_of` are used: reading the day's finished high/low would be
        hindsight, and would quietly make every target look achievable.
        """
        if adr <= 0:
            return None
        today = self._day_slice(snapshot.df_entry, snapshot.session_date, 0, 24 * 60)
        as_of = localize_like(as_of, today.index) if len(today) else as_of
        today = today[today.index <= as_of]
        if len(today) == 0:
            return None
        pct = self.config['adr_target_pct']
        if direction == 'long':
            return float(today['low'].min()) + adr * pct
        return float(today['high'].max()) - adr * pct

    def _target(self, snapshot: JudasSnapshot, judas: dict, direction: str,
                entry: float, stop: float, target_pools: list, adr: float,
                depth_class: str) -> float:
        risk = abs(stop - entry)

        # Deep Judas: expansion is capped, so aim for the session opening price
        # rather than a full-range objective (Mobius judas_swing.json).
        if depth_class == 'deep' and self.config['deep_judas_action'] == 'open_target':
            px = judas['anchor_price']
            if (direction == 'long' and px > entry) or (direction == 'short' and px < entry):
                self._log(f"Deep Judas: target cut back to the session open {px:.5f}")
                return float(px)

        target = None
        if self.config['target_mode'] == 'range_end':
            pool = judas['pool']
            far = pool.get('range_high') if direction == 'long' else pool.get('range_low')
            if far is not None:
                far = float(far)
                ok = far > entry if direction == 'long' else far < entry
                if ok and risk > 0 and abs(far - entry) / risk >= self.config['target_min_rr']:
                    target = far
                    self._log(f"Target: far side of the raided range {far:.5f}")

        if target is None:
            live = self._live_pools(snapshot.df_entry, target_pools, direction,
                                    judas['sweep_time'])
            target = find_liquidity_target(direction, entry=entry, stop=stop,
                                           target_pools=live,
                                           min_rr=self.config['target_min_rr'])

        if self.config['cap_target_at_adr']:
            bound = self._adr_bound(snapshot, direction, adr, judas['sweep_time'])
            # A cap that lands on the wrong side of entry is not a cap, it is a
            # sign the day has already delivered its range - leave the target
            # alone and let the R:R floor throw the setup out if it must.
            on_side = bound is not None and (
                bound > entry if direction == 'long' else bound < entry)
            if on_side:
                capped = min(target, bound) if direction == 'long' else max(target, bound)
                if capped != target:
                    self._log(f"Target capped at the ADR5 projection: "
                              f"{target:.5f} -> {capped:.5f}")
                target = capped
            elif bound is not None:
                self._log(f"ADR5 projection {bound:.5f} already passed - cap not applied")
        return float(target)

    def _live_pools(self, df: pd.DataFrame, pools: list, direction: str,
                    as_of: pd.Timestamp) -> list:
        """Drop target pools whose liquidity has already been taken."""
        side = 'high' if direction == 'long' else 'low'
        as_of = localize_like(as_of, df.index)
        live = []
        for pool in pools:
            ts, _ = first_breach(df, pool['extreme_price'], side,
                                 after=pool.get('formed_at'))
            if ts is None or ts > as_of:
                live.append(pool)
        return live

    # â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate_signal(self, snapshot: JudasSnapshot,
                        session: str = 'london',
                        force_direction: Optional[str] = None) -> dict:
        """Evaluate one session of one day for a Judas-swing reversal setup.

        Args:
            snapshot:        Bars and date for the session.
            session:         Key into SESSION_PROFILES ('london', 'ny', 'asia').
            force_direction: 'bullish'/'bearish' to bypass the bias gate and
                             evaluate that side only (used to test both sides).

        Returns a dict; `actionable` is False unless every checklist item passed.
        """
        self.checks = []
        base = {'actionable': False, 'session': session, 'direction': None,
                'entry': None, 'entry_time': None, 'stop': None, 'target': None,
                'checks': self.checks}

        if session not in SESSION_PROFILES:
            raise ValueError(f"Unknown session {session!r}; "
                             f"expected one of {sorted(SESSION_PROFILES)}")
        prof = self._profile(session)

        if snapshot.session_date.weekday() in self.config['skip_weekdays']:
            self._log(f"Stand aside: {snapshot.session_date:%A} is on the skip list")
            return base

        # 1. Bias â€” the reversal direction. The Judas runs the other way.
        if force_direction is not None:
            bias = force_direction
            self._log(f"Bias: FORCED {bias.upper()}")
        else:
            bias = self._bias(snapshot)
            if bias == 'neutral':
                self._log("Rule of exclusion: no HTF bias â€” a Judas without bias is chop")
                return base

        direction = 'long' if bias == 'bullish' else 'short'
        judas_dir = 'bearish' if bias == 'bullish' else 'bullish'

        if self.config['require_pd_zone']:
            pd_signal = premium_discount(snapshot.df_daily,
                                         self.config['dealing_range_days'])
            if pd_signal != bias:
                self._log(f"Rule of exclusion: P/D zone {pd_signal} misaligned with bias")
                return base

        adr = self._adr(snapshot.df_daily)
        anchor_min = self._hhmm(prof['anchor'])

        # 2. Stand-aside gates, cheapest first.
        if not self._range_used_gate(snapshot, anchor_min, adr):
            return base
        if not self._prior_session_gate(snapshot, session, adr):
            return base

        # 3. Pools. The Judas raids the side OPPOSITE the day's delivery.
        pools = self._build_pools(snapshot, session)
        judas_pools = pools['lows'] if judas_dir == 'bearish' else pools['highs']
        target_pools = pools['highs'] if direction == 'long' else pools['lows']
        if not judas_pools:
            self._log("Rule of exclusion: no prior-session pool for the Judas to raid")
            return base
        if not self._both_sides_gate(snapshot, session, pools, anchor_min):
            return base

        # 4. The Judas leg itself.
        # Volatility yardstick for the depth and stop-floor tests. Measured on
        # the bars leading into the anchor; the London anchor is midnight, which
        # has no same-day predecessor, so it falls back to the Asian session.
        pre = self._day_slice(snapshot.df_entry, snapshot.session_date, 0, anchor_min)
        if len(pre) < 5:
            prior = snapshot.session_date - pd.Timedelta(days=1)
            pre = self._day_slice(snapshot.df_entry, prior, 20 * 60, 24 * 60)
        session_atr = self._atr(pre)
        atr = session_atr
        judas = self._find_judas(snapshot, session, judas_pools, judas_dir, atr)
        if judas is None:
            return base

        depth_class = self._depth_ok(judas, adr)
        if depth_class is None:
            return base

        # 5. Reversal â€” or Judas failure.
        reversal = self._confirm_reversal(snapshot, session, judas, bias)
        if reversal is None:
            return base

        # 6. Entry. Either the retrace into the array the displacement left, or
        #    the confirmation candle itself.
        style = self.config['entry_style']
        if style == 'market':
            bar = snapshot.df_entry.iloc[reversal['pos']]
            array = FVG(timestamp=reversal['time'], direction=bias,
                        top=float(bar['high']), bottom=float(bar['low']),
                        entry=float(bar['close']))
            self._log(f"Entry at the {reversal['kind']} close {array.entry:.5f} "
                      f"(market, filled on the next bar's open)")
        elif style == 'limit':
            array = self._find_entry_array(snapshot, reversal, bias)
            if array is None:
                return base
        else:
            raise ValueError(f"Unknown entry_style: {style!r} "
                             f"(expected 'limit' or 'market')")

        # 7. Stop: beyond the Judas extreme, plus the structural buffer.
        buf = self.config['stop_buffer_pts']
        stop = (reversal['judas_extreme'] - buf if direction == 'long'
                else reversal['judas_extreme'] + buf)
        if direction == 'long' and stop >= array.entry:
            self._log(f"Rule of exclusion: stop {stop:.5f} not below entry {array.entry:.5f}")
            return base
        if direction == 'short' and stop <= array.entry:
            self._log(f"Rule of exclusion: stop {stop:.5f} not above entry {array.entry:.5f}")
            return base

        risk = abs(stop - array.entry)
        if risk > self.config['max_stop_pts']:
            self._log(f"Rule of exclusion: stop {risk:.5f} wider than "
                      f"{self.config['max_stop_pts']:.5f} â€” the Judas was too deep to fade")
            return base
        if risk < self.config['min_stop_pts']:
            self._log(f"Rule of exclusion: stop {risk:.5f} tighter than "
                      f"{self.config['min_stop_pts']:.5f} â€” inside the noise band")
            return base
        atr_floor = self.config['min_stop_atr_mult'] * session_atr
        if atr_floor > 0 and risk < atr_floor:
            self._log(f"Rule of exclusion: stop {risk:.5f} < {atr_floor:.5f} "
                      f"({self.config['min_stop_atr_mult']}x session ATR) - the "
                      f"structural level is inside the noise band")
            return base
        if adr > 0:
            lo_adr = self.config['min_stop_adr_pct'] * adr
            hi_adr = self.config['max_stop_adr_pct'] * adr
            if lo_adr > 0 and risk < lo_adr:
                self._log(f"Rule of exclusion: stop {risk:.5f} < "
                          f"{self.config['min_stop_adr_pct']:.0%} of ADR5 "
                          f"({lo_adr:.5f}) - too tight to be structural")
                return base
            if hi_adr > 0 and risk > hi_adr:
                self._log(f"Rule of exclusion: stop {risk:.5f} > "
                          f"{self.config['max_stop_adr_pct']:.0%} of ADR5 "
                          f"({hi_adr:.5f}) - the Judas was too deep to fade")
                return base

        # 8. Target.
        target = self._target(snapshot, judas, direction, array.entry, stop,
                              target_pools, adr, depth_class)
        rr = abs(target - array.entry) / risk if risk else 0.0
        if rr < self.config['target_min_rr']:
            self._log(f"Rule of exclusion: R:R {rr:.2f} below the "
                      f"{self.config['target_min_rr']} floor")
            return base

        # The limit lives until `entry_fill_minutes` OR the end of the killzone,
        # whichever comes first: the institutional participation the setup rides
        # ends with the window, so a fill after it is a different trade.
        deadline = array.timestamp + pd.Timedelta(minutes=self.config['entry_fill_minutes'])
        if self.config['limit_fill_to_killzone']:
            kz_close = self._ny_time_on(snapshot, prof['kz_end'])
            if kz_close is not None:
                deadline = min(deadline, kz_close)
        self._log(f"Signal: {direction.upper()} entry={array.entry:.5f} "
                  f"stop={stop:.5f} target={target:.5f} R:R={rr:.2f}")

        return {
            'actionable': True,
            'session': session,
            'direction': direction,
            'entry': float(array.entry),
            'entry_time': array.timestamp,
            'entry_deadline': deadline,
            'stop': float(stop),
            'target': float(target),
            'rr': float(rr),
            'fvg_top': float(array.top),
            'fvg_bottom': float(array.bottom),
            'judas_extreme': float(reversal['judas_extreme']),
            'judas_depth_class': depth_class,
            'sweep_level': float(judas['level']),
            'sweep_time': judas['sweep_time'],
            'swept_pool': judas['pool']['source'],
            'swept_cluster': judas['pool'],
            'anchor_price': float(judas['anchor_price']),
            'confirmation': reversal['kind'],
            'entry_style': style,
            'bms_time': reversal['time'],
            'bms_swing_level': float(reversal['level']),
            'exit_ny': prof['exit_ny'],
            'adr5': float(adr),
            'checks': self.checks,
        }
