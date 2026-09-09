"""
ICT SMT Divergence Model
========================

``bias picks the side -> a relative extreme inside the comparison window is
raided -> the correlated instruments refuse to confirm -> reversal -> deliver``

SMT is a confirmation layer, not a trigger. Measured over 806,715 aligned 5m
bars (2015-2026), an unfiltered divergence performs *worse* than the no-SMT
baseline in half the window/side cells tested. What separates it is the stack
around it: HTF order flow choosing which side to compare, a live pool to raid,
a rejection, and a reversal. This model is that stack, built so the SMT gate can
be switched off (`require_smt: False`) and the identical machinery re-run — the
difference between the two is the only honest measure of what SMT contributes.

Checklist (bullish - mirrored for bearish):
  1. Bias      - daily order flow is UP, so we compare LOWS. Not optional; a
                 divergence read on the wrong side is a different claim about
                 the market, and measures worse than no signal at all.
  2. Window    - inside a stated comparison window, NY clock, DST-aware:
                 AM 05:00-09:30, PM 12:00-15:00.
  3. Pool      - the earlier relative low being compared is still resting: it
                 was not already breached between forming and the raid.
  4. Raid      - the leader trades through that low and is rejected (closes back
                 above it inside `close_back_within_bars`).
  5. SMT       - at the SAME timestamps, at least one correlated instrument
                 refused to make its own new low, with `clarity` inside a band.
                 A band, not a floor: the largest divergences are decorrelation
                 events, not engineered sweeps, and they carry the worst
                 adverse excursion of any group measured.
  6. Reversal  - CISD or MSS within `max_minutes_to_reversal` of the raid.
  7. Entry     - limit into the FVG/OB left by the displacement, or market on
                 the confirmation close.
  8. Stop      - at the level whose breach would negate the SMT itself, plus a
                 buffer, clamped to a fraction of ADR. The floor is mandatory:
                 the Judas post-mortem showed 5-6 pip stops surrendered 43% of
                 the edge to spread.
  9. Target    - opposing liquidity, fixed R, or a time-of-day exit.

See smt-model.md
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from infrastructure.registry import concept
from infrastructure.time_utils import (localize_like, window_positions,
                                       before_positions)
from concepts import (
    find_smt_events, find_smt_breach_events, negated_at, SMTEvent,
    find_fvgs, find_order_blocks, detect_cisd, detect_mss,
    find_relative_equal_levels, find_liquidity_target,
)
from models.intermediate.daily_bias.daily_bias import daily_bias_components


WINDOWS = {
    # New York local time. Both are stated verbatim in the Month 10 index
    # lectures: AM at [07:57] "between 5 AM and 9:30 AM New York time, relative
    # highs and lows should be compared"; PM at [07:34] "the highs and the lows
    # between noon and 3 PM New York time".
    'am': ('05:00', '09:30'),
    'pm': ('12:00', '15:00'),
    # Not from the index lectures - the London killzone, included so the model
    # can be evaluated at a trade frequency the AM/PM pair alone cannot reach.
    # Kept explicitly separate from the two sourced windows so its results are
    # never quoted as doctrine.
    'london': ('02:00', '05:00'),
}

# Forced-flat time per window, NY clock. AM setups are held "minimum 10:30 to
# 11 o'clock" (M10-INDEX [02:41]); PM setups are held toward the 15:00 bond
# close and the last hour (M10-PM [12:12]).
WINDOW_EXIT = {'am': '11:00', 'pm': '16:00', 'london': '08:00'}


@dataclass
class SMTSnapshot:
    """Everything one (session, window) evaluation may look at.

    `frames` must share a single bar index across instruments - SMT compares the
    same time-of-day candle, and that is only sound on a common index. Use
    ``DataLoader.read_smt_group``.
    """
    frames: Dict[str, pd.DataFrame]          # entry-timeframe OHLCV, aligned
    daily: Dict[str, pd.DataFrame]           # daily bars per instrument
    session_date: pd.Timestamp
    inverse: Tuple[str, ...] = ()
    four_h: Optional[Dict[str, pd.DataFrame]] = None


@concept("smt-model", depends_on=[
    "smt-divergence", "daily-bias", "liquidity-sweep-stop-hunt",
    "market-structure-shift", "change-in-state-of-delivery",
    "fair-value-gap", "order-block", "targets-and-exits",
    "relative-equal-highs-lows", "swing-points",
])
class SMTModel:
    """Full SMT strategy. Every price bound is a fraction of the instrument's
    own ADR, so one config runs unchanged on index CFDs and on FX."""

    DEFAULT_CONFIG = {
        # ── group & bias ────────────────────────────────────────────────────
        'bias_instrument': None,        # None -> first member of the group
        'use_daily_bias': True,         # False evaluates both sides (research)
        # 'strict'    - the daily-bias gate (order flow AND draw AND 4H). This
        #               removed ~67% of sessions in the first index run, matching
        #               the Judas model's ~70%, so it is the dominant filter in
        #               the whole stack and deserves to be an explicit axis.
        # 'orderflow' - the daily order-flow component alone.
        # 'off' takes no directional veto at all and evaluates both sides. It
        # exists because the strict gate rejected 68.6% of sessions, which is
        # most of the reason the model produced ~1 trade a month. Bias is then
        # recorded on every signal so it can be used as a ranking feature
        # instead of a veto.
        'bias_mode': 'strict',          # 'strict' | 'orderflow' | 'off'
        'bias_lookback': 20,
        'bias_dealing_range_days': 20,
        'bias_swing_lookback': 3,
        'four_h_gate': 'soft',

        # ── comparison window ───────────────────────────────────────────────
        'windows': ['pm'],              # subset of ('am','pm')
        'adr_days': 14,

        # ── SMT detection ───────────────────────────────────────────────────
        # 'pivot'  - compare two confirmed swing pivots. Faithful to the
        #            "relative highs and lows" phrasing, but a pivot needs bars
        #            to its right, so the signal is always late and can only
        #            support a retrace entry.
        # 'breach' - the pool is history, the breach is live: at each bar, has
        #            this instrument run through the session's running extreme
        #            while the others have not. Strictly causal, actionable on
        #            the bar it prints, and the reading that makes a turtle-soup
        #            entry fillable at all.
        'smt_mode': 'pivot',
        'min_pool_bars': 6,
        'pool_gap_bars': 1,
        'max_breach_events': 1,
        'swing_lookback': 2,
        'pivot_dedupe_bars': 2,
        'match_tolerance_bars': 1,
        'min_window_bars': 12,

        # ── SMT quality ─────────────────────────────────────────────────────
        # require_smt=False is ARM B: the identical stack with the divergence
        # gate removed. It is the control, not a relaxation.
        'require_smt': True,
        'min_clarity_adr': 0.0,
        'max_clarity_adr': 1e9,         # a BAND - see the concept spec
        'min_sweep_depth_adr': 0.0,
        # SMT presumes the instruments are correlated *now*. Below this, a
        # "divergence" is two markets doing unrelated things.
        'corr_lookback': 288,           # 5m bars ~ one 24h session
        'min_abs_corr': 0.0,

        # ── pool ────────────────────────────────────────────────────────────
        'require_pool_live': True,
        'require_named_pool': False,    # also coincide with PDH/PDL/session ext
        'named_pool_tolerance_adr': 0.05,

        # ── raid rejection ──────────────────────────────────────────────────
        'require_rejection': True,
        'close_back_within_bars': 3,

        # ── reversal ────────────────────────────────────────────────────────
        'require_reversal': True,
        'confirmation': 'cisd_or_mss',  # 'cisd' | 'mss' | 'cisd_or_mss'
        'max_minutes_to_reversal': 45,
        'mss_swing_lookback': 3,

        # ── which instrument to trade (unresolved in the corpus) ────────────
        'instrument_selection': 'diverging',   # 'diverging'|'leader'|'fixed'
        'fixed_instrument': None,

        # ── entry ───────────────────────────────────────────────────────────
        # ICT teaches two entries for this setup and neither is a delayed FVG
        # retrace (Month 10 AM [15:56] "buying on a stop on a down-closed
        # candle"; [16:27] "identify that as turtle soup... buy it below the
        # old low"). Both act AT the extreme, which is what keeps the stop
        # tight enough for 1.5R to be reachable.
        #   'turtle_soup' - limit inside the sweep zone, below the raided level
        #   'buy_stop'    - stop entry beyond the last opposing-close candle
        #   'fvg'         - retrace into the displacement array (was 'limit')
        #   'market'      - at the open after confirmation
        'entry_style': 'fvg',
        # 0.0 = at the raided level, 1.0 = at the sweep extreme.
        'turtle_entry_depth': 0.5,
        'max_leg_lookback': 10,         # bars of manipulation leg for buy_stop
        'entry_arrays': ['fvg', 'ob'],
        'fvg_entry_depth': 'ce',        # 'ce' (consequent encroachment) | 'edge'
        'entry_fill_minutes': 120,
        'entry_search_bars': 12,        # bars past confirmation to look for the array

        # ── stop ────────────────────────────────────────────────────────────
        # 'smt_invalidation' puts the stop at the level whose breach kills the
        # premise: for a leader that is its own sweep extreme, for a diverging
        # instrument it is the level it refused. One rule, both cases.
        'stop_mode': 'smt_invalidation',   # | 'swept_extreme'
        'stop_buffer_adr': 0.02,
        # Two independent floors, because they guard different failures.
        #   ADR floor    - the stop must clear market noise.
        #   spread floor - the stop must be large enough that execution cost is
        #                  a small fraction of risk. The Judas post-mortem
        #                  measured a 0.3-pip spread against 5-6 pip stops
        #                  taking 43% of the edge; requiring risk >= 10x spread
        #                  caps round-turn spread at ~10% of risk.
        # A single 5%-of-ADR floor conflated the two and rejected every
        # turtle-soup entry outright - the tight stop is the entire point of
        # that technique, and the ADR floor was silently banning it.
        'min_stop_adr_pct': 0.02,
        'min_stop_spread_mult': 10.0,
        # What to do when the structural stop is tighter than the cost floor.
        #   'skip'  - stand aside. Treats the floor as a property of the setup.
        #   'widen' - keep the trade, push the stop out to the floor. This is
        #             what a trader does when structure is tighter than the
        #             spread: the setup is still valid, the risk is simply
        #             bigger, and R stays honest because the target scales with
        #             it. Turtle-soup entries at the first breach bar risk
        #             0.3-4 points against a 5-21 point spread floor on these
        #             CFDs, so 'skip' rejects the technique outright.
        'stop_floor_action': 'widen',
        'max_stop_adr_pct': 0.40,

        # ── target ──────────────────────────────────────────────────────────
        'target_mode': 'rr',            # 'rr' | 'liquidity' | 'time'
        'target_rr': 2.0,
        'target_min_rr': 2.0,
        'cap_target_at_adr': True,
        'adr_target_pct': 0.90,
        'target_pool_lookback_days': 5,
        'target_pool_tolerance_adr': 0.05,

        # ── session ─────────────────────────────────────────────────────────
        'exit_ny': None,                # None -> WINDOW_EXIT per window
        'one_trade_per_day': True,
    }

    def __init__(self, config: Optional[dict] = None):
        unknown = set(config or {}) - set(self.DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown SMTModel config keys: {sorted(unknown)}. "
                             f"Known: {sorted(self.DEFAULT_CONFIG)}")
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.checks: List[str] = []
        # Bias is a daily quantity but generate_signal runs once per window, so
        # without this the same daily swing scan runs twice per session.
        self._bias_cache: Dict[tuple, str] = {}

    # ── plumbing ────────────────────────────────────────────────────────────

    def log(self, item: str) -> None:
        self.checks.append(item)

    def _reject(self, stage: str, reason: str) -> dict:
        """Uniform rejection so the funnel can attribute every skipped day."""
        self.log(f"REJECT[{stage}] {reason}")
        return {'actionable': False, 'direction': None, 'instrument': None,
                'entry': None, 'stop': None, 'target': None,
                'reject_stage': stage, 'reject_reason': reason,
                'checks': list(self.checks)}

    def _skip(self, stage: str, reason: str) -> None:
        """Record why one side was abandoned.

        The side loop can fail at nine different places. Without recording the
        stage, every one of them collapses into a single 'no_setup' count and
        the rejection funnel cannot say whether the gate stack is working or
        one step is silently eating every candidate.
        """
        self.log(f"skip[{stage}] {reason}")
        self._last_skip = (stage, reason)

    def _adr(self, daily: pd.DataFrame, as_of: pd.Timestamp) -> float:
        """Mean daily range over `adr_days` sessions strictly before `as_of`."""
        n = self.config['adr_days']
        if daily is None or len(daily) < n + 1:
            return 0.0
        cut = localize_like(pd.Timestamp(as_of), daily.index)
        prior = daily[daily.index < cut]
        if len(prior) < n:
            return 0.0
        rng = (prior['high'] - prior['low']).tail(n)
        v = float(rng.mean())
        return v if np.isfinite(v) and v > 0 else 0.0

    # Both slicers go through the memoised NY parts and index positionally.
    # Boolean-masking an 806k-row frame once per session per instrument was the
    # entire runtime of the first full-period run.

    @staticmethod
    def _window_slice(df: pd.DataFrame, session_date: pd.Timestamp,
                      window: Tuple[str, str]) -> pd.DataFrame:
        return df.iloc[window_positions(df.index, session_date,
                                        window[0], window[1])]

    @staticmethod
    def _session_slice(df: pd.DataFrame, session_date: pd.Timestamp,
                       end_ny: str) -> pd.DataFrame:
        return df.iloc[before_positions(df.index, session_date, end_ny)]

    # ── checklist steps ─────────────────────────────────────────────────────

    def _bias(self, snapshot: SMTSnapshot) -> str:
        """HTF order flow. Chooses which side of the market gets compared."""
        if not self.config['use_daily_bias'] or self.config['bias_mode'] == 'off':
            return 'both'
        ref = self.config['bias_instrument'] or next(iter(snapshot.frames))
        key = (ref, pd.Timestamp(snapshot.session_date).normalize())
        if key in self._bias_cache:
            return self._bias_cache[key]
        dly = snapshot.daily.get(ref)
        if dly is None:
            return 'neutral'
        cut = localize_like(pd.Timestamp(snapshot.session_date), dly.index)
        hist = dly[dly.index < cut]
        if len(hist) < self.config['bias_lookback'] + 2:
            return 'neutral'
        four_h = None
        if snapshot.four_h and ref in snapshot.four_h:
            f = snapshot.four_h[ref]
            four_h = f[f.index < localize_like(pd.Timestamp(snapshot.session_date), f.index)]
        comp = daily_bias_components(
            hist,
            lookback=self.config['bias_lookback'],
            dealing_range_days=self.config['bias_dealing_range_days'],
            swing_lookback=self.config['bias_swing_lookback'],
            df_4h=four_h,
            four_h_gate=self.config['four_h_gate'],
        )
        mode = self.config['bias_mode']
        if mode == 'strict':
            out = comp['bias']
        elif mode == 'orderflow':
            out = comp['order_flow']
        else:
            raise ValueError(f"unknown bias_mode {mode!r}")
        self._bias_cache[key] = out
        return out

    def _pool_is_live(self, df: pd.DataFrame, level: float, side: str,
                      formed_at: pd.Timestamp, as_of: pd.Timestamp) -> bool:
        """True if `level` was still resting when the raid arrived.

        A low that had already been traded through earlier in the window is not
        liquidity any more; running it a second time is price revisiting a
        mitigated pool, which is not the setup.
        """
        seg = df[(df.index > formed_at) & (df.index < as_of)]
        if len(seg) == 0:
            return True
        return (seg['low'].min() >= level) if side == 'low' else (seg['high'].max() <= level)

    def _named_pool_hit(self, snapshot: SMTSnapshot, symbol: str, level: float,
                        side: str, adr: float) -> bool:
        """Does the compared level coincide with a standing pool (PDH/PDL)?"""
        dly = snapshot.daily.get(symbol)
        if dly is None or adr <= 0:
            return False
        cut = localize_like(pd.Timestamp(snapshot.session_date), dly.index)
        prior = dly[dly.index < cut]
        if len(prior) == 0:
            return False
        tol = self.config['named_pool_tolerance_adr'] * adr
        ref = float(prior['low'].iloc[-1]) if side == 'low' else float(prior['high'].iloc[-1])
        return abs(level - ref) <= tol

    def _rejection_ok(self, df: pd.DataFrame, level: float, side: str,
                      raid_time: pd.Timestamp) -> bool:
        """The raid must be rejected: a close back inside the level.

        Without this the pattern is a run, not a sweep - price accepted beyond
        the pool and kept going, which is continuation, not the reversal traded
        here.
        """
        n = self.config['close_back_within_bars']
        after = df[df.index >= localize_like(pd.Timestamp(raid_time), df.index)].head(n + 1)
        if len(after) == 0:
            return False
        return bool((after['close'] > level).any() if side == 'low'
                    else (after['close'] < level).any())

    def _reversal(self, df: pd.DataFrame, direction: str,
                  raid_time: pd.Timestamp, deadline: pd.Timestamp) -> Optional[dict]:
        """CISD or MSS confirming the turn, inside the reversal window."""
        mode = self.config['confirmation']
        seg = df[df.index <= localize_like(pd.Timestamp(deadline), df.index)]
        if len(seg) < 5:
            return None
        want = 'bullish' if direction == 'long' else 'bearish'

        out = None
        if mode in ('cisd', 'cisd_or_mss'):
            c = detect_cisd(seg, want, raid_time,
                            max_bars_after_sweep=len(seg))
            if c:
                out = {'kind': 'cisd', 'time': c['cisd_time'],
                       'level': c['reference_level']}
        if out is None and mode in ('mss', 'cisd_or_mss'):
            m = detect_mss(seg, want, swing_lookback=self.config['mss_swing_lookback'],
                           sweep_time=raid_time, max_bars_after_sweep=len(seg))
            if m:
                out = {'kind': 'mss', 'time': m['break_time'],
                       'level': m['broken_level']}
        if out is None:
            return None
        t = localize_like(pd.Timestamp(out['time']), df.index)
        limit = localize_like(pd.Timestamp(deadline), df.index)
        return out if t <= limit else None

    def _entry_array(self, df: pd.DataFrame, direction: str,
                     raid_time: pd.Timestamp, rev_time: pd.Timestamp) -> Optional[dict]:
        """The FVG or OB left by the displacement out of the raid.

        The search runs from the raid to `entry_search_bars` PAST the
        confirmation, not up to it. The gap that a trader actually enters is
        carved by the displacement leg that *causes* the CISD/MSS, and a
        three-candle gap is only complete one bar after its middle candle — so
        a window ending at the confirmation bar frequently contains no
        completed gap at all. That off-by-a-leg error rejected 10 of 11
        surviving setups in the first run.

        No lookahead is introduced: `confirm_time` is the bar on which the gap
        became visible, and the caller only arms the limit strictly after it.
        """
        want = 'bullish' if direction == 'long' else 'bearish'
        lo = localize_like(pd.Timestamp(raid_time), df.index)
        rev = localize_like(pd.Timestamp(rev_time), df.index)
        after_rev = df[df.index > rev]
        n_extra = self.config['entry_search_bars']
        hi = after_rev.index[min(n_extra, len(after_rev)) - 1] if len(after_rev) else rev
        seg = df[(df.index >= lo) & (df.index <= hi)]
        if len(seg) < 3:
            return None

        def confirm_of(ts) -> Optional[pd.Timestamp]:
            """Bar on which the array became visible to a live trader."""
            loc = seg.index.get_indexer([ts], method='nearest')[0]
            # find_fvgs stamps a gap to its MIDDLE candle; the third candle is
            # what completes it, so nothing can be acted on before loc + 1.
            return seg.index[loc + 1] if loc + 1 < len(seg) else None

        arrays = self.config['entry_arrays']
        best = None
        if 'fvg' in arrays:
            for g in find_fvgs(seg, direction=want):
                ct = confirm_of(g.timestamp)
                if ct is None:
                    continue
                px = ((g.top + g.bottom) / 2.0
                      if self.config['fvg_entry_depth'] == 'ce' else g.entry)
                cand = {'kind': 'fvg', 'time': g.timestamp, 'confirm_time': ct,
                        'entry': float(px), 'top': float(g.top),
                        'bottom': float(g.bottom)}
                # Earliest qualifying array, not the latest: a later gap means
                # price has already travelled, which is a worse entry and a
                # wider stop for the same premise.
                if best is None or cand['time'] < best['time']:
                    best = cand
        if best is None and 'ob' in arrays:
            try:
                obs = find_order_blocks(seg, direction=want)
            except TypeError:
                obs = find_order_blocks(seg)
            for ob in obs or []:
                top = float(getattr(ob, 'top', np.nan))
                bottom = float(getattr(ob, 'bottom', np.nan))
                ts = getattr(ob, 'timestamp', None)
                if not np.isfinite(top) or not np.isfinite(bottom) or ts is None:
                    continue
                ct = confirm_of(ts)
                if ct is None:
                    continue
                px = bottom if direction == 'long' else top
                cand = {'kind': 'ob', 'time': ts, 'confirm_time': ct,
                        'entry': float(px), 'top': top, 'bottom': bottom}
                if best is None or cand['time'] < best['time']:
                    best = cand
        return best

    def _doctrinal_entry(self, df: pd.DataFrame, direction: str, style: str,
                         raid_level: float, sweep_extreme: float, adr: float,
                         raid_time: pd.Timestamp,
                         knowable: pd.Timestamp) -> Optional[dict]:
        """The two entries ICT actually teaches for index SMT.

        Both act at the extreme rather than waiting for a retrace into a
        displacement array. That is the point: the delayed entry is what made
        our stop wide enough that only 29% of trades ever reached 1.5R, with a
        median adverse excursion (1.03R) larger than the median favourable one
        (0.76R).

        ``turtle_soup`` — "identify that as turtle soup... buy it below the old
        low" (M10-AM [16:27]). A limit resting inside the sweep zone, between
        the raided level and the sweep extreme. `turtle_entry_depth` picks where:
        0.0 at the level, 1.0 at the extreme.

        ``buy_stop`` — "we can be ahead of the curve by buying on a stop on a
        down-closed candle, whatever that high is" (M10-AM [15:56]). A stop
        entry beyond the high of the last down-close candle of the raid leg
        (mirrored for shorts). This is the buying-strength variant.

        Both are armed no earlier than `knowable`, so neither can act on a
        pivot that had not yet formed.
        """
        if not np.isfinite(raid_level) or not np.isfinite(sweep_extreme):
            return None

        if style == 'turtle_soup':
            depth = float(self.config['turtle_entry_depth'])
            entry = raid_level + depth * (sweep_extreme - raid_level)
            # The limit rests from the moment the event is knowable; a fill
            # requires price to return into the zone, which simulate_entry
            # enforces by only scanning bars strictly after this time.
            return {'kind': 'turtle_soup', 'time': raid_time,
                    'confirm_time': knowable, 'entry': float(entry),
                    'top': float(max(raid_level, sweep_extreme)),
                    'bottom': float(min(raid_level, sweep_extreme))}

        # buy_stop: walk back from the raid bar over the manipulation leg and
        # take the extreme of the candles that drove it, which is the level a
        # stop order sits beyond.
        idx = df.index.get_indexer([localize_like(pd.Timestamp(raid_time),
                                                  df.index)], method='nearest')
        if len(idx) == 0 or idx[0] < 0:
            return None
        i = int(idx[0])
        lookback = int(self.config['max_leg_lookback'])
        lo = max(i - lookback, 0)
        leg = df.iloc[lo:i + 1]
        if len(leg) == 0:
            return None
        if direction == 'long':
            opposing = leg[leg['close'] < leg['open']]      # down-close candles
            ref = float(opposing['high'].max()) if len(opposing) else float(leg['high'].max())
        else:
            opposing = leg[leg['close'] > leg['open']]      # up-close candles
            ref = float(opposing['low'].min()) if len(opposing) else float(leg['low'].min())
        if not np.isfinite(ref):
            return None
        buf = float(self.config['stop_buffer_adr']) * adr * 0.25
        entry = ref + buf if direction == 'long' else ref - buf
        return {'kind': 'buy_stop', 'time': raid_time, 'confirm_time': knowable,
                'entry': float(entry),
                'top': float(max(ref, sweep_extreme)),
                'bottom': float(min(ref, sweep_extreme))}

    def _target_pools(self, df: pd.DataFrame, direction: str,
                      before: pd.Timestamp, adr: float) -> list:
        side = 'highs' if direction == 'long' else 'lows'
        tol = max(self.config['target_pool_tolerance_adr'] * adr, 1e-9)
        try:
            return find_relative_equal_levels(
                df, before=before, side=side,
                lookback_days=self.config['target_pool_lookback_days'],
                tolerance_pts=tol,
            )
        except Exception:
            return []

    def _adr_cap(self, df_today: pd.DataFrame, direction: str, adr: float) -> Optional[float]:
        """Ceiling on the target: don't project past the day's expected range."""
        if not self.config['cap_target_at_adr'] or adr <= 0 or len(df_today) == 0:
            return None
        pct = self.config['adr_target_pct']
        return (float(df_today['low'].min()) + adr * pct if direction == 'long'
                else float(df_today['high'].max()) - adr * pct)

    # ── main ────────────────────────────────────────────────────────────────

    def generate_signal(self, snapshot: SMTSnapshot,
                        window_name: str = 'pm') -> dict:
        """Evaluate one (session, window). Rule of exclusion throughout."""
        self.checks = []
        self._last_skip: Optional[Tuple[str, str]] = None
        cfg = self.config
        window = WINDOWS[window_name]
        frames = snapshot.frames
        symbols = list(frames)

        adr = {s: self._adr(snapshot.daily.get(s), snapshot.session_date)
               for s in symbols}
        if any(adr[s] <= 0 for s in symbols):
            return self._reject('adr', 'insufficient daily history for ADR')
        self.log(f"ADR{cfg['adr_days']}: " +
                 ', '.join(f"{s}={adr[s]:.4f}" for s in symbols))

        # 1 ── bias chooses the side
        bias = self._bias(snapshot)
        if cfg['use_daily_bias'] and bias == 'neutral':
            return self._reject('bias', 'daily bias neutral')
        sides = (['low', 'high'] if bias == 'both'
                 else ['low' if bias == 'bullish' else 'high'])
        self.log(f"bias={bias} -> compare {'/'.join(sides)}")

        exit_ny = cfg['exit_ny'] or WINDOW_EXIT[window_name]

        for side in sides:
            direction = 'long' if side == 'low' else 'short'

            # 2 ── the comparison window
            if cfg['smt_mode'] == 'breach':
                events = find_smt_breach_events(
                    frames, snapshot.session_date, window, side,
                    scale=adr, inverse=snapshot.inverse,
                    min_pool_bars=cfg['min_pool_bars'],
                    pool_gap_bars=cfg['pool_gap_bars'],
                    min_window_bars=cfg['min_window_bars'],
                    max_events=cfg['max_breach_events'],
                    corr_lookback=cfg['corr_lookback'],
                    min_abs_corr=cfg['min_abs_corr'],
                )
            else:
                events = find_smt_events(
                    frames, snapshot.session_date, window, side,
                    scale=adr, inverse=snapshot.inverse,
                    swing_lookback=cfg['swing_lookback'],
                    pivot_dedupe_bars=cfg['pivot_dedupe_bars'],
                    match_tolerance_bars=cfg['match_tolerance_bars'],
                    min_window_bars=cfg['min_window_bars'],
                    corr_lookback=cfg['corr_lookback'],
                    min_abs_corr=cfg['min_abs_corr'],
                )

            if cfg['require_smt']:
                if not events:
                    self._skip('smt', f"no divergence in {window[0]}-{window[1]} NY")
                    continue
                event = events[0] if cfg['smt_mode'] == 'breach' else events[-1]
                self.log(f"SMT {event}")
            else:
                # ARM B: no divergence required. Synthesise the same anchor from
                # the window's own last two relative extremes so the rest of the
                # stack is byte-identical and the comparison is honest.
                event = self._tandem_anchor(frames, snapshot, window, side, adr)
                if event is None:
                    self._skip('anchor', "no comparable extremes in window")
                    continue
                # Arm B is "the same setup without the cross-instrument
                # confirmation", so it still requires a raid: the instrument
                # must have taken out its own prior relative extreme. An anchor
                # where nothing was swept is not a control for this model, it is
                # a different setup entirely.
                if not event.leader:
                    self._skip('anchor', "no raid of the prior relative extreme")
                    continue
                self.log(f"anchor (SMT gate OFF) {event}")

            # 5 ── SMT quality band
            if cfg['require_smt']:
                if np.isfinite(event.clarity):
                    if event.clarity < cfg['min_clarity_adr']:
                        self._skip('clarity_low', f"clarity {event.clarity:.3f} < {cfg['min_clarity_adr']:.3f} ADR - not obvious")
                        continue
                    if event.clarity > cfg['max_clarity_adr']:
                        self._skip('clarity_high', f"clarity {event.clarity:.3f} > {cfg['max_clarity_adr']:.3f} ADR - decorrelation")
                        continue
                if (np.isfinite(event.sweep_depth)
                        and event.sweep_depth < cfg['min_sweep_depth_adr']):
                    self._skip('sweep_depth', f"sweep depth {event.sweep_depth:.3f} < {cfg['min_sweep_depth_adr']:.3f} ADR - a graze")
                    continue

            # choose the instrument to trade. With the SMT gate off there is no
            # leader/diverging split to choose between - the anchor is built on
            # one instrument - so the selection rule collapses to that one.
            sym = (self._select_instrument(event, symbols) if cfg['require_smt']
                   else (cfg['fixed_instrument'] or symbols[0]))
            if sym is None or sym not in frames:
                self._skip('select', f"no instrument under rule {cfg['instrument_selection']!r}")
                continue
            if sym not in event.levels:
                self._skip('select', f"{sym} has no compared level in this event")
                continue
            df = frames[sym]
            v1, v2 = event.levels[sym]
            eff_side = ('high' if (sym in snapshot.inverse) ^ (side == 'high')
                        else 'low')
            # The instrument we trade may be inverse to the group; its own trade
            # direction follows its own compared side, not the group's.
            direction = 'long' if eff_side == 'low' else 'short'

            # 3 ── the raided level must still have been resting
            raid_sym = event.leader[0] if event.leader else sym
            raid_df = frames[raid_sym]
            raid_v1, _ = event.levels[raid_sym]
            raid_eff = ('high' if (raid_sym in snapshot.inverse) ^ (side == 'high')
                        else 'low')
            if cfg['require_pool_live'] and not self._pool_is_live(
                    raid_df, raid_v1, raid_eff, event.pivot_times[0], event.time):
                self._skip('pool_dead', f"pool at {raid_v1:.4f} already mitigated before the raid")
                continue
            if cfg['require_named_pool'] and not self._named_pool_hit(
                    snapshot, raid_sym, raid_v1, raid_eff, adr[raid_sym]):
                self._skip('pool_unnamed', "raided level is not a standing PDH/PDL pool")
                continue

            # 4 ── the raid must be rejected
            if cfg['require_rejection'] and not self._rejection_ok(
                    raid_df, raid_v1, raid_eff, event.time):
                self._skip('rejection', f"no close back inside within {cfg['close_back_within_bars']} bars - a run")
                continue

            # 6 ── reversal
            deadline = event.time + pd.Timedelta(minutes=cfg['max_minutes_to_reversal'])
            session_df = self._session_slice(df, snapshot.session_date, exit_ny)
            if cfg['require_reversal']:
                rev = self._reversal(session_df, direction, event.time, deadline)
                if rev is None:
                    self._skip('reversal', f"no {cfg['confirmation']} within {cfg['max_minutes_to_reversal']}min")
                    continue
                self.log(f"reversal {rev['kind']} @ {rev['time']}")
            else:
                rev = {'kind': 'none', 'time': event.time, 'level': v2}

            # 10 ── negation before we even place the order
            if cfg['require_smt']:
                dead = negated_at(frames, event, until=rev['time'],
                                  inverse=snapshot.inverse)
                if dead is not None:
                    self._skip('negated', f"SMT negated at {dead} before entry")
                    continue

            # Nothing may be acted on before the event was knowable. `event.time`
            # is the pivot bar; the pivot needs bars to its right to exist at all.
            knowable = localize_like(
                pd.Timestamp(event.confirmed_time or event.time), df.index)

            # 7 ── entry
            if cfg['entry_style'] in ('turtle_soup', 'buy_stop'):
                # Both levels must come from the instrument being TRADED. Mixing
                # the leader's level with the selected instrument's put the
                # entry halfway between two differently-priced markets (NQ ~4.5k
                # against YM ~17k), which rejected every setup as stop_wide.
                if cfg['entry_style'] == 'turtle_soup' and sym not in event.leader:
                    # Turtle soup is by definition the instrument that ran the
                    # stops - "the one that does go lower, that's a turtle
                    # soup" (M10-AM [18:30]). A diverging instrument has no
                    # sweep zone to buy into, so pair this entry with
                    # instrument_selection: leader.
                    self._skip('turtle_needs_leader',
                               f"{sym} did not sweep; turtle soup must be traded "
                               f"on the leader ({'/'.join(event.leader)})")
                    continue
                built = self._doctrinal_entry(
                    session_df, direction, cfg['entry_style'],
                    raid_level=v1, sweep_extreme=v2, adr=adr[sym],
                    raid_time=event.time, knowable=knowable)
                if built is None:
                    self._skip('entry_array',
                               f"no {cfg['entry_style']} entry could be built")
                    continue
                entry = built['entry']
                array = built
                arm_time = max(knowable,
                               localize_like(pd.Timestamp(built['confirm_time']),
                                             df.index))
            elif cfg['entry_style'] == 'market':
                floor = max(knowable,
                            localize_like(pd.Timestamp(rev['time']), df.index))
                nxt = session_df[session_df.index > floor]
                if len(nxt) == 0:
                    self._skip('entry_bar', "no bar after confirmation to enter on")
                    continue
                entry = float(nxt['open'].iloc[0])
                array = {'kind': 'market', 'time': floor, 'confirm_time': floor}
                arm_time = floor
            else:
                array = self._entry_array(session_df, direction, event.time, rev['time'])
                if array is None:
                    self._skip('entry_array', f"no {'/'.join(cfg['entry_arrays'])} left by the displacement")
                    continue
                entry = array['entry']
                # Arm the limit only once the array was visible AND the event
                # itself was knowable. max() keeps both as floors so neither an
                # early gap nor an unconfirmed pivot can back-date the entry.
                arm_time = max(knowable,
                               localize_like(pd.Timestamp(rev['time']), df.index),
                               localize_like(pd.Timestamp(array['confirm_time']), df.index))

            # 8 ── stop
            if cfg['stop_mode'] == 'smt_invalidation':
                # The level whose breach kills the premise. For a leader that is
                # its own sweep extreme (v2); for a diverging instrument it is
                # the level it refused (v1). min/max picks the right one without
                # branching on role.
                anchor = min(v1, v2) if direction == 'long' else max(v1, v2)
            else:
                anchor = v2

            # The entry array must also be inside the stop. A pivot level is
            # measured in a +/-1 bar neighbourhood, so the sweep can run deeper
            # than the level it is anchored to, leaving the FVG being entered
            # BELOW the invalidation level - which produced a long whose stop
            # sat above its own entry. A real stop goes beyond both the
            # invalidation level and the array being traded from.
            far = (array.get('bottom') if direction == 'long'
                   else array.get('top'))
            if far is None or not np.isfinite(far):
                far = entry
            anchor = (min(anchor, float(far)) if direction == 'long'
                      else max(anchor, float(far)))

            buf = cfg['stop_buffer_adr'] * adr[sym]
            stop = anchor - buf if direction == 'long' else anchor + buf

            risk = abs(entry - stop)
            if risk <= 0:
                self._skip('risk', f"non-positive risk ({entry:.4f} vs {stop:.4f})")
                continue
            # Belt and braces on the geometry the runner asserts on.
            if (direction == 'long' and stop >= entry) or \
               (direction == 'short' and stop <= entry):
                self._skip('risk', f"stop on the wrong side of entry "
                                   f"({entry:.4f} vs {stop:.4f})")
                continue
            lo = cfg['min_stop_adr_pct'] * adr[sym]
            spread_floor = 0.0
            mult = cfg['min_stop_spread_mult']
            if mult and 'spread' in session_df.columns and len(session_df):
                med = float(session_df['spread'].median())
                if np.isfinite(med) and med > 0:
                    spread_floor = mult * med
            lo = max(lo, spread_floor)
            hi = cfg['max_stop_adr_pct'] * adr[sym]
            widened = False
            if lo > 0 and risk < lo:
                which = ('spread' if spread_floor >= cfg['min_stop_adr_pct'] * adr[sym]
                         else 'ADR')
                if cfg['stop_floor_action'] == 'widen':
                    # Push the stop out to the floor, keeping the entry. The
                    # setup is unchanged; only the risk it is sized against is.
                    stop = entry - lo if direction == 'long' else entry + lo
                    risk = lo
                    widened = True
                    self.log(f"stop widened to the {which} floor ({lo:.4f})")
                else:
                    self._skip('stop_tight',
                               f"stop {risk:.4f} < {which} floor ({lo:.4f}) - "
                               f"execution cost would dominate the risk")
                    continue
            if hi > 0 and risk > hi:
                self._skip('stop_wide', f"stop {risk:.4f} > {cfg['max_stop_adr_pct']:.0%} of ADR ({hi:.4f}) - delivery, not a fake")
                continue

            # 9 ── target
            target = self._resolve_target(df, snapshot, direction, entry, stop,
                                          event.time, adr[sym], exit_ny)
            if target is None:
                self._skip('target', f"no target satisfied min RR {cfg['target_min_rr']}")
                continue

            self.log(f"ACTIONABLE {sym} {direction} entry={entry:.4f} "
                     f"stop={stop:.4f} target={target:.4f} "
                     f"RR={abs(target - entry) / risk:.2f}")
            return {
                'actionable': True,
                'direction': direction,
                'instrument': sym,
                'entry': float(entry),
                'stop': float(stop),
                'target': float(target),
                'entry_time': arm_time,
                # How the simulator must fill it, which is not the same thing
                # as which ICT technique produced it.
                'entry_style': {'turtle_soup': 'limit', 'buy_stop': 'stop',
                                'fvg': 'limit', 'limit': 'limit',
                                'market': 'market'}[cfg['entry_style']],
                'entry_technique': cfg['entry_style'],
                'entry_array': array['kind'],
                'window': window_name,
                'side': side,
                'bias': bias,
                'smt': cfg['require_smt'],
                'leader': ','.join(event.leader),
                'diverging': ','.join(event.diverging),
                'correlation': (float(event.correlation)
                                if event.correlation is not None
                                and np.isfinite(event.correlation) else None),
                'clarity_adr': float(event.clarity) if np.isfinite(event.clarity) else None,
                'sweep_depth_adr': float(event.sweep_depth) if np.isfinite(event.sweep_depth) else None,
                'raid_time': event.time,
                'reversal_kind': rev['kind'],
                'adr': float(adr[sym]),
                'risk': float(risk),
                'stop_widened': bool(widened),
                'exit_ny': exit_ny,
                'reject_stage': None,
                'reject_reason': None,
                'checks': list(self.checks),
            }

        # Attribute the failure to the furthest stage any side reached, so the
        # funnel shows which gate is actually doing the filtering.
        if self._last_skip is not None:
            return self._reject(*self._last_skip)
        return self._reject('no_setup', 'no side produced an actionable setup')

    # ── helpers used by generate_signal ─────────────────────────────────────

    def _select_instrument(self, event: SMTEvent, symbols: Sequence[str]) -> Optional[str]:
        """Resolve the corpus's three-way disagreement by configuration.

        The source teaches all of: turtle-soup the leader, no preference, and
        always trade one fixed instrument. This model takes no position; the
        rule is an experiment axis. See smt-divergence.md.
        """
        rule = self.config['instrument_selection']
        if rule == 'fixed':
            return self.config['fixed_instrument'] or (symbols[0] if symbols else None)
        if rule == 'leader':
            return event.leader[0] if event.leader else None
        if rule == 'diverging':
            return event.diverging[0] if event.diverging else None
        raise ValueError(f"unknown instrument_selection {rule!r}")

    def _tandem_anchor(self, frames, snapshot, window, side, adr) -> Optional[SMTEvent]:
        """ARM B anchor: the window's last two relative extremes, no divergence.

        Built with the same pivot machinery so the only difference between arm A
        and arm B is whether the instruments had to disagree.
        """
        sym = self.config['fixed_instrument'] or next(iter(frames))
        events = find_smt_events(
            {sym: frames[sym]}, snapshot.session_date, window, side,
            scale={sym: adr[sym]}, inverse=snapshot.inverse,
            swing_lookback=self.config['swing_lookback'],
            pivot_dedupe_bars=self.config['pivot_dedupe_bars'],
            match_tolerance_bars=self.config['match_tolerance_bars'],
            min_window_bars=self.config['min_window_bars'],
        )
        if events:
            return events[-1]
        # A single instrument can never "diverge" from itself, so find_smt_events
        # returns nothing. Rebuild the anchor directly from its pivots.
        return self._solo_anchor(frames, snapshot, window, side, adr, sym)

    def _solo_anchor(self, frames, snapshot, window, side, adr, sym) -> Optional[SMTEvent]:
        from concepts.ict.entries.smt_divergence.smt_divergence import _pivots
        df = frames[sym]
        w = self._window_slice(df, snapshot.session_date, window)
        if len(w) < self.config['min_window_bars']:
            return None
        arr = (w['low'] if side == 'low' else w['high']).to_numpy()
        piv = _pivots(arr, self.config['swing_lookback'], low=(side == 'low'))
        ded: List[int] = []
        for p in sorted(piv.tolist()):
            if not ded or p - ded[-1] > self.config['pivot_dedupe_bars']:
                ded.append(p)
        if len(ded) < 2:
            return None
        i1, i2 = ded[-2], ded[-1]
        v1, v2 = float(arr[i1]), float(arr[i2])
        made = (v2 < v1) if side == 'low' else (v2 > v1)
        signed = (v1 - v2) if side == 'low' else (v2 - v1)
        return SMTEvent(
            time=w.index[i2], side=side,
            leader=(sym,) if made else (),
            diverging=() if made else (sym,),
            pivot_times=(w.index[i1], w.index[i2]),
            levels={sym: (v1, v2)},
            clarity=float('nan'), sweep_depth=signed / adr[sym] if adr[sym] else float('nan'),
            window=window, scale={sym: adr[sym]},
        )

    def _resolve_target(self, df, snapshot, direction, entry, stop,
                        before, adr, exit_ny) -> Optional[float]:
        cfg = self.config
        risk = abs(entry - stop)
        if cfg['target_mode'] == 'time':
            # Exit is purely time-of-day; the target is a formality placed far
            # enough away that it never fills first.
            far = 1000.0 * risk
            return entry + far if direction == 'long' else entry - far
        if cfg['target_mode'] == 'rr':
            t = (entry + cfg['target_rr'] * risk if direction == 'long'
                 else entry - cfg['target_rr'] * risk)
        else:
            pools = self._target_pools(df, direction, before, adr)
            t = find_liquidity_target(direction, entry, stop, pools,
                                      min_rr=cfg['target_min_rr'])
        today = self._session_slice(df, snapshot.session_date, exit_ny)
        cap = self._adr_cap(today, direction, adr)
        if cap is not None:
            # The cap is a CEILING on ambition, never a relocation. When the
            # session has already delivered most of its expected range, the cap
            # can sit on the far side of the entry; a bare min()/max() then
            # moves the target through the entry and prices a long's target
            # BELOW its own stop. simulate_exit fires "target hit" on the very
            # next bar and books a ~2R loss labelled TP - which is how 31% of
            # signals became fake winners in the first full run.
            #
            # No room left is a reason to stand aside, not to invert the trade.
            if direction == 'long':
                if cap <= entry:
                    return None
                t = min(t, cap)
            else:
                if cap >= entry:
                    return None
                t = max(t, cap)
        # Belt and braces: whatever produced `t`, it must sit beyond the entry
        # in the direction of the trade.
        if (direction == 'long' and t <= entry) or (direction == 'short' and t >= entry):
            return None
        if risk > 0 and abs(t - entry) / risk < cfg['target_min_rr']:
            return None
        return float(t)
