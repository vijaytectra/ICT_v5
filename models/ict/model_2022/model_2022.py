"""
ICT 2022 Model
==================================

``context -> sweep -> MSS w/ displacement -> retrace into FVG -> deliver to draw``

Checklist (bearish — inverted for bullish):
  1. Daily bias  — order flow + draw + 4H structure must agree; P/D is a reference signal.
  2. Pool set    — PDH/PDL, opening range, REH/REL from the 15m chart.
  3. Sweep       — a buy-side pool raided inside a killzone.
  4. MSS         — 2m close through the prior swing with displacement + FVG.
  5. Entry/stop  — limit into FVG; stop beyond swept extreme.
  6. Target      — opposing liquidity, 1:3 R:R floor.

See model-2022.md
"""

import dataclasses
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from infrastructure.registry import concept, params as registry_params
from models.intermediate.daily_bias import daily_bias_components
from concepts import find_fvgs, FVG
from concepts import session_mask
from concepts import first_breach
from concepts import detect_mss, find_relative_equal_levels
from concepts import premium_discount
from concepts import (opening_range, ons_range, chicago_range, asian_range,
                      am_hour_range, find_order_blocks, find_breaker_blocks)
from concepts import find_liquidity_target
from infrastructure.time_utils import localize_like


@dataclass
class Model2022Snapshot:
    """
    Attributes:
        df_2m:        2-minute bars for the session (entry timeframe).
        df_15m:       15-minute bars for pool detection.
        df_daily:     Daily bars for bias (pre-session).
        session_date: Calendar date being analyzed (tz-naive local date).
        df_4h:        4-hour bars for bias confirmation (optional).
    """
    df_2m: pd.DataFrame
    df_15m: pd.DataFrame
    df_daily: pd.DataFrame
    session_date: pd.Timestamp
    df_4h: Optional[pd.DataFrame] = None


@concept("model-2022", depends_on=["daily-bias", "fair-value-gap", "killzones", "liquidity-sweep-stop-hunt", "market-structure-shift", "relative-equal-highs-lows", "sessions-and-ranges", "targets-and-exits"])
class Model2022:
    """
    ICT 2022 Model — FVG + Liquidity Sweep intraday model.

    Config keys:
        eq_tolerance_pts      float  Points for REH/REL cluster tolerance (default 5)
        swing_lookback        int    Bar lookback for SwingPointScanner (default 1)
        killzones             list   [(start,end), ...] NY 'HH:MM' windows for sweep
        use_asian_range       bool   Add the prior 20:00-00:00 NY Asian range as a pool
        use_am_hour_range     bool   Add the 09:00-10:00 NY hour range as a pool (default False)
        am_hour_start         str    NY start of that hour range (default '09:00')
        daily_lookback        int    Daily bars for order-flow vote (default 5)
        dealing_range_days    int    Daily bars for premium/discount range (default 20)
        reh_lookback_days     int    Days back to scan 15m for equal pools (default 5)
        opening_range_start   str    NY start time for opening range (default '09:30')
        opening_range_minutes int    Minutes after opening_range_start (default 30)
        max_bars_sweep_to_mss int    Cap on 2m bars between sweep and MSS (default 30)
        target_min_rr         float  Minimum R:R; also the fixed fallback (default 3.0)
        max_stop_pts          float  Max allowed stop distance in points; wider stops excluded (default 20.0)
        stop_buffer_pts       float  Buffer beyond the structural stop level (default 0.0; 2-5 pips on FX)
        min_sweep_wick_pct    float  Reject sweeps whose rejection wick is < this share of the bar range (default 0.0)
        require_sweep_close_inside bool Reject sweeps that close beyond the raided level (default False)
        min_kz_minutes_remaining int  Minutes of the killzone that must remain at the sweep (default 0)
        fvg_entry_depth       str    'edge' (near edge, default) or 'ce' (array midpoint)
        entry_arrays          list   Arrays to try in order: 'fvg', 'ob', 'breaker' (default ['fvg'])
        array_lookback_bars   int    Bars before the MSS scanned for OB/breaker (default 20)
        min_fvg_atr_mult      float  Reject FVGs shorter than this multiple of ATR (default 0.0 = off)
        require_displacement_quality bool Enforce body-dominance / opposing-wick checks on the MSS candle
        require_pd_zone       bool   Gate on premium/discount alignment with bias (default False)
        target_mode           str    'liquidity' (default), 'range_end' or 'pool_only'
        range_end_enforce_rr  bool   Apply target_min_rr to the range_end target (default True)
        require_mss           bool   Require an MSS between sweep and entry (default True)
    """

    # The opening-range window is owned by the sessions-and-ranges concept; its
    # canonical default lives in that concept's frontmatter (see __init__). The
    # values below are only fallbacks if the frontmatter can't be read.
    DEFAULT_CONFIG = {
        # ── Timing ────────────────────────────────────────────────────────────
        'eq_tolerance_pts': 5.0,
        'swing_lookback': 1,
        # Swing width used for the MSS specifically. `swing_lookback` also drives
        # REH/REL clustering, so tuning MSS quality through it changes the pool
        # set at the same time. None here means "follow swing_lookback", keeping
        # existing behaviour; set it to isolate the MSS.
        #
        # Two independent implementations put MSS sensitivity at the centre:
        # a 5-candle swing with a 1.0x ATR body produced 94 direction-flipping
        # signals in two years, while 10 candles with a 1.5x ATR body produced
        # 17 usable ones; a second treats the MSS as a 15m event with 5m only as
        # execution detail. This model runs MSS on 2m bars at lookback 1.
        'mss_swing_lookback': None,
        # REH/REL clustering also reads `swing_lookback`, so tuning the MSS
        # through it silently rebuilt the pool set at the same time and the
        # MSS sweep was never a clean single-variable experiment. Pin this to
        # hold the pools fixed while `mss_swing_lookback` moves. None follows
        # `swing_lookback`, preserving existing behaviour.
        'reh_rel_swing_lookback': None,
        'killzones': [('02:00', '05:00'), ('07:00', '10:00'), ('13:30', '16:00')],
        'daily_lookback': 5,
        'dealing_range_days': 20,
        'reh_lookback_days': 5,
        'opening_range_start': '09:30',
        'opening_range_minutes': 30,
        'max_bars_sweep_to_mss': 30,
        # ── Pool component toggles ─────────────────────────────────────────────
        'use_pdh_pdl': True,
        'use_ons': True,
        # Forex analogue of ONS/Chicago: the 20:00-00:00 NY Asian range, whose
        # stops London raids first. Off by default so index-futures runs are
        # unchanged; the forex configs turn it on.
        'use_asian_range': False,
        # The 09:00-10:00 NY hour range - the pool the Mobius Silver Bullet
        # definition is built on (sweep one side after 10:00, target the other).
        'use_am_hour_range': False,
        'am_hour_start': '09:00',
        'use_chicago_prior': True,
        'use_chicago_same_day': True,
        'use_opening_range': True,
        'use_reh_rel': True,
        # ── Stop placement ────────────────────────────────────────────────────
        # stop_mode: 'wick'  → stop at swept wick extreme (capped by max_stop_pts)
        #            'atr'   → stop at entry ± stop_atr_mult × ATR(stop_atr_period)
        #            'fixed' → stop at entry ± stop_fixed_pts
        'stop_mode': 'wick',
        'stop_atr_period': 14,
        'stop_atr_mult': 1.0,
        'stop_fixed_pts': 20.0,
        # Buffer added beyond the structural stop level so routine noise does not
        # wick us out. ICT teaches 2-5 pips on FX / 2-5 points on indices; a
        # pixel-precise stop at the exact invalidation is a known failure mode.
        # Default 0.0 preserves the previous index-futures behaviour.
        'stop_buffer_pts': 0.0,
        # Sweep quality (ICT 2025 Silver Bullet refinement, and the antidote to
        # the "sweep-and-continue" failure mode): a genuine stop hunt leaves a
        # long rejection wick and closes back inside the level it raided. A bar
        # that closes beyond the pool was run-and-continue - the liquidity was
        # fuel for the move, not a reversal trigger.
        # Both default to off so existing runs are unchanged.
        'min_sweep_wick_pct': 0.0,      # e.g. 0.60 per the 2025 SB rules
        'require_sweep_close_inside': False,
        # Window-edge guard (SB failure mode 4): a setup firing in the last
        # minutes of a killzone rolls into the next session unfinished, because
        # the institutional participation ends with the window. Minutes of the
        # window that must still remain at the sweep. 0 disables the check.
        'min_kz_minutes_remaining': 0,
        # FVG entry depth: 'edge' takes the near edge (first touch, the original
        # behaviour); 'ce' takes Consequent Encroachment, the FVG midpoint, which
        # ICT made the 2025 default - near-edge fills get stopped when price
        # rebalances the gap fully.
        'fvg_entry_depth': 'edge',
        # Entry arrays, tried in order until one is found. The 2022 model is
        # FVG-only by spec; the ICT entry checklist in the Mobius knowledge base
        # accepts "FVG, OB, OTE or Breaker Block", and the Mobius Silver Bullet
        # definition names Breaker Block / Order Block / FVG. Default keeps the
        # spec's FVG-only behaviour.
        'entry_arrays': ['fvg'],
        # Bars before the MSS to scan when looking for an OB or breaker (they
        # need swing context, unlike an FVG which is a 3-bar pattern).
        'array_lookback_bars': 20,
        # Array/displacement quality. Both come straight from the Common Mistakes
        # sections of the ict-knowledge-library concepts:
        #   fair-value-gap: "Tiny FVGs. A 1-2 pip FVG on M5 is noise; filter by ATR."
        #   displacement:   "Calling any wide candle displacement. Without the
        #                    body-dominance and opposing-wick checks, wide candles
        #                    in choppy markets aren't displacement."
        # Defaults are permissive so existing runs are unchanged.
        'min_fvg_atr_mult': 0.0,            # FVG height as a multiple of ATR
        'require_displacement_quality': False,
        # The 2022 model requires an MSS between the sweep and the entry array.
        # The Silver Bullet sources do not: their sequence is sweep -> breaker /
        # OB / FVG -> entry, with no structure-shift step. Set False to run that
        # sequence; the entry array is then searched from the sweep bar forward.
        'require_mss': True,
        'displacement_body_mult': 1.5,      # body >= 1.5x recent average body
        'displacement_body_pct': 0.70,      # body >= 70% of candle range
        'displacement_opp_wick_pct': 0.20,  # opposing wick <= 20% of range
        # Premium/discount as a hard gate. The project's own spec deliberately
        # keeps this a logged check, reasoning that a trending day sits in
        # premium/discount by definition. Both the ICT entry checklist in the
        # Mobius knowledge base ("discount for longs, premium for shorts") and
        # the SrsBlack library treat it as a required confirmation, so it is
        # available as an opt-in gate. Default False keeps the spec's behaviour.
        'require_pd_zone': False,
        'max_stop_pts': 20.0,           # cap applied after wick/atr/fixed placement
        # ── Target ────────────────────────────────────────────────────────────
        'target_min_rr': 3.0,
        # Target selection. 'liquidity' walks the live opposing pools for the
        # nearest one clearing the R:R floor (the project's rule). 'range_end'
        # implements the Mobius Silver Bullet target: when the swept pool is one
        # side of a measured range (AM hour, opening range), aim for the other
        # side of that same range, falling back to 'liquidity' when the swept
        # pool carries no range.
        # 'liquidity'  nearest live opposing pool if it clears target_min_rr,
        #              otherwise a synthetic target placed at exactly that R.
        # 'range_end'  far side of the swept range, else fall back to liquidity.
        # 'pool_only'  ALWAYS the real pool, never a synthetic R multiple. If the
        #              nearest live pool sits closer than target_min_rr the setup
        #              is skipped rather than re-targeted. Measured on the index
        #              baseline, 'liquidity' put 91% of targets on the synthetic
        #              3R because the real pool was nearer than the floor, and
        #              that synthetic-target subset was a net loser.
        'target_mode': 'liquidity',
        # With 'range_end', take the far side of the swept range even when it is
        # closer than target_min_rr. The Silver Bullet case studies target the
        # opposite end of the 9AM range and the R:R happens to land near 3; the
        # floor should not veto the setup's own defined target.
        'range_end_enforce_rr': True,
        # ── Daily bias ────────────────────────────────────────────────────────
        # four_h_gate: 'soft' — neutral 4H abstains (does not veto daily signal)
        #              'hard' — 4H must actively agree; neutral 4H vetoes
        'four_h_gate': 'soft',
    }

    def __init__(self, config: Optional[dict] = None):
        # Single source of truth for the OR window: the sessions-and-ranges
        # concept frontmatter. Fall back to DEFAULT_CONFIG, then apply caller config.
        defaults = dict(self.DEFAULT_CONFIG)
        or_params = registry_params('sessions-and-ranges')
        if 'opening_range_start' in or_params:
            defaults['opening_range_start'] = str(or_params['opening_range_start'])
        if 'opening_range_minutes' in or_params:
            defaults['opening_range_minutes'] = int(or_params['opening_range_minutes'])
        self.config = {**defaults, **(config or {})}
        self.checks: list = []

    def _log(self, msg: str):
        self.checks.append(msg)

    # ------------------------------------------------------------------
    # Step 1: Daily bias
    # ------------------------------------------------------------------

    def _daily_bias(self, daily_df: pd.DataFrame, df_4h: Optional[pd.DataFrame] = None) -> str:
        components = daily_bias_components(
            daily_df,
            lookback=self.config['daily_lookback'],
            dealing_range_days=self.config['dealing_range_days'],
            swing_lookback=self.config['swing_lookback'],
            df_4h=df_4h,
            four_h_gate=self.config.get('four_h_gate', 'soft'),
        )
        bias = components['bias']
        if bias == 'neutral':
            self._log(
                f"Daily bias: neutral (order_flow={components['order_flow']}, "
                f"draw={components['draw']}, "
                f"4H={components['four_h_structure']}, "
                f"premium_discount={components['premium_discount']})"
            )
        else:
            self._log(
                f"Daily bias: {bias.upper()} "
                f"(order_flow={components['order_flow']}, draw={components['draw']}, "
                f"4H={components['four_h_structure']}, "
                f"prior_day_mid={components['prior_day_midpoint']})"
            )
        return bias

    # ------------------------------------------------------------------
    # Step 2: Session liquidity pool set
    # ------------------------------------------------------------------

    def _build_pools(self, snapshot: Model2022Snapshot, before: pd.Timestamp) -> dict:
        """PDH/PDL + opening range + REH/REL. Returns {highs: [...], lows: [...]}."""
        highs: list = []
        lows:  list = []

        session_open = pd.Timestamp(f"{snapshot.session_date.date()} 00:00",
                                    tz='America/New_York').tz_convert('UTC')

        if self.config['use_pdh_pdl']:
            daily = snapshot.df_daily
            if daily is not None and len(daily) > 0:
                before_daily = localize_like(before, daily.index)
                prior = daily[daily.index < before_daily]
                if len(prior) > 0:
                    pdh = float(prior['high'].iloc[-1])
                    pdl = float(prior['low'].iloc[-1])
                    pdh_ts = prior.index[-1]
                    highs.append({'extreme_price': pdh, 'price': pdh, 'source': 'PDH',
                                  'timestamps': [pdh_ts], 'formed_at': session_open})
                    lows.append({'extreme_price': pdl, 'price': pdl, 'source': 'PDL',
                                 'timestamps': [pdh_ts], 'formed_at': session_open})

        if self.config['use_chicago_prior']:
            prior_date = snapshot.session_date - pd.Timedelta(days=1)
            chicago_prior = chicago_range(snapshot.df_2m, prior_date)
            if chicago_prior is not None:
                highs.append({'extreme_price': chicago_prior['high'], 'price': chicago_prior['high'],
                              'source': 'Chicago-prior-high', 'timestamps': [chicago_prior['end']],
                              'formed_at': session_open})
                lows.append({'extreme_price': chicago_prior['low'], 'price': chicago_prior['low'],
                             'source': 'Chicago-prior-low', 'timestamps': [chicago_prior['end']],
                             'formed_at': session_open})

        if self.config['use_chicago_same_day']:
            chicago_today = chicago_range(snapshot.df_2m, snapshot.session_date)
            if chicago_today is not None:
                highs.append({'extreme_price': chicago_today['high'], 'price': chicago_today['high'],
                              'source': 'Chicago-high', 'timestamps': [chicago_today['end']],
                              'formed_at': chicago_today['end']})
                lows.append({'extreme_price': chicago_today['low'], 'price': chicago_today['low'],
                             'source': 'Chicago-low', 'timestamps': [chicago_today['end']],
                             'formed_at': chicago_today['end']})

        if self.config['use_ons']:
            ons = ons_range(snapshot.df_2m, snapshot.session_date)
            if ons is not None:
                highs.append({'extreme_price': ons['high'], 'price': ons['high'], 'source': 'ONS-high',
                              'timestamps': [ons['end']], 'formed_at': ons['end']})
                lows.append({'extreme_price': ons['low'], 'price': ons['low'], 'source': 'ONS-low',
                             'timestamps': [ons['end']], 'formed_at': ons['end']})

        if self.config['use_asian_range']:
            asia = asian_range(snapshot.df_2m, snapshot.session_date)
            if asia is not None:
                highs.append({'extreme_price': asia['high'], 'price': asia['high'],
                              'source': 'Asia-high', 'timestamps': [asia['end']],
                              'formed_at': asia['end']})
                lows.append({'extreme_price': asia['low'], 'price': asia['low'],
                             'source': 'Asia-low', 'timestamps': [asia['end']],
                             'formed_at': asia['end']})

        if self.config['use_am_hour_range']:
            amr = am_hour_range(snapshot.df_2m, snapshot.session_date,
                                start_ny=self.config['am_hour_start'])
            if amr is not None:
                highs.append({'extreme_price': amr['high'], 'price': amr['high'],
                              'source': 'AMhour-high', 'timestamps': [amr['end']],
                              'formed_at': amr['end'], 'range_low': amr['low']})
                lows.append({'extreme_price': amr['low'], 'price': amr['low'],
                             'source': 'AMhour-low', 'timestamps': [amr['end']],
                             'formed_at': amr['end'], 'range_high': amr['high']})

        if self.config['use_opening_range']:
            orng = opening_range(
                snapshot.df_2m, snapshot.session_date,
                minutes=self.config['opening_range_minutes'],
                start_ny=self.config['opening_range_start'],
            )
            if orng is not None:
                highs.append({'extreme_price': orng['high'], 'price': orng['high'], 'source': 'OR-high',
                              'timestamps': [orng['high_time']], 'formed_at': orng['range_end'],
                              'range_start': orng['range_start'], 'range_end': orng['range_end'],
                              'range_low': orng['low']})
                lows.append({'extreme_price': orng['low'],  'price': orng['low'],  'source': 'OR-low',
                             'timestamps': [orng['low_time']], 'formed_at': orng['range_end'],
                             'range_start': orng['range_start'], 'range_end': orng['range_end'],
                             'range_high': orng['high']})

        if self.config['use_reh_rel']:
            reh_swing = (self.config['reh_rel_swing_lookback']
                         or self.config['swing_lookback'])
            for cl in find_relative_equal_levels(
                snapshot.df_15m, before, 'highs',
                lookback_days=self.config['reh_lookback_days'],
                tolerance_pts=self.config['eq_tolerance_pts'],
                swing_lookback=reh_swing,
            ):
                cl['source'] = f"REH×{cl['count']}"
                cl['formed_at'] = max(cl['timestamps']) if cl.get('timestamps') else None
                highs.append(cl)

            for cl in find_relative_equal_levels(
                snapshot.df_15m, before, 'lows',
                lookback_days=self.config['reh_lookback_days'],
                tolerance_pts=self.config['eq_tolerance_pts'],
                swing_lookback=reh_swing,
            ):
                cl['source'] = f"REL×{cl['count']}"
                cl['formed_at'] = max(cl['timestamps']) if cl.get('timestamps') else None
                lows.append(cl)

        self._log(f"Pools: {len(highs)} buy-side / {len(lows)} sell-side")
        return {'highs': highs, 'lows': lows}

    # ------------------------------------------------------------------
    # Step 3: Sweep (inside a killzone)
    # ------------------------------------------------------------------

    def _kz_minutes_remaining(self, ts: pd.Timestamp) -> Optional[int]:
        """Minutes left in the configured killzone window containing `ts`.

        Windows are 'HH:MM' NY local, so the comparison is done in NY time (DST
        safe). Returns None if `ts` falls in no configured window.
        """
        ny = ts.tz_convert('America/New_York') if ts.tzinfo else ts
        mins = ny.hour * 60 + ny.minute
        for start, end in self.config['killzones']:
            sh, sm = map(int, str(start).split(':'))
            eh, em = map(int, str(end).split(':'))
            s_min, e_min = sh * 60 + sm, eh * 60 + em
            if s_min <= mins < e_min:
                return e_min - mins
        return None

    def _window_edge_ok(self, ts: pd.Timestamp) -> bool:
        """Reject setups firing too close to the end of their killzone."""
        need = self.config['min_kz_minutes_remaining']
        if not need:
            return True
        left = self._kz_minutes_remaining(ts)
        if left is None:
            return True  # not in a window at all - the killzone mask governs that
        if left < need:
            self._log(f"Sweep rejected: only {left} min left in killzone (need {need})")
            return False
        return True

    @staticmethod
    def _sweep_metrics(row, level: float, side: str) -> dict:
        """Shape of the raid bar, measured whether or not the filters are on.

        These are always computed and logged so the sweep-quality thresholds can
        be chosen from the realised distribution instead of guessed: a genuine
        stop hunt spikes through the pool and is rejected (long wick, close back
        inside), while a run-and-continue bar closes beyond the level it took.
        """
        o, h, l, c = (float(row['open']), float(row['high']),
                      float(row['low']), float(row['close']))
        bar_range = h - l
        if side == 'high':
            wick, extreme, closed_inside = h - max(o, c), h, c < level
        else:
            wick, extreme, closed_inside = min(o, c) - l, l, c > level
        return {
            'sweep_pool_level':      level,
            'sweep_extreme':         extreme,
            'sweep_penetration_pts': abs(extreme - level),
            'sweep_bar_range':       bar_range,
            'sweep_wick_pts':        wick,
            'sweep_wick_pct':        (wick / bar_range) if bar_range > 0 else None,
            'sweep_close_inside':    bool(closed_inside),
        }

    def _sweep_quality_ok(self, m: dict) -> bool:
        """Reject run-and-continue bars masquerading as stop hunts.

        Both checks are opt-in via config (see min_sweep_wick_pct and
        require_sweep_close_inside); with the defaults this always passes.
        """
        min_pct = self.config['min_sweep_wick_pct']
        need_inside = self.config['require_sweep_close_inside']
        if m['sweep_bar_range'] <= 0:
            return False
        if not min_pct and not need_inside:
            return True
        if min_pct and m['sweep_wick_pct'] < min_pct:
            self._log(f"Sweep rejected: wick {m['sweep_wick_pct']:.0%} < {min_pct:.0%} of bar range")
            return False
        if need_inside and not m['sweep_close_inside']:
            self._log(f"Sweep rejected: closed beyond the raided level {m['sweep_pool_level']:.5f}")
            return False
        return True

    def _detect_sweep(
        self,
        df_2m: pd.DataFrame,
        kz_mask: pd.Series,
        pools: list,
        direction: str,
    ) -> Optional[dict]:
        """
        Find the earliest valid stop hunt.

        A pool is only a valid sweep target if its liquidity is still resting:
        the FIRST time price trades through it since it formed must land inside a
        killzone. If the level was first taken earlier — outside a killzone or on
        a prior session — the pool is dead (already mitigated) and is skipped.
        See ict.concepts.liquidity_sweep.first_breach.
        """
        side = 'high' if direction == 'bearish' else 'low'
        kz_times = set(df_2m.index[kz_mask.to_numpy()])

        best = None
        for pool in pools:
            lvl = pool['extreme_price']
            ts, row = first_breach(df_2m, lvl, side, after=pool.get('formed_at'))
            if ts is None or ts not in kz_times:
                continue  # never taken, or first taken outside a killzone → dead
            metrics = self._sweep_metrics(row, lvl, side)
            if not self._sweep_quality_ok(metrics):
                continue
            if not self._window_edge_ok(ts):
                continue
            if best is None or ts < best['sweep_time']:
                stop_level = float(row['high'] if side == 'high' else row['low'])
                best = {'sweep_time': ts, 'cluster': pool, 'stop_level': stop_level,
                        'metrics': metrics}

        if best is not None:
            c = best['cluster']
            self._log(
                f"Stop hunt: swept {c.get('source', side)} at "
                f"{c['extreme_price']:.2f} @ {best['sweep_time']} "
                f"(first raid since formation)"
            )
        return best

    # ------------------------------------------------------------------
    # Target pool validity filter
    # ------------------------------------------------------------------

    def _nearest_pool_target(self, pools: list, direction: str,
                             entry: float, stop: float) -> Optional[float]:
        """Nearest live opposing pool, or None when none sits far enough.

        Unlike the 'liquidity' mode this never substitutes a synthetic R
        multiple: if the market is not offering a real draw at or beyond
        target_min_rr, there is no target and the setup is not taken.
        """
        risk = abs(stop - entry)
        if risk <= 0:
            return None
        if direction == 'short':
            cands = [float(p['extreme_price']) for p in pools
                     if float(p['extreme_price']) < entry]
            nearest = max(cands) if cands else None
        else:
            cands = [float(p['extreme_price']) for p in pools
                     if float(p['extreme_price']) > entry]
            nearest = min(cands) if cands else None
        if nearest is None:
            return None
        if abs(entry - nearest) / risk < self.config['target_min_rr']:
            return None
        self._log(f"pool_only target {nearest:.5f} "
                  f"({abs(entry - nearest) / risk:.2f}R)")
        return nearest

    def _target_diag(self, pools: list, direction: str, entry: float,
                     stop: float, target) -> dict:
        """How far real liquidity actually was, and whether we used it.

        `synthetic_target_used` is the honest test of the 'liquidity' mode: it is
        True whenever the chosen target does not coincide with any live opposing
        pool, i.e. the R:R floor manufactured the level instead of the market
        offering it.
        """
        risk = abs(stop - entry)
        if direction == 'short':
            cands = [float(p['extreme_price']) for p in pools
                     if float(p['extreme_price']) < entry]
            nearest = max(cands) if cands else None
        else:
            cands = [float(p['extreme_price']) for p in pools
                     if float(p['extreme_price']) > entry]
            nearest = min(cands) if cands else None
        nearest_rr = (abs(entry - nearest) / risk
                      if nearest is not None and risk > 0 else None)
        synthetic = True
        if target is not None:
            tol = max(risk * 1e-6, 1e-9)
            synthetic = not any(abs(float(p['extreme_price']) - target) <= tol
                                for p in pools)
        return {
            'n_live_target_pools':   len(pools),
            'nearest_pool_price':    nearest,
            'nearest_pool_rr':       nearest_rr,
            'synthetic_target_used': synthetic,
            'real_target_rr':        None if synthetic else nearest_rr,
        }

    def _range_end_target(self, pool: dict, direction: str,
                          entry: float, stop: float) -> Optional[float]:
        """Opposite side of the measured range whose edge was swept.

        Mobius defines the Silver Bullet target as price returning to the far end
        of the 9:00 AM range after the near end is raided. Pools that carry a
        range (AM hour, opening range) record the other side as `range_low` /
        `range_high`; pools without one (PDH/PDL, REH/REL) return None so the
        caller falls back to the liquidity target.
        """
        far = pool.get('range_low') if direction == 'short' else pool.get('range_high')
        if far is None:
            return None
        far = float(far)
        # Must sit the right side of entry, and still clear the R:R floor.
        if direction == 'short' and far >= entry:
            return None
        if direction == 'long' and far <= entry:
            return None
        risk = abs(stop - entry)
        if self.config['range_end_enforce_rr']:
            if risk <= 0 or abs(far - entry) / risk < self.config['target_min_rr']:
                self._log(f"range_end target {far:.5f} below R:R floor - falling back to liquidity")
                return None
        elif risk <= 0:
            return None
        self._log(f"range_end target from {pool.get('source')}: {far:.5f}")
        return far

    def _live_target_pools(
        self,
        df_2m: pd.DataFrame,
        pools: list,
        direction: str,
        as_of: pd.Timestamp,
    ) -> list:
        """
        Return only pools that have NOT yet been run as of as_of.

        A target pool is dead once price has already traded through it — there's
        no resting liquidity left to draw toward. The sweep side for a target is
        the opposite of entry: short targets are sell-side (price must trade DOWN
        through them), long targets are buy-side (price must trade UP through them).
        """
        side = 'low' if direction == 'short' else 'high'
        as_of = localize_like(as_of, df_2m.index)
        live = []
        for pool in pools:
            lvl = pool['extreme_price']
            formed_at = pool.get('formed_at')
            ts, _ = first_breach(df_2m, lvl, side, after=formed_at)
            if ts is None or ts > as_of:
                live.append(pool)
        if len(live) < len(pools):
            self._log(f"Target pools: {len(pools) - len(live)} already run, {len(live)} live")
        return live

    # ------------------------------------------------------------------
    # Step 4: MSS with displacement on 2m
    # ------------------------------------------------------------------

    def _detect_bms(
        self,
        df_2m: pd.DataFrame,
        direction: str,
        after_time: pd.Timestamp,
    ) -> Optional[dict]:
        """Thin wrapper around detect_mss; adds df_after slice for FVG search."""
        after_time = localize_like(after_time, df_2m.index)
        result = detect_mss(
            df_2m,
            direction=direction,
            swing_lookback=(self.config['mss_swing_lookback']
                            or self.config['swing_lookback']),
            max_bars_after_sweep=self.config['max_bars_sweep_to_mss'],
            sweep_time=after_time,
        )
        if result is None:
            return None

        cutoff = df_2m.index.get_indexer([after_time], method='nearest')[0]
        end     = min(cutoff + self.config['max_bars_sweep_to_mss'] + 1, len(df_2m))
        df_after = df_2m.iloc[cutoff:end]
        bms_pos  = int(df_after.index.get_indexer([result['break_time']], method='nearest')[0])

        metrics = self._mss_metrics(df_after, bms_pos, direction,
                                    result['broken_level'])
        if not self._displacement_ok(df_after, bms_pos, direction):
            return None

        self._log(
            f"MSS: closed {'below' if direction == 'bearish' else 'above'} "
            f"swing at {result['broken_level']:.2f} @ {result['break_time']}"
        )
        return {
            'bms_time':       result['break_time'],
            'bms_swing_level': result['broken_level'],
            'bms_pos':        bms_pos,
            'df_after':       df_after,
            'metrics':        metrics,
        }

    @staticmethod
    def _mss_metrics(df: pd.DataFrame, pos: int, direction: str,
                     broken_level) -> dict:
        """Shape of the MSS break candle, measured whether or not the
        displacement filter is enforced.

        `mss_size_pts` is how far the break close travelled past the swing it
        broke - the displacement itself, not the distance to the entry. The
        body/wick ratios are the same quantities `_displacement_ok` tests, so
        the thresholds can be read off the realised distribution.
        """
        if pos < 0 or pos >= len(df):
            return {}
        c = df.iloc[pos]
        o, h, l, cl = (float(c['open']), float(c['high']),
                       float(c['low']), float(c['close']))
        rng, body = h - l, abs(cl - o)
        prior = df.iloc[max(0, pos - 14):pos]
        avg_body = float((prior['close'] - prior['open']).abs().mean() or 0.0)
        opp = (h - max(o, cl)) if direction == 'bullish' else (min(o, cl) - l)
        return {
            'mss_size_pts':   (abs(cl - float(broken_level))
                               if broken_level is not None else None),
            'mss_body_pts':   body,
            'mss_range_pts':  rng,
            'mss_body_pct':   (body / rng) if rng > 0 else None,
            'mss_body_mult':  (body / avg_body) if avg_body > 0 else None,
            'mss_opposing_wick_pct': (opp / rng) if rng > 0 else None,
            'mss_to_sweep_bars': pos,
        }

    # ------------------------------------------------------------------
    # Step 5: FVG from the displacement candle
    # ------------------------------------------------------------------

    def _fvg_big_enough(self, f, df: pd.DataFrame, pos: int) -> bool:
        """Drop FVGs that are noise-sized relative to current volatility."""
        mult = self.config['min_fvg_atr_mult']
        if not mult:
            return True
        atr = self._atr(df.iloc[:max(1, pos)])
        if atr <= 0:
            return True
        height = abs(f.top - f.bottom)
        if height < mult * atr:
            self._log(f"FVG rejected: height {height:.5f} < {mult} x ATR {atr:.5f}")
            return False
        return True

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        """Simple ATR over the tail of df; 0.0 when there is not enough data."""
        if len(df) < 2:
            return 0.0
        w = df.tail(period + 1)
        pc = w['close'].shift(1)
        tr = pd.concat([w['high'] - w['low'],
                        (w['high'] - pc).abs(),
                        (w['low'] - pc).abs()], axis=1).max(axis=1)
        return float(tr.dropna().mean() or 0.0)

    def _displacement_ok(self, df: pd.DataFrame, pos: int, direction: str) -> bool:
        """Body-dominance / opposing-wick test on the MSS break candle.

        A wide candle in a choppy session is not displacement. Thresholds follow
        the library's displacement-definition formula block.
        """
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
            self._log(f"Displacement rejected: body {body / rng:.0%} of range "
                      f"< {self.config['displacement_body_pct']:.0%}")
            return False
        opp = (h - max(o, cl)) if direction == 'bullish' else (min(o, cl) - l)
        if opp / rng > self.config['displacement_opp_wick_pct']:
            self._log(f"Displacement rejected: opposing wick {opp / rng:.0%} of range "
                      f"> {self.config['displacement_opp_wick_pct']:.0%}")
            return False
        return True

    def _post_sweep_window(self, df_2m: pd.DataFrame,
                           sweep_time: pd.Timestamp) -> Optional[dict]:
        """A bms-shaped window anchored on the sweep, for require_mss=False.

        `_find_entry_fvg` slices around `bms_pos` inside `df_after`; with no MSS
        to anchor on, the sweep bar itself becomes position 0 and the arrays are
        searched forward from there.
        """
        ts = localize_like(sweep_time, df_2m.index)
        pos = int(df_2m.index.get_indexer([ts], method='nearest')[0])
        end = min(pos + self.config['max_bars_sweep_to_mss'] + 1, len(df_2m))
        if end - pos < 3:
            return None
        # Include bars before the sweep: a breaker or order block is built from
        # swing structure that formed *before* the raid, so slicing from the
        # sweep bar alone leaves those detectors nothing to work with.
        start = max(0, pos - self.config['array_lookback_bars'])
        return {'df_after': df_2m.iloc[start:end], 'bms_pos': pos - start,
                'break_time': df_2m.index[pos], 'broken_level': None}

    def _find_entry_fvg(self, bms_result: dict, direction: str) -> Optional[FVG]:
        """First entry array found at the MSS displacement, in configured order.

        OB and breaker results are normalised into the FVG shape (top/bottom/
        entry) so stop placement, targeting and the trade log stay untouched;
        `fvg_entry_depth` then picks the near edge or the zone midpoint (the
        mean threshold) uniformly across all three array types.
        """
        df_after = bms_result['df_after']
        pos      = bms_result['bms_pos']
        end      = min(len(df_after), pos + 3)
        atr      = self._atr(df_after.iloc[:max(1, pos)])
        self._array_diag = {}

        def _diag(kind, f):
            size = abs(f.top - f.bottom)
            self._array_diag = {
                'entry_array':   kind,
                'fvg_size_pts':  size,
                'fvg_size_atr':  (size / atr) if atr > 0 else None,
            }
            return f

        for kind in self.config['entry_arrays']:
            if kind == 'fvg':
                search = df_after.iloc[max(0, pos - 2): end]
                if len(search) < 3:
                    continue
                found = find_fvgs(search, direction=direction)
                found = [g for g in found if self._fvg_big_enough(g, df_after, pos)]
                if found:
                    f = self._apply_entry_depth(found[0])
                    self._log(f"FVG: {direction} gap {f.bottom:.5f}-{f.top:.5f} "
                              f"@ {f.timestamp}, entry {f.entry:.5f}")
                    return _diag('fvg', f)
                continue

            wide = df_after.iloc[max(0, pos - self.config['array_lookback_bars']): end]
            if len(wide) < 5:
                continue

            if kind == 'ob':
                zones = find_order_blocks(wide, direction=direction, validated_only=False)
            elif kind == 'breaker':
                zones = find_breaker_blocks(wide, direction=direction)
            else:
                raise ValueError(f"Unknown entry array: {kind!r} (expected 'fvg', 'ob' or 'breaker')")
            if not zones:
                continue

            z = zones[-1]   # most recent zone before/at the displacement
            f = self._apply_entry_depth(FVG(
                timestamp=z.timestamp, direction=direction,
                top=float(z.zone_top), bottom=float(z.zone_bot),
                entry=float(z.proximal),
            ))
            self._log(f"{kind.upper()}: {direction} zone {f.bottom:.5f}-{f.top:.5f} "
                      f"@ {f.timestamp}, entry {f.entry:.5f}")
            return _diag(kind, f)

        return None

    def _apply_entry_depth(self, f: FVG) -> FVG:
        """Move the limit entry to Consequent Encroachment when configured.

        find_fvgs returns the near edge as `entry` - the first price to touch the
        gap. ICT's 2025 default is CE, the gap's 50% midpoint: near-edge fills get
        stopped out when price rebalances the gap fully, while far-edge limits
        often never fill. See the ict-knowledge-library concept
        06-fair-value-gaps/ce-as-primary-entry.
        """
        depth = self.config['fvg_entry_depth']
        if depth == 'edge':
            return f
        if depth != 'ce':
            raise ValueError(f"Unknown fvg_entry_depth: {depth!r} (expected 'edge' or 'ce')")
        return dataclasses.replace(f, entry=(f.top + f.bottom) / 2.0)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        snapshot: Model2022Snapshot,
        force_direction: Optional[str] = None,
    ) -> dict:
        """Evaluate one session for a valid 2022-model setup.

        Args:
            force_direction: 'bearish' or 'bullish' — skips the daily bias gate
                and evaluates only that side. Pass None to use daily bias normally.
        """
        self.checks = []
        self._array_diag = {}
        base = {
            'actionable': False, 'direction': None,
            'entry': None, 'entry_time': None,
            'stop': None, 'target': None,
            'fvg_top': None, 'fvg_bottom': None,
            'sweep_level': None, 'session_midpoint': None,
            'checks': self.checks,
        }

        # 1. Daily bias (skipped when force_direction is provided)
        if force_direction is not None:
            bias = force_direction
            self._log(f"Daily bias: FORCED {bias.upper()} (gate bypassed)")
        else:
            bias = self._daily_bias(snapshot.df_daily, df_4h=snapshot.df_4h)
            if bias == 'neutral':
                self._log("Rule of exclusion: no daily bias → no trade")
                return base

        direction = 'short' if bias == 'bearish' else 'long'

        # 1b. Premium/discount entry zone (soft check — log misalignment)
        pd_signal = premium_discount(snapshot.df_daily, self.config['dealing_range_days'])
        pd_aligned = (bias == 'bearish' and pd_signal == 'bearish') or \
                     (bias == 'bullish' and pd_signal == 'bullish')
        self._log(f"P/D zone: {pd_signal} ({'aligned' if pd_aligned else 'misaligned with bias'})")
        if self.config['require_pd_zone'] and not pd_aligned:
            self._log("Rule of exclusion: P/D zone misaligned with bias")
            return base

        # 2. Pool set
        before = pd.Timestamp(snapshot.session_date.date())
        pools  = self._build_pools(snapshot, before)
        entry_pools  = pools['highs'] if bias == 'bearish' else pools['lows']
        target_pools = pools['lows']  if bias == 'bearish' else pools['highs']
        if not entry_pools:
            self._log(f"Rule of exclusion: no {'buy' if bias == 'bearish' else 'sell'}-side pools")
            return base

        # 3. Killzone filter + sweep
        kz_mask = session_mask(snapshot.df_2m, snapshot.session_date, self.config['killzones'])
        if int(kz_mask.sum()) < 5:
            self._log("Rule of exclusion: insufficient killzone data")
            return base
        sweep = self._detect_sweep(snapshot.df_2m, kz_mask, entry_pools, bias)
        if sweep is None:
            self._log("Rule of exclusion: no stop hunt in killzone")
            return base

        # 4. MSS
        bms = self._detect_bms(snapshot.df_2m, bias, sweep['sweep_time'])
        if bms is None:
            if self.config['require_mss']:
                self._log("Rule of exclusion: no MSS after stop hunt")
                return base
            bms = self._post_sweep_window(snapshot.df_2m, sweep['sweep_time'])
            if bms is None:
                self._log("Rule of exclusion: no bars after the sweep")
                return base
            self._log("MSS not required - searching entry arrays from the sweep bar")

        # 5. FVG
        fvg = self._find_entry_fvg(bms, bias)
        if fvg is None:
            self._log("Rule of exclusion: no FVG at MSS displacement")
            return base

        # 6. Stop placement — mode-dependent, then capped by max_stop_pts.
        stop_mode = self.config['stop_mode']
        if stop_mode == 'wick':
            stop = sweep['stop_level']
        elif stop_mode == 'atr':
            period = self.config['stop_atr_period']
            mult   = self.config['stop_atr_mult']
            idx    = snapshot.df_2m.index.searchsorted(fvg.timestamp)
            win    = snapshot.df_2m.iloc[max(0, idx - period): idx]
            if len(win) >= 2:
                tr = pd.concat([
                    win['high'] - win['low'],
                    (win['high'] - win['close'].shift()).abs(),
                    (win['low']  - win['close'].shift()).abs(),
                ], axis=1).max(axis=1)
                atr = float(tr.mean())
            else:
                atr = self.config['max_stop_pts']
            stop = fvg.entry + atr * mult if direction == 'short' else fvg.entry - atr * mult
            self._log(f"ATR stop: ATR={atr:.2f} × {mult} → stop={stop:.2f}")
        elif stop_mode == 'fixed':
            pts  = self.config['stop_fixed_pts']
            stop = fvg.entry + pts if direction == 'short' else fvg.entry - pts
            self._log(f"Fixed stop: ±{pts} pts → stop={stop:.2f}")
        else:
            raise ValueError(f"Unknown stop_mode: {stop_mode!r}  (expected 'wick', 'atr', 'fixed')")

        if direction == 'short' and stop <= fvg.entry:
            self._log(f"Rule of exclusion: stop {stop:.2f} not above entry {fvg.entry:.2f}")
            return base
        if direction == 'long' and stop >= fvg.entry:
            self._log(f"Rule of exclusion: stop {stop:.2f} not below entry {fvg.entry:.2f}")
            return base

        buffer = self.config['stop_buffer_pts']
        if buffer:
            stop = stop + buffer if direction == 'short' else stop - buffer
            self._log(f"Stop buffer {buffer} applied -> stop={stop:.5f}")

        max_stop = self.config['max_stop_pts']
        if abs(stop - fvg.entry) > max_stop:
            stop = fvg.entry + max_stop if direction == 'short' else fvg.entry - max_stop
            self._log(f"Stop capped at {max_stop} pts: {stop:.2f}")

        # 7. Target — filter to pools not yet run as of entry time, then find
        #    the nearest opposing pool meeting the R:R floor.
        live_targets = self._live_target_pools(
            snapshot.df_2m, target_pools, direction, fvg.timestamp,
        )
        target = None
        if self.config['target_mode'] == 'pool_only':
            target = self._nearest_pool_target(live_targets, direction, fvg.entry, stop)
            if target is None:
                self._log("Rule of exclusion: no live opposing pool at or beyond "
                          f"{self.config['target_min_rr']}R")
                return base
        elif self.config['target_mode'] == 'range_end':
            target = self._range_end_target(sweep['cluster'], direction, fvg.entry, stop)
        if target is None:
            target = find_liquidity_target(
                direction, entry=fvg.entry, stop=stop,
                target_pools=live_targets, min_rr=self.config['target_min_rr'],
            )
        rr = abs(target - fvg.entry) / abs(stop - fvg.entry) if stop != fvg.entry else 0
        tdiag = self._target_diag(live_targets, direction, fvg.entry, stop, target)
        self._log(f"Target: {target:.2f}  R:R={rr:.2f}"
                  + ("  [SYNTHETIC]" if tdiag['synthetic_target_used'] else ""))

        # Session midpoint (chart context) — over the killzone bars up to the sweep
        kz_bars = snapshot.df_2m[kz_mask.to_numpy()]
        pre_sweep = kz_bars[kz_bars.index <= sweep['sweep_time']]
        ref = pre_sweep if len(pre_sweep) > 0 else kz_bars
        midpoint = (float(ref['high'].max()) + float(ref['low'].min())) / 2

        self._log(f"Signal: {direction.upper()} entry={fvg.entry:.2f} stop={stop:.2f} target={target:.2f}")

        return {
            'actionable':      True,
            'direction':       direction,
            # Diagnostics. Every one is measured regardless of which filters are
            # switched on, so a threshold can be chosen from the realised
            # distribution instead of being guessed and then justified.
            **sweep.get('metrics', {}),
            **bms.get('metrics', {}),
            **self._array_diag,
            **tdiag,
            'planned_rr':      rr,
            'entry':           fvg.entry,
            'entry_time':      fvg.timestamp,
            'stop':            stop,
            'target':          target,
            'fvg_top':         fvg.top,
            'fvg_bottom':      fvg.bottom,
            'sweep_level':     sweep['stop_level'],
            'sweep_time':      sweep['sweep_time'],
            'swept_pool':      sweep['cluster'].get('source'),
            'swept_cluster':   sweep['cluster'],
            'bms_time':        bms['bms_time'],
            'bms_swing_level': bms['bms_swing_level'],
            'session_midpoint': midpoint,
            'checks':          self.checks,
        }
