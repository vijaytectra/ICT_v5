"""
Execution Cost Model & Trade Simulator
======================================

Backtests in this repo priced fills at the **mid**: a stop filled exactly at the
stop, a limit filled the moment price touched it, and nothing was paid to cross
the book. That is not a small optimism on intraday FX — the median EURUSD spread
in this dataset is 0.30 pips and a typical Judas stop is 6-12 pips, so the
round-turn spread alone is 5-10% of risk before slippage or commission.

This module makes the cost explicit and configurable, and it uses the
**historical per-bar spread already stored in the cleaned CSVs** rather than a
made-up constant. Nothing here fabricates a price: with `CostModel()` (all
defaults zero, `spread_mode='none'`, `gap_fill='stop'`) the simulator reproduces
the previous mid-price behaviour exactly, so old and new baselines are
comparable.

Directional convention
----------------------
The dataset stores one `spread` value per bar, not separate bid/ask series, so
the executable prices are derived deterministically as ``mid ± spread/2``:

    LONG   entry pays the ask  = mid + half_spread + slippage
           exit  gets the bid  = mid - half_spread - slippage
    SHORT  entry gets the bid  = mid - half_spread - slippage
           exit  pays the ask  = mid + half_spread + slippage

Slippage is always adverse, never favourable.

What this model does NOT claim
------------------------------
- **Limit fills assume zero queue latency.** A limit is filled the instant the
  bar's range touches it. Real resting orders sit in a queue and a touch that
  does not trade through you does not fill you. This is optimistic and is not
  corrected here; it is recorded so the optimism is visible.
- **Stop triggers are evaluated on the mid**, unchanged from the legacy engine.
  Only the *fill price* is adjusted. A bid/ask-triggered stop would fire slightly
  earlier for longs.
- Spread is the bar's mean of the source 1-minute close-to-close spreads, not a
  tick-level path. Intrabar spread widening around news is therefore understated.

.. note::
   ``models/ict/model_2022/backtests/plot_trades.py`` carries its own
   ``CostModel`` with the same directional convention, written in parallel for
   the 2022 model. The two are independent on purpose while both are in flight;
   they should be consolidated onto this module once neither is being edited.
   Only this one is used by the Judas model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence

import pandas as pd


# Exit reasons, as the enum the trade log and statistics agree on.
TP = 'TP'
SL = 'SL'
EOD_PROFIT = 'EOD_PROFIT'
EOD_LOSS = 'EOD_LOSS'
BREAKEVEN = 'BREAKEVEN'
OTHER = 'OTHER'
EXIT_REASONS = (TP, SL, EOD_PROFIT, EOD_LOSS, BREAKEVEN, OTHER)


@dataclass(frozen=True)
class CostModel:
    """Execution costs for one instrument.

    All price-unit fields are in the instrument's own quote units (0.0001 = 1 pip
    on a 5-decimal FX pair, 1.0 = 1 index point), so they scale the same way the
    model's other ``_pts`` parameters do.

    Attributes:
        spread_mode:      'none'       - no spread (legacy mid-price behaviour)
                          'historical' - the per-bar `spread` column from the
                                         cleaned CSV; bars with a missing or
                                         negative value contribute zero
                          'fixed'      - `fixed_spread` on every bar
        fixed_spread:     Spread in price units, used when spread_mode='fixed'.
        spread_multiplier: Scales the applied spread. 1.0 = as recorded. Used for
                          cost-stress runs, not for tuning.
        slippage_entry:   Adverse price units added to every entry fill.
        slippage_exit:    Adverse price units on target / EOD / partial exits.
        slippage_stop:    Adverse price units on stop exits. Separate because
                          stops are marketable and typically slip more.
        commission_round_turn: USD per round turn per unit traded. Default 0.0
                          preserves the historical baseline; when it is zero the
                          statistics say so explicitly rather than silently
                          omitting the line.
        gap_fill:         'stop' - a bar that opens beyond the stop still fills
                                   at the stop price (legacy behaviour)
                          'open' - it fills at the bar's open, which is the first
                                   price actually available. Realistic, and
                                   strictly worse; kept off by default so the
                                   zero-cost reproduction is exact.
    """
    spread_mode: str = 'none'
    fixed_spread: float = 0.0
    spread_multiplier: float = 1.0
    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    slippage_stop: float = 0.0
    commission_round_turn: float = 0.0
    gap_fill: str = 'stop'

    def __post_init__(self):
        if self.spread_mode not in ('none', 'historical', 'fixed'):
            raise ValueError(f"spread_mode must be 'none', 'historical' or "
                             f"'fixed', got {self.spread_mode!r}")
        if self.gap_fill not in ('stop', 'open'):
            raise ValueError(f"gap_fill must be 'stop' or 'open', got {self.gap_fill!r}")
        for name in ('fixed_spread', 'slippage_entry', 'slippage_exit',
                     'slippage_stop', 'commission_round_turn'):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 (costs are always adverse)")
        if self.spread_multiplier < 0:
            raise ValueError('spread_multiplier must be >= 0')

    @property
    def is_zero(self) -> bool:
        """True when this model charges nothing and fills like the legacy engine."""
        return (self.spread_mode == 'none'
                and self.slippage_entry == 0.0
                and self.slippage_exit == 0.0
                and self.slippage_stop == 0.0
                and self.commission_round_turn == 0.0
                and self.gap_fill == 'stop')

    def describe(self) -> str:
        if self.is_zero:
            return ('zero-cost (mid prices, no spread, no slippage, '
                    'commission = 0, legacy stop fills)')
        bits = [f"spread={self.spread_mode}"]
        if self.spread_mode == 'fixed':
            bits.append(f"fixed={self.fixed_spread}")
        if self.spread_multiplier != 1.0:
            bits.append(f"x{self.spread_multiplier}")
        if self.slippage_entry or self.slippage_exit or self.slippage_stop:
            bits.append(f"slip(entry={self.slippage_entry},exit={self.slippage_exit},"
                        f"stop={self.slippage_stop})")
        bits.append(f"commission={self.commission_round_turn}"
                    + (' (ZERO)' if self.commission_round_turn == 0 else ''))
        bits.append(f"gap_fill={self.gap_fill}")
        return ', '.join(bits)

    def as_dict(self) -> dict:
        return asdict(self)


ZERO_COST = CostModel()


def half_spread(row, cost: CostModel) -> float:
    """Half the bar's spread, in price units. Never negative, never NaN.

    The cleaned CSVs carry a handful of crossed quotes (155 bars on EURUSD out of
    1.0M) where ask < bid and the recorded spread is negative. Those are data
    artefacts, not free money, so they are clamped to zero rather than credited.
    """
    if cost.spread_mode == 'none':
        return 0.0
    if cost.spread_mode == 'fixed':
        raw = cost.fixed_spread
    else:
        try:
            raw = float(row['spread'])
        except (KeyError, TypeError, ValueError):
            return 0.0
    if raw is None or not math.isfinite(raw) or raw <= 0:
        return 0.0
    return 0.5 * raw * cost.spread_multiplier


def _sign(direction: str) -> float:
    return 1.0 if direction == 'long' else -1.0


def entry_fill_price(mid: float, direction: str, row, cost: CostModel) -> tuple:
    """Executable entry price and the cost components paid to get it."""
    hs = half_spread(row, cost)
    s = _sign(direction)
    return mid + s * (hs + cost.slippage_entry), hs, cost.slippage_entry


def exit_fill_price(mid: float, direction: str, row, cost: CostModel,
                    is_stop: bool = False) -> tuple:
    """Executable exit price and the cost components paid to get it."""
    hs = half_spread(row, cost)
    slip = cost.slippage_stop if is_stop else cost.slippage_exit
    s = _sign(direction)
    return mid - s * (hs + slip), hs, slip


def _classify(reason: str, pnl: float, stop_moved_to_be: bool) -> str:
    """Map an internal exit label onto the reported exit-reason enum."""
    if reason == 'target':
        return TP
    if reason == 'partial':
        # Every partial leg filled at its R multiple, so the position closed on
        # profit targets rather than on a stop.
        return TP
    if reason == 'breakeven':
        return BREAKEVEN
    if reason == 'stop':
        if stop_moved_to_be and abs(pnl) < 1e-12:
            return BREAKEVEN
        return SL
    if reason == 'eod':
        if pnl > 0:
            return EOD_PROFIT
        if pnl < 0:
            return EOD_LOSS
        return BREAKEVEN
    return OTHER


def simulate_entry(
    df: pd.DataFrame,
    signal: dict,
    deadline: Optional[pd.Timestamp],
    cost: CostModel = ZERO_COST,
) -> Optional[dict]:
    """Fill the signal's entry, honouring its style, and charge entry costs.

    ``entry_style='market'`` fills at the OPEN of the bar after the confirmation
    candle — the first price actually available once that candle closed.

    ``entry_style='limit'`` (or an unset style) fills the first time the bar range
    touches the limit price, up to `deadline`. **Touch-fill assumes zero queue
    latency**: a real resting limit only fills if price trades through the level
    with enough size ahead of it. This is optimistic and is deliberately left
    uncorrected so the optimism stays visible rather than being buried in a
    fudge factor.

    Returns None when no fill occurred.
    """
    direction = signal['direction']
    entry_mid = float(signal['entry'])
    after = signal['entry_time']
    if df.index.tz is not None and after.tzinfo is None:
        after = after.tz_localize(df.index.tz)
    elif df.index.tz is None and after.tzinfo is not None:
        after = after.tz_localize(None)

    window = df[df.index > after]
    if deadline is not None:
        if df.index.tz is not None and deadline.tzinfo is None:
            deadline = deadline.tz_localize(df.index.tz)
        elif df.index.tz is None and deadline.tzinfo is not None:
            deadline = deadline.tz_localize(None)
        window = window[window.index <= deadline]
    if len(window) == 0:
        return None

    style = signal.get('entry_style')
    if style == 'market':
        ts, row = window.index[0], window.iloc[0]
        mid = float(row['open'])
    else:
        mid = entry_mid
        if style == 'stop':
            # A stop entry is triggered by price moving THROUGH the level in
            # the trade's own direction - the opposite test to a limit. Used by
            # the buy-stop-above-the-down-close-candle entry, where a limit
            # test would fill on the wrong side of the market entirely.
            hit = (window['high'] >= mid) if direction == 'long' else (window['low'] <= mid)
        else:
            hit = (window['low'] <= mid) if direction == 'long' else (window['high'] >= mid)
        idx = hit.to_numpy().argmax() if bool(hit.any()) else None
        if idx is None:
            return None
        ts, row = window.index[idx], window.iloc[idx]

    price, hs, slip = entry_fill_price(mid, direction, row, cost)
    return {'fill_time': ts, 'fill_mid': mid, 'fill_price': price,
            'entry_spread_cost': hs, 'entry_slippage_cost': slip,
            # The exit scan starts on the NEXT bar, so it needs to know what the
            # entry bar itself did - see the gap test in simulate_exit.
            'fill_bar_high': float(row['high']), 'fill_bar_low': float(row['low'])}


def simulate_exit(
    df: pd.DataFrame,
    signal: dict,
    fill: dict,
    eod: pd.Timestamp,
    cost: CostModel = ZERO_COST,
    be_trigger_r: float = 0.0,
    partials: Optional[Sequence] = None,
    partial_be: bool = True,
) -> dict:
    """Walk bars from the fill to `eod`, resolving target / stop / partials.

    Trigger logic is unchanged from the legacy engine — target and stop are both
    tested on the bar's mid high/low, and both are resolved *before* the partial
    legs, so a bar that reaches a leg and also takes out the stop is scored
    against us. Only the resulting fill *prices* carry costs.

    Gap handling: with ``gap_fill='open'`` a bar that opens already beyond the
    stop fills at that open rather than at the stop, because the stop price was
    never available. With the default ``'stop'`` the legacy assumption is kept so
    the zero-cost reproduction is exact.
    """
    direction = signal['direction']
    target = float(signal['target'])
    stop = float(signal['stop'])
    entry_exec = float(fill['fill_price'])
    entry_mid = float(fill['fill_mid'])

    fill_time = fill['fill_time']
    if df.index.tz is not None and fill_time.tzinfo is None:
        fill_time = fill_time.tz_localize(df.index.tz)
    if df.index.tz is not None and eod.tzinfo is None:
        eod = eod.tz_localize(df.index.tz)
    elif df.index.tz is None and eod.tzinfo is not None:
        eod = eod.tz_localize(None)

    window = df[(df.index > fill_time) & (df.index <= eod)]

    s = _sign(direction)
    # Range of the last bar the position was open through but which the exit scan
    # does not visit (the fill bar). Used to tell a real gap from a level that was
    # simply reachable earlier - see `_gapped` below.
    prev_high = float(fill.get('fill_bar_high', entry_mid))
    prev_low = float(fill.get('fill_bar_low', entry_mid))

    def _gapped(op: float, level: float) -> bool:
        """True only when the stop was genuinely jumped, not merely missed.

        `gap_fill='open'` exists for the case where price was never available at
        the stop. It must not fire when the previous bar already traded through
        the level: there the stop was reachable and the honest fill is the stop
        itself. Without this test a limit that fills on the same bar as a violent
        move gets 'stopped' at the *next* bar's open - which on an NFP candle
        manufactured a -20R loss out of a -1R trade.
        """
        if cost.gap_fill != 'open':
            return False
        if direction == 'short':
            return op > level and prev_high < level
        return op < level and prev_low > level

    risk = abs(stop - float(signal['entry']))
    be_distance = be_trigger_r * risk if (be_trigger_r and risk > 0) else None
    be_armed = False

    legs = sorted(partials or [], key=lambda x: x[1])
    pending = [(float(f), entry_mid + s * float(r) * risk) for f, r in legs] if risk > 0 else []
    realized = 0.0        # fraction-weighted executed P&L, price units
    spread_cost = float(fill.get('entry_spread_cost', 0.0))
    slip_cost = float(fill.get('entry_slippage_cost', 0.0))
    remaining = 1.0
    mae = mfe = 0.0

    def finish(ts, mid, reason, row, n_bars, is_stop=False):
        nonlocal spread_cost, slip_cost
        px, hs, slip = exit_fill_price(mid, direction, row, cost, is_stop=is_stop)
        spread_cost += hs * remaining
        slip_cost += slip * remaining
        pnl = realized + remaining * s * (px - entry_exec)
        return {
            'exit_time': ts, 'exit_price': px, 'exit_mid': mid,
            'exit_reason': _classify(reason, pnl, be_armed),
            'raw_exit_reason': reason,
            'pnl_pts': pnl,
            'spread_cost_pts': spread_cost,
            'slippage_cost_pts': slip_cost,
            'mae_pts': mae, 'mfe_pts': mfe, 'bars_held': n_bars,
            'partials_filled': len(legs) - len(pending),
        }

    for n, (ts, row) in enumerate(window.iterrows(), 1):
        hi, lo, op = float(row['high']), float(row['low']), float(row['open'])
        if direction == 'short':
            mae = max(mae, hi - entry_mid)
            mfe = max(mfe, entry_mid - lo)
        else:
            mae = max(mae, entry_mid - lo)
            mfe = max(mfe, hi - entry_mid)

        if direction == 'short':
            if lo <= target:
                return finish(ts, target, 'target', row, n)
            if hi >= stop:
                mid = op if _gapped(op, stop) else stop
                return finish(ts, mid, 'breakeven' if be_armed else 'stop', row, n,
                              is_stop=True)
        else:
            if hi >= target:
                return finish(ts, target, 'target', row, n)
            if lo <= stop:
                mid = op if _gapped(op, stop) else stop
                return finish(ts, mid, 'breakeven' if be_armed else 'stop', row, n,
                              is_stop=True)

        while pending:
            frac, level = pending[0]
            hit = lo <= level if direction == 'short' else hi >= level
            if not hit:
                break
            pending.pop(0)
            take = min(frac, remaining)
            px, hs, slip = exit_fill_price(level, direction, row, cost)
            realized += take * s * (px - entry_exec)
            spread_cost += hs * take
            slip_cost += slip * take
            remaining -= take
            if partial_be and not be_armed:
                be_armed = True
                stop = entry_mid
            if remaining <= 1e-9:
                return {
                    'exit_time': ts, 'exit_price': px, 'exit_mid': level,
                    'exit_reason': _classify('partial', realized, be_armed),
                    'raw_exit_reason': 'partial',
                    'pnl_pts': realized,
                    'spread_cost_pts': spread_cost,
                    'slippage_cost_pts': slip_cost,
                    'mae_pts': mae, 'mfe_pts': mfe, 'bars_held': n,
                    'partials_filled': len(legs),
                }

        if be_distance is not None and not be_armed and mfe >= be_distance:
            be_armed = True
            stop = entry_mid

        prev_high, prev_low = hi, lo

    if len(window) > 0:
        last_ts, last = window.index[-1], window.iloc[-1]
        return finish(last_ts, float(last['close']), 'eod', last, len(window))
    # Never traded after the fill bar: flat at the entry mid.
    return finish(fill_time, entry_mid, 'eod', fill, 0)


def trade_costs(exit_info: dict, direction: str, entry_mid: float,
                cost: CostModel, usd_per_point: float) -> dict:
    """Break the round-turn execution cost into its components.

    `pnl_pts` from simulate_exit is already net of spread and slippage, because
    those are baked into the fill prices. This reports the gross-vs-net gap so it
    can be attributed rather than inferred.
    """
    spread_pts = float(exit_info.get('spread_cost_pts', 0.0))
    slip_pts = float(exit_info.get('slippage_cost_pts', 0.0))
    commission_usd = float(cost.commission_round_turn)
    exec_pts = spread_pts + slip_pts
    return {
        'spread_cost_pts': spread_pts,
        'slippage_cost_pts': slip_pts,
        'spread_cost_usd': spread_pts * usd_per_point,
        'slippage_cost_usd': slip_pts * usd_per_point,
        'commission_usd': commission_usd,
        'exec_cost_pts': exec_pts,
        'exec_cost_usd': exec_pts * usd_per_point + commission_usd,
        'pnl_gross_pts': float(exit_info['pnl_pts']) + exec_pts,
    }


def cost_model_from_config(cfg: dict) -> CostModel:
    """Build a CostModel from a backtest config's `costs:` section.

    A missing section means zero cost, which is the legacy behaviour.
    """
    section = (cfg or {}).get('costs') or {}
    known = {f for f in CostModel.__dataclass_fields__}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown cost config keys: {sorted(unknown)}. "
                         f"Known: {sorted(known)}")
    return CostModel(**section)
