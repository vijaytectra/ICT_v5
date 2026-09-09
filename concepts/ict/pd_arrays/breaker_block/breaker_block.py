"""
Breaker Block
=============

A failed order block: price sweeps a swing, then displaces back *through* the
candle body that produced the sweep, so that body flips polarity and becomes
support/resistance on the retest.

See breaker-block.md
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from concepts.ict.market_structure.market_structure import SwingPointScanner
from infrastructure.registry import concept


@dataclass
class BreakerBlock:
    timestamp: pd.Timestamp     # the candle whose body forms the zone
    direction: str              # 'bullish' (zone is support) or 'bearish' (resistance)
    zone_top: float
    zone_bot: float
    sweep_level: float          # the swing level raided before the displacement (L2 / H2)
    break_time: pd.Timestamp    # the displacement close that broke structure
    structure_level: float      # H1 (bullish) or L1 (bearish)

    @property
    def midpoint(self) -> float:
        """Mean threshold - the 50% of the zone."""
        return (self.zone_top + self.zone_bot) / 2.0

    @property
    def proximal(self) -> float:
        """Edge price retraces into first: zone top for bullish, bottom for bearish."""
        return self.zone_top if self.direction == 'bullish' else self.zone_bot


@concept("breaker-block")
def find_breaker_blocks(
    df: pd.DataFrame,
    direction: str = 'both',
    swing_lookback: int = 2,
    max_bars_sweep_to_break: int = 30,
) -> list:
    """
    Scan df for breaker blocks.

    Implements the sequence in breaker-block.md, bullish case:
      swings L1 < H1  ->  sweep L2 < L1  ->  displacement close above H1.
    The zone is the body of the last down-closed candle between H1 and L2 that
    the displacement traded through. Bearish is the mirror.

    Args:
        df:                      OHLCV DataFrame with DatetimeIndex.
        direction:               'bullish', 'bearish' or 'both'.
        swing_lookback:          Fractal width for swing detection.
        max_bars_sweep_to_break: Bars allowed between the sweep and the
                                 displacement close; beyond this the sequence
                                 is stale.

    Returns:
        List of BreakerBlock, chronological by break_time.
    """
    if len(df) < 5:
        return []

    sw = SwingPointScanner(df, lookback=swing_lookback).identify_swings()
    highs = [(i, float(p)) for i, p in sw['swing_high_price'].items() if pd.notna(p)]
    lows = [(i, float(p)) for i, p in sw['swing_low_price'].items() if pd.notna(p)]
    if not highs or not lows:
        return []

    out = []
    if direction in ('bullish', 'both'):
        out += _scan(df, lows, highs, 'bullish', max_bars_sweep_to_break)
    if direction in ('bearish', 'both'):
        out += _scan(df, highs, lows, 'bearish', max_bars_sweep_to_break)
    out.sort(key=lambda b: b.break_time)
    return out


def _scan(df, own_swings, opp_swings, direction: str, max_bars: int) -> list:
    """Shared bullish/bearish scan; `own_swings` are the ones that get raided."""
    bullish = direction == 'bullish'
    idx = df.index
    pos = {ts: i for i, ts in enumerate(idx)}
    out = []

    for k in range(1, len(own_swings)):
        l1_ts, l1 = own_swings[k - 1]
        l2_ts, l2 = own_swings[k]
        # L2 must actually raid L1 (a lower low, or a higher high when bearish).
        if (l2 >= l1) if bullish else (l2 <= l1):
            continue
        # H1: the opposing swing sitting between L1 and L2 - the level the
        # displacement has to close beyond to break structure.
        between = [(t, p) for t, p in opp_swings if l1_ts < t < l2_ts]
        if not between:
            continue
        h1_ts, h1 = (max(between, key=lambda x: x[1]) if bullish
                     else min(between, key=lambda x: x[1]))

        start = pos[l2_ts]
        window = df.iloc[start + 1: start + 1 + max_bars]
        broke = window[window['close'] > h1] if bullish else window[window['close'] < h1]
        if len(broke) == 0:
            continue
        break_ts = broke.index[0]

        # Zone: last opposite-closed candle between H1's bar and the break.
        seg = df.iloc[pos[h1_ts]: pos[break_ts] + 1]
        cands = seg[seg['close'] < seg['open']] if bullish else seg[seg['close'] > seg['open']]
        if len(cands) == 0:
            continue
        c = cands.iloc[-1]
        top, bot = max(float(c['open']), float(c['close'])), min(float(c['open']), float(c['close']))
        if top <= bot:
            continue

        out.append(BreakerBlock(
            timestamp=cands.index[-1], direction=direction,
            zone_top=top, zone_bot=bot,
            sweep_level=l2, break_time=break_ts, structure_level=h1,
        ))
    return out
