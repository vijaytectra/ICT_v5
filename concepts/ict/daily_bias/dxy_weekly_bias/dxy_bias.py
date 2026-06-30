"""
DXY Weekly Bias
===============

Classifies the weekly directional bias for equity indices (NQ/ES/YM) based on
DXY liquidity pool raids and subsequent reaction confirmation.

Logic:
  1. Scan DXY weekly bars for failure-swing highs and lows (valid pools).
  2. Find the most recently raided pool before `as_of`.
  3. After the raid, confirm a reaction via MSS or CISD.
  4. Buy-side pool raid + bearish DXY reaction  → DXY going lower → equities BULLISH.
     Sell-side pool raid + bullish DXY reaction → DXY going higher → equities BEARISH.

See dxy-bias.md
"""

from __future__ import annotations

from typing import Optional, Literal
import pandas as pd

from infrastructure.registry import concept
from concepts import SwingPointScanner, detect_mss, detect_cisd
from concepts import classify_swing_liquidity


@concept(
    "dxy-weekly-bias",
    depends_on=["swing-without-liquidity", "market-structure-shift", "change-in-state-of-delivery"],
)
def dxy_weekly_bias(
    dxy_weekly: pd.DataFrame,
    as_of: pd.Timestamp,
    swing_lookback: int = 2,
    reaction_bars: int = 3,
    reaction_type: Literal["mss", "cisd", "either"] = "either",
    max_leg_lookback: int = 5,
) -> Optional[dict]:
    """
    Classify equity index weekly bias from DXY liquidity pool reactions.

    Args:
        dxy_weekly:      DXY weekly OHLCV bars (DatetimeIndex, ascending).
        as_of:           Start of the week being predicted. Only bars strictly
                         before this timestamp are used (no lookahead).
        swing_lookback:  Bars on each side to confirm a swing high/low.
        reaction_bars:   Max weekly bars after the raid to find MSS/CISD.
        reaction_type:   Which confirmation to accept: 'mss', 'cisd', or 'either'.
        max_leg_lookback: Bars back from the raid to find the CISD manipulation leg.

    Returns:
        dict with keys:
          bias          — 'bullish' or 'bearish' (equity index direction)
          dxy_direction — 'bearish' or 'bullish' (DXY reaction direction)
          reaction      — 'mss' or 'cisd'
          pool_side     — 'high' (buy-side pool raided) or 'low' (sell-side)
          pool_level    — price of the raided pool
          raid_time     — timestamp of the raid bar
          confirm_time  — timestamp of the MSS/CISD confirmation bar
        or None if no valid setup found.
    """
    history = dxy_weekly[dxy_weekly.index < as_of].copy()
    if len(history) < swing_lookback * 2 + reaction_bars + 2:
        return None

    scanner = SwingPointScanner(history, lookback=swing_lookback)
    scanner.identify_swings()

    if scanner.swing_highs is None or scanner.swing_lows is None:
        return None

    # Classify each swing as failure_swing (untouched liquidity) or no_liquidity.
    classified_highs = classify_swing_liquidity(scanner.swing_highs, "high")
    classified_lows  = classify_swing_liquidity(scanner.swing_lows,  "low")

    valid_highs = classified_highs[classified_highs["classification"] == "failure_swing"]
    valid_lows  = classified_lows[classified_lows["classification"]  == "failure_swing"]

    # For each valid pool find the most recent raid bar.
    candidates = []

    for ts, row in valid_highs.iterrows():
        level = row["swing_high_price"]
        future = history[history.index > ts]
        raided = future[future["high"] > level]
        if len(raided) == 0:
            continue
        raid_time = raided.index[0]
        candidates.append({
            "pool_side":  "high",
            "pool_level": level,
            "pool_time":  ts,
            "raid_time":  raid_time,
        })

    for ts, row in valid_lows.iterrows():
        level = row["swing_low_price"]
        future = history[history.index > ts]
        raided = future[future["low"] < level]
        if len(raided) == 0:
            continue
        raid_time = raided.index[0]
        candidates.append({
            "pool_side":  "low",
            "pool_level": level,
            "pool_time":  ts,
            "raid_time":  raid_time,
        })

    if not candidates:
        return None

    # Pick the most recently raided pool.
    most_recent = max(candidates, key=lambda c: c["raid_time"])
    raid_time  = most_recent["raid_time"]
    pool_side  = most_recent["pool_side"]

    # Extract the reaction window: bars from the raid onward (inclusive).
    reaction_window = history[history.index >= raid_time].head(reaction_bars + 1)
    if len(reaction_window) < 2:
        return None

    # Direction of expected DXY reaction:
    #   buy-side pool raid  → expect DXY to fail → bearish DXY reaction
    #   sell-side pool raid → expect DXY to rally → bullish DXY reaction
    dxy_direction = "bearish" if pool_side == "high" else "bullish"

    confirmed  = None
    conf_type  = None

    if reaction_type in ("mss", "either"):
        result = detect_mss(
            reaction_window,
            direction=dxy_direction,
            swing_lookback=1,
            max_bars_after_sweep=reaction_bars,
            sweep_time=raid_time,
        )
        if result is not None:
            confirmed = result["break_time"]
            conf_type = "mss"

    if confirmed is None and reaction_type in ("cisd", "either"):
        result = detect_cisd(
            reaction_window,
            direction=dxy_direction,
            sweep_time=raid_time,
            max_leg_lookback=max_leg_lookback,
            max_bars_after_sweep=reaction_bars,
        )
        if result is not None:
            confirmed = result["cisd_time"]
            conf_type = "cisd"

    if confirmed is None:
        return None

    # DXY bearish → equities bullish; DXY bullish → equities bearish.
    equity_bias = "bullish" if dxy_direction == "bearish" else "bearish"

    return {
        "bias":          equity_bias,
        "dxy_direction": dxy_direction,
        "reaction":      conf_type,
        "pool_side":     pool_side,
        "pool_level":    most_recent["pool_level"],
        "raid_time":     raid_time,
        "confirm_time":  confirmed,
    }
