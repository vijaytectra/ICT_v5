"""Exit-policy simulator - replay one trade's bars under a different exit rule.

The win-rate/payoff frontier study found `corr(win rate, payoff) = -0.945` across
13 configurations: they are one number, not two, and the exit policy is what
chooses the point on the curve. This module is what lets that curve be swept
without touching a single entry rule.

Why bars and not MAE/MFE: a trade log's `mae_r` and `mfe_r` say how far the trade
ran each way but not in which order. Whether +1.5 R arrived before or after
-0.8 R decides whether a partial exit fired at all, so the path has to be
replayed rather than summarised.

Two conventions, both deliberately pessimistic:

* **Intrabar ordering** - if a bar's range contains both the stop and the
  target, the stop is taken. Without tick data the order is unknowable, and the
  cost of guessing wrong should fall on the strategy, not on the operator.
* **R is measured against the risk taken at entry**, frozen. Once a stop trails
  to breakeven the live stop distance is zero; dividing by it is the bug that
  cost the judas baseline a fabricated trade.

Costs are charged in R because that is where they bite: a 0.15 R round turn is a
twentieth of a 3 R win and a quarter of a 0.6 R win. Any move toward the
high-win-rate end of the frontier makes cost a larger share of the payoff, and
charging in R is what makes that visible instead of invisible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ExitPolicy:
    """One point on the exit frontier.

    ``target_r``        final target in R; None runs to the time stop or data end.
    ``partial_r``       take ``partial_fraction`` off here; None disables partials.
    ``breakeven_r``     move the stop to entry once this R is reached.
    ``time_stop_bars``  close whatever is left at this bar's close.
    """
    target_r: Optional[float] = None
    partial_r: Optional[float] = None
    partial_fraction: float = 0.5
    breakeven_r: Optional[float] = None
    time_stop_bars: Optional[int] = None

    def label(self) -> str:
        bits = [f"tp{self.target_r:g}" if self.target_r else "tp-"]
        if self.partial_r:
            bits.append(f"p{self.partial_fraction:g}@{self.partial_r:g}")
        if self.breakeven_r:
            bits.append(f"be{self.breakeven_r:g}")
        if self.time_stop_bars:
            bits.append(f"t{self.time_stop_bars}")
        return "_".join(bits)


@dataclass(frozen=True)
class ExitResult:
    pnl_r: float
    exit_reason: str
    bars_held: int
    partial_hit: bool
    mfe_r: float


def simulate_exit(bars: pd.DataFrame, entry_price: float, stop: float,
                  direction: str, policy: ExitPolicy,
                  cost_r: float = 0.0,
                  risk: Optional[float] = None) -> ExitResult:
    """Replay ``bars`` (those strictly after the fill) under ``policy``.

    Returns realised R for the whole position, net of ``cost_r``.

    ``risk`` overrides the entry-to-stop distance as the R unit. A trade log may
    measure R on the *mid* entry while filling at the ask - the judas log does,
    with ``risk_pts`` reconciling to ``|entry_mid - stop|`` exactly and the
    spread charged separately. Deriving risk from the fill price in that case
    silently rescales every R in the replay, so the caller must be able to say
    which distance is the unit.
    """
    if risk is None:
        risk = abs(entry_price - stop)
    if risk <= 0:
        raise ValueError(
            f"risk must be positive: entry={entry_price}, stop={stop}, "
            f"risk={risk}")

    long = direction == 'long'
    sign = 1.0 if long else -1.0

    def r_of(price: float) -> float:
        return sign * (price - entry_price) / risk

    def price_at(r: float) -> float:
        return entry_price + sign * r * risk

    if bars is None or len(bars) == 0:
        return ExitResult(pnl_r=0.0, exit_reason='EOD', bars_held=0,
                          partial_hit=False, mfe_r=0.0)

    stop_price = stop
    remaining = 1.0
    booked = 0.0
    partial_hit = False
    mfe_r = 0.0

    target_price = price_at(policy.target_r) if policy.target_r else None
    partial_price = price_at(policy.partial_r) if policy.partial_r else None
    be_price = price_at(policy.breakeven_r) if policy.breakeven_r else None

    for i, (_, bar) in enumerate(bars.iterrows(), start=1):
        hi, lo = float(bar['high']), float(bar['low'])
        best = hi if long else lo
        mfe_r = max(mfe_r, r_of(best))

        # 1. adverse first - the pessimistic convention.
        stopped = lo <= stop_price if long else hi >= stop_price
        if stopped:
            booked += remaining * r_of(stop_price)
            reason = 'BE' if stop_price == entry_price else 'SL'
            return ExitResult(pnl_r=booked - cost_r, exit_reason=reason,
                              bars_held=i, partial_hit=partial_hit,
                              mfe_r=mfe_r)

        # 2. partial.
        if (partial_price is not None and not partial_hit
                and (hi >= partial_price if long else lo <= partial_price)):
            booked += policy.partial_fraction * r_of(partial_price)
            remaining -= policy.partial_fraction
            partial_hit = True

        # 3. arm breakeven.
        if (be_price is not None
                and (hi >= be_price if long else lo <= be_price)):
            stop_price = entry_price

        # 4. target.
        if (target_price is not None
                and (hi >= target_price if long else lo <= target_price)):
            booked += remaining * r_of(target_price)
            return ExitResult(pnl_r=booked - cost_r, exit_reason='TP',
                              bars_held=i, partial_hit=partial_hit,
                              mfe_r=mfe_r)

        # 5. time stop.
        if policy.time_stop_bars is not None and i >= policy.time_stop_bars:
            booked += remaining * r_of(float(bar['close']))
            return ExitResult(pnl_r=booked - cost_r, exit_reason='TIME',
                              bars_held=i, partial_hit=partial_hit,
                              mfe_r=mfe_r)

    booked += remaining * r_of(float(bars['close'].iloc[-1]))
    return ExitResult(pnl_r=booked - cost_r, exit_reason='EOD',
                      bars_held=len(bars), partial_hit=partial_hit,
                      mfe_r=mfe_r)
