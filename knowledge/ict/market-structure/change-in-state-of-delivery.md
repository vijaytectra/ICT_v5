---
name: change-in-state-of-delivery
aliases: [CISD, change in the state of delivery]
category: market-structure
related: [market-structure-shift, displacement, order-block, daily-bias]
parameters:
  reference: "highest high of the consecutive bearish candles (bullish CISD) / lowest low of consecutive bullish candles (bearish CISD) in the manipulation leg"
  max_leg_lookback: 10
  max_bars_after_sweep: 20
detection: implemented
---

# Change in State of Delivery (CISD)

A finer-grained reversal confirmation than the MSS, popular in later ICT teaching and
the TTrades-style mechanical frameworks. Delivery "state" = whether candles are being
delivered bearish (sell-side delivery) or bullish (buy-side delivery).

A **bullish CISD**: after a down-leg sweeps liquidity, price closes **above the high
of the consecutive bearish candles** that made that final low. Closing above a
manipulation-leg candle's high proves the bearish delivery has failed — price is now
being delivered bullish. Mirror for bearish (close below the low of the bullish
manipulation candles). CISD often triggers a few bars *earlier* than a full MSS (which
needs a swing break), so some models use CISD as the entry trigger and MSS as added
confirmation.

## Detection Rules

- Identify the manipulation leg into the sweep: walk back from the sweep bar, collect
  consecutive same-color candles (bearish = close < open for bullish CISD; bullish =
  close > open for bearish CISD).
- Reference level:
  - Bullish CISD → `reference_high` = highest high among those bearish candles.
  - Bearish CISD → `reference_low` = lowest low among those bullish candles.
- Bullish CISD: `close[t] > reference_high` after the sweep low is in place.
- Bearish CISD: `close[t] < reference_low` after the sweep high is in place.
- Emit {time, reference_level, sweep_ref, direction, leg_candle_count}; the candle
  series also defines the order block zone for re-entry.

## Sources

- [MMXM smart money reversal components (CISD) — tradingview.com](https://www.tradingview.com/script/4eQPT3aC-MMXM-ICT-TradingFinder-Market-Maker-Model-PO3-CHoCH-CSID-FVG/)
- [Change in state of delivery — opofinance.com](https://blog.opofinance.com/en/ict-change-in-state-of-delivery/)
- [TTrades daily bias framework — ttrades.com](https://ttrades.com/ict-daily-bias-most-simple-mechanical-framework/)
