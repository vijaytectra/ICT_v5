"""
ICT Concepts — colocated semantic + code layer
==============================================

Each concept lives in its own folder under ``concepts/<author>/<category>/<name>/``
holding its spec (``<slug>.md``), implementation (``<name>.py``), ``tests/`` and
``analysis/``.  This package re-exports every detector's public API flat, so call
sites use a single stable import regardless of where a concept sits in the tree::

    from concepts import find_fvgs, SwingPointScanner, detect_mss

Imports below are ordered leaf-first (dependencies before dependents) so that a
concept module's own ``from concepts import ...`` resolves against the names this
``__init__`` has already bound.  Importing this package also fires every
``@concept`` registration as a side effect (used by ``ict.registry``).
"""

# ── leaves (no concept dependencies) ───────────────────────────────────────
from concepts.ict.pd_arrays.fair_value_gap.fair_value_gap import FVG, find_fvgs
from concepts.ict.market_structure.market_structure import (
    SwingPointScanner, detect_mss, detect_cisd, find_relative_equal_levels,
)
from concepts.merchant.price_action.failure_swing.hpfs import find_hpfs
from concepts.ict.pd_arrays.premium_discount.premium_discount import (
    premium_discount, premium_discount_levels,
)
from concepts.ict.daily_bias.ohlc_profiles.ohlc_profiles import (
    ohlc_candle_profile, entry_side_valid,
)
from concepts.ict.liquidity.liquidity_sweep.liquidity_sweep import (
    detect_liquidity_sweep, classify_sweep_or_run, first_breach,
)
from concepts.ict.entries.smt_divergence.smt_divergence import detect_smt, smt_confirmed
from concepts.ict.time_and_price.killzones.killzones import session_mask, in_killzone
from concepts.ict.time_and_price.sessions.sessions import (
    opening_range, session_high_low, ons_range, chicago_range,
)
from concepts.ict.entries.targets.targets import find_liquidity_target, rr_ratio
from concepts.ict.daily_bias.draw_on_liquidity.draw_on_liquidity import (
    draw_on_liquidity, draw_on_liquidity_levels,
)

# ── composites (depend on the above) ───────────────────────────────────────
from concepts.merchant.price_action.pool_validity.pool_validity import (
    classify_swing_liquidity, hpfs_pools, find_raid_bar,
)
from concepts.merchant.price_action.break_away_gap.break_away_gap import (
    find_break_away_gaps, BreakAwayGap,
)
from concepts.merchant.price_action.dark_pool.dark_pool import find_dark_pools, DarkPool
from concepts.ict.pd_arrays.order_block.order_block import OrderBlock, find_order_blocks

__all__ = [
    'FVG', 'find_fvgs',
    'SwingPointScanner', 'detect_mss', 'detect_cisd', 'find_relative_equal_levels',
    'find_hpfs',
    'premium_discount', 'premium_discount_levels',
    'ohlc_candle_profile', 'entry_side_valid',
    'detect_liquidity_sweep', 'classify_sweep_or_run', 'first_breach',
    'detect_smt', 'smt_confirmed',
    'session_mask', 'in_killzone',
    'opening_range', 'session_high_low', 'ons_range', 'chicago_range',
    'find_liquidity_target', 'rr_ratio',
    'draw_on_liquidity', 'draw_on_liquidity_levels',
    'classify_swing_liquidity', 'hpfs_pools', 'find_raid_bar',
    'find_break_away_gaps', 'BreakAwayGap',
    'find_dark_pools', 'DarkPool',
    'OrderBlock', 'find_order_blocks',
]
