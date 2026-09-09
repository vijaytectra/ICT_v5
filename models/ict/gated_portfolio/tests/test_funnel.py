"""Funnel scanner tests. Synthetic bars only - no data files needed.

`scan_session` is deliberately thin over the existing concept detectors. These
tests are about the *composition* - which stage a session drops out at, which
direction a swept pool implies, how the target is chosen and how R is computed -
not about re-testing find_fvgs or detect_mss.
"""
import pandas as pd
import pytest

from models.ict.gated_portfolio.gated_portfolio import (FUNNEL_STAGES, Candidate,
                                                        scan_session)


def _bars(rows, start='2024-01-02 13:30', freq='5min'):
    """rows: list of (open, high, low, close)."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz='UTC')
    return pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'], index=idx)


# A session that runs the full chain: sweeps 110, shifts below the 98 swing low,
# then prints a bearish FVG whose entry is 95.
_FULL_CHAIN = [
    (100, 102, 99, 101),
    (101, 103, 100, 102),
    (102, 104, 101, 103),
    (103, 104, 98, 99),        # 3: swing low at 98
    (99, 103, 98.5, 102),
    (102, 105, 101, 104),
    (104, 106, 103, 105),
    (105, 107, 104, 106),
    (106, 108, 105, 107),
    (107, 109, 106, 108),
    (108, 112, 107, 109),      # 10: sweeps 110, closes back below
    (109, 110, 105, 106),
    (106, 107, 100, 101),
    (101, 102, 96, 97),        # 13: closes below 98 -> bearish MSS
    (97, 98, 95, 96),          # 14: low 95
    (96, 96, 90, 91),          # 15: displacement
    (91, 94, 89, 90),          # 16: high 94 -> bearish FVG (95 / 94)
]

_HIGH_POOL = {'name': 'asia_high', 'level': 110.0, 'side': 'high'}


def _pools(*extra):
    return [_HIGH_POOL, *extra]


# --- stage accounting -----------------------------------------------------

def test_stage_keys_are_the_declared_funnel():
    scan = scan_session(_bars(_FULL_CHAIN), _pools())
    assert list(scan.stages.keys()) == list(FUNNEL_STAGES)


def test_untouched_pool_stops_at_the_sweep_stage():
    pools = [{'name': 'far_high', 'level': 200.0, 'side': 'high'}]
    scan = scan_session(_bars(_FULL_CHAIN), pools)
    assert scan.stages['pools'] == 1
    assert scan.stages['sweeps'] == 0
    assert scan.candidates == []


def test_sweep_without_a_structure_shift_stops_at_the_reversal_stage():
    """Bars after the sweep drift sideways - the 98 swing low is never broken."""
    rows = _FULL_CHAIN[:11] + [(109, 110, 105, 106)] * 6
    scan = scan_session(_bars(rows), _pools())
    assert scan.stages['sweeps'] == 1
    assert scan.stages['reversals'] == 0


def test_shift_without_an_entry_array_stops_at_the_entry_stage():
    """Price shifts down one bar at a time, leaving no three-bar imbalance."""
    rows = _FULL_CHAIN[:14] + [
        (97, 98, 96, 97), (97, 98, 95, 96), (96, 97, 94, 95), (95, 96, 93, 94)]
    scan = scan_session(_bars(rows), _pools())
    assert scan.stages['reversals'] == 1
    assert scan.stages['entries'] == 0


def test_full_chain_with_a_distant_target_produces_a_candidate():
    pools = _pools({'name': 'pdl', 'level': 60.0, 'side': 'low'})
    scan = scan_session(_bars(_FULL_CHAIN), pools)
    assert scan.stages['entries'] == 1
    assert scan.stages['rr_pass'] == 1
    assert len(scan.candidates) == 1


def test_full_chain_with_a_near_target_fails_the_reward_risk_floor():
    """Same setup, target at 90: R = (95-90)/(112-95) = 0.29."""
    pools = _pools({'name': 'asia_low', 'level': 90.0, 'side': 'low'})
    scan = scan_session(_bars(_FULL_CHAIN), pools)
    assert scan.stages['entries'] == 1
    assert scan.stages['rr_pass'] == 0
    assert scan.candidates == []


# --- what the candidate carries -------------------------------------------

def test_sweeping_a_high_produces_a_short_candidate():
    pools = _pools({'name': 'pdl', 'level': 60.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools).candidates[0]
    assert isinstance(c, Candidate)
    assert c.direction == 'short'
    assert c.swept_pool == 'asia_high'


def test_stop_sits_beyond_the_sweep_extreme():
    pools = _pools({'name': 'pdl', 'level': 60.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools).candidates[0]
    assert c.stop == pytest.approx(112.0)
    assert c.entry == pytest.approx(95.0)


def test_reward_risk_is_computed_from_entry_stop_and_target():
    pools = _pools({'name': 'pdl', 'level': 60.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools).candidates[0]
    assert c.reward_risk == pytest.approx((95.0 - 60.0) / (112.0 - 95.0), rel=1e-6)


def test_nearest_opposite_pool_is_chosen_as_the_target():
    """Two valid downside targets: the nearer one is the honest denominator."""
    pools = _pools({'name': 'far', 'level': 40.0, 'side': 'low'},
                   {'name': 'near', 'level': 68.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools).candidates[0]
    assert c.target == pytest.approx(68.0)
    assert c.target_pool == 'near'


def test_a_near_target_below_the_floor_rejects_even_when_a_far_pool_would_pass():
    """R is measured to the first level in the path, not to the best available.

    near=70 gives R = (95-70)/17 = 1.47; the 40 pool would give 3.24. Pricing in
    the further move the setup has not earned is exactly the inflation the floor
    exists to catch, so the candidate is rejected.
    """
    pools = _pools({'name': 'far', 'level': 40.0, 'side': 'low'},
                   {'name': 'near', 'level': 70.0, 'side': 'low'})
    scan = scan_session(_bars(_FULL_CHAIN), pools)
    assert scan.stages['entries'] == 1
    assert scan.stages['rr_pass'] == 0
    assert scan.candidates == []


def test_a_pool_on_the_wrong_side_of_entry_is_not_a_target():
    """A 'low' pool sitting above the short entry cannot be a downside target."""
    pools = _pools({'name': 'above', 'level': 99.0, 'side': 'low'},
                   {'name': 'pdl', 'level': 60.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools).candidates[0]
    assert c.target == pytest.approx(60.0)


def test_no_opposite_pool_means_no_candidate():
    scan = scan_session(_bars(_FULL_CHAIN), _pools())
    assert scan.stages['entries'] == 1
    assert scan.candidates == []


def test_session_label_is_carried_onto_the_candidate():
    pools = _pools({'name': 'pdl', 'level': 60.0, 'side': 'low'})
    c = scan_session(_bars(_FULL_CHAIN), pools, session='ny_am').candidates[0]
    assert c.session == 'ny_am'


# --- structural context ---------------------------------------------------
#
# A sweep landing early in a session has almost no bars ahead of it inside the
# window, so the swing the shift would break was formed *before* the window
# opened. Scanning the window alone therefore reports "no reversal" for reasons
# of slicing rather than of price - which is what understated the NY AM funnel
# (368 sweeps, 16 reversals) against London's (105 sweeps, 27 reversals).

def test_sweep_at_the_window_open_finds_no_shift_without_context():
    full = _bars(_FULL_CHAIN)
    window = full.iloc[10:]                      # opens on the sweep bar itself
    scan = scan_session(window, _pools())
    assert scan.stages['sweeps'] == 1
    assert scan.stages['reversals'] == 0


def test_same_sweep_finds_its_shift_when_context_is_supplied():
    full = _bars(_FULL_CHAIN)
    window = full.iloc[10:]
    scan = scan_session(window, _pools(), context=full)
    assert scan.stages['sweeps'] == 1
    assert scan.stages['reversals'] == 1


def test_context_does_not_admit_sweeps_that_happened_before_the_window():
    """The 110 pool is swept at bar 10; a window starting after that sees none."""
    full = _bars(_FULL_CHAIN)
    window = full.iloc[11:]
    scan = scan_session(window, _pools(), context=full)
    assert scan.stages['sweeps'] == 0
    assert scan.candidates == []


def test_context_defaults_to_the_window():
    full = _bars(_FULL_CHAIN)
    assert (scan_session(full, _pools()).stages
            == scan_session(full, _pools(), context=full).stages)


# --- configuration --------------------------------------------------------

def test_lowering_the_floor_admits_the_near_target():
    pools = _pools({'name': 'asia_low', 'level': 90.0, 'side': 'low'})
    scan = scan_session(_bars(_FULL_CHAIN), pools, min_rr=0.2)
    assert scan.stages['rr_pass'] == 1


def test_empty_frame_returns_zeroed_stages_without_raising():
    scan = scan_session(_bars([]), _pools())
    assert scan.stages['sweeps'] == 0
    assert scan.candidates == []


def test_pools_stage_counts_every_pool_offered():
    pools = _pools({'name': 'a', 'level': 60.0, 'side': 'low'},
                   {'name': 'b', 'level': 50.0, 'side': 'low'})
    scan = scan_session(_bars(_FULL_CHAIN), pools)
    assert scan.stages['pools'] == 3
