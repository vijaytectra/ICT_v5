"""Exit-policy simulator tests. Synthetic bars only - no data files needed.

Every case uses entry 100, initial stop 90, so risk = 10 points and 1 R = 10
points. Target/partial/breakeven levels therefore land on round numbers:
1.0 R = 110, 1.5 R = 115, 2.0 R = 120, 3.0 R = 130.
"""
import pandas as pd
import pytest

from infrastructure.exits import ExitPolicy, simulate_exit


def _bars(rows, start='2024-01-02 14:00'):
    idx = pd.date_range(start, periods=len(rows), freq='5min', tz='UTC')
    return pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'], index=idx)


def _long(bars, policy, **kw):
    return simulate_exit(bars, entry_price=100.0, stop=90.0,
                         direction='long', policy=policy, **kw)


# --- the two simple outcomes ----------------------------------------------

def test_stop_hit_loses_one_r():
    r = _long(_bars([(100, 102, 89, 90)]), ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(-1.0)
    assert r.exit_reason == 'SL'


def test_target_hit_wins_the_target_r():
    r = _long(_bars([(100, 121, 99, 120)]), ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(2.0)
    assert r.exit_reason == 'TP'


def test_stop_wins_when_both_are_touched_in_one_bar():
    """Conservative intrabar ordering: assume the adverse level came first."""
    r = _long(_bars([(100, 121, 89, 100)]), ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(-1.0)
    assert r.exit_reason == 'SL'


# --- partial exits --------------------------------------------------------

def test_partial_then_breakeven_stop_banks_only_the_partial():
    """Half off at 1.5 R, stop to entry, reversal: +0.75 R, not -1 R."""
    policy = ExitPolicy(target_r=3.0, partial_r=1.5, partial_fraction=0.5,
                        breakeven_r=1.5)
    r = _long(_bars([(100, 116, 99, 115),
                     (115, 116, 99, 100)]), policy)
    assert r.pnl_r == pytest.approx(0.75)
    assert r.partial_hit is True
    assert r.exit_reason == 'BE'


def test_partial_then_target_banks_both_legs():
    policy = ExitPolicy(target_r=3.0, partial_r=1.5, partial_fraction=0.5,
                        breakeven_r=1.5)
    r = _long(_bars([(100, 116, 99, 115),
                     (115, 131, 114, 130)]), policy)
    assert r.pnl_r == pytest.approx(0.5 * 1.5 + 0.5 * 3.0)
    assert r.exit_reason == 'TP'


def test_partial_fraction_is_respected():
    policy = ExitPolicy(target_r=3.0, partial_r=1.5, partial_fraction=0.25,
                        breakeven_r=1.5)
    r = _long(_bars([(100, 116, 99, 115),
                     (115, 116, 99, 100)]), policy)
    assert r.pnl_r == pytest.approx(0.25 * 1.5)


def test_stop_before_the_partial_is_still_a_full_loss():
    policy = ExitPolicy(target_r=3.0, partial_r=1.5, partial_fraction=0.5,
                        breakeven_r=1.5)
    r = _long(_bars([(100, 105, 89, 90)]), policy)
    assert r.pnl_r == pytest.approx(-1.0)
    assert r.partial_hit is False


# --- breakeven without a partial ------------------------------------------

def test_breakeven_alone_turns_a_loser_into_a_scratch():
    policy = ExitPolicy(target_r=3.0, breakeven_r=1.0)
    r = _long(_bars([(100, 111, 99, 110),
                     (110, 111, 99, 100)]), policy)
    assert r.pnl_r == pytest.approx(0.0)
    assert r.exit_reason == 'BE'


def test_breakeven_does_not_arm_below_its_trigger():
    policy = ExitPolicy(target_r=3.0, breakeven_r=2.0)
    r = _long(_bars([(100, 111, 99, 110),
                     (110, 111, 89, 90)]), policy)
    assert r.pnl_r == pytest.approx(-1.0)
    assert r.exit_reason == 'SL'


# --- time and data exhaustion ---------------------------------------------

def test_time_stop_exits_at_that_bars_close():
    policy = ExitPolicy(time_stop_bars=2)
    r = _long(_bars([(100, 105, 99, 104),
                     (104, 108, 103, 107),
                     (107, 109, 106, 108)]), policy)
    assert r.pnl_r == pytest.approx(0.7)
    assert r.exit_reason == 'TIME'
    assert r.bars_held == 2


def test_running_out_of_bars_exits_at_the_last_close():
    r = _long(_bars([(100, 105, 99, 104), (104, 108, 103, 106)]),
              ExitPolicy(target_r=5.0))
    assert r.pnl_r == pytest.approx(0.6)
    assert r.exit_reason == 'EOD'


# --- shorts mirror longs ---------------------------------------------------

def test_short_target_mirrors_the_long_case():
    r = simulate_exit(_bars([(100, 101, 79, 80)]), entry_price=100.0,
                      stop=110.0, direction='short',
                      policy=ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(2.0)
    assert r.exit_reason == 'TP'


def test_short_stop_mirrors_the_long_case():
    r = simulate_exit(_bars([(100, 111, 99, 110)]), entry_price=100.0,
                      stop=110.0, direction='short',
                      policy=ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(-1.0)


def test_short_partial_then_breakeven():
    policy = ExitPolicy(target_r=3.0, partial_r=1.5, partial_fraction=0.5,
                        breakeven_r=1.5)
    r = simulate_exit(_bars([(100, 101, 84, 85),
                             (85, 101, 84, 100)]), entry_price=100.0,
                      stop=110.0, direction='short', policy=policy)
    assert r.pnl_r == pytest.approx(0.75)


# --- R is always measured against the risk taken at entry ------------------

def test_r_uses_the_original_risk_after_the_stop_has_moved():
    """The judas baseline lost a trade to this: once the stop trails to entry,
    live stop distance is zero and any R computed from it divides by zero."""
    policy = ExitPolicy(target_r=3.0, breakeven_r=1.0)
    r = _long(_bars([(100, 111, 99, 110),
                     (110, 131, 109, 130)]), policy)
    assert r.pnl_r == pytest.approx(3.0)


def test_supplied_risk_overrides_the_entry_to_stop_distance():
    """A trade log may measure R on the mid entry while filling at the ask.

    The judas log does exactly this: `risk_pts` reconciles to
    `|entry_mid - stop|` exactly, and the spread is charged separately. Deriving
    risk from the fill price instead silently rescales every R in the replay.
    """
    r = simulate_exit(_bars([(100, 102, 89, 90)]), entry_price=100.0,
                      stop=90.0, direction='long',
                      policy=ExitPolicy(target_r=2.0), risk=20.0)
    assert r.pnl_r == pytest.approx(-0.5)


def test_supplied_risk_also_places_the_target():
    r = simulate_exit(_bars([(100, 141, 99, 140)]), entry_price=100.0,
                      stop=90.0, direction='long',
                      policy=ExitPolicy(target_r=2.0), risk=20.0)
    assert r.pnl_r == pytest.approx(2.0)
    assert r.exit_reason == 'TP'


def test_supplied_risk_must_be_positive():
    with pytest.raises(ValueError):
        simulate_exit(_bars([(100, 102, 99, 101)]), entry_price=100.0,
                      stop=90.0, direction='long', policy=ExitPolicy(),
                      risk=0.0)


def test_zero_risk_is_rejected():
    with pytest.raises(ValueError):
        simulate_exit(_bars([(100, 101, 99, 100)]), entry_price=100.0,
                      stop=100.0, direction='long', policy=ExitPolicy())


# --- costs -----------------------------------------------------------------

def test_cost_is_charged_in_r():
    bars = _bars([(100, 121, 99, 120)])
    free = _long(bars, ExitPolicy(target_r=2.0))
    costed = _long(bars, ExitPolicy(target_r=2.0), cost_r=0.15)
    assert costed.pnl_r == pytest.approx(free.pnl_r - 0.15)


def test_cost_makes_a_small_win_a_loss():
    """The whole point of charging in R: at a 0.6 R payoff a 0.15 R round turn
    is a quarter of the win, against a twentieth of a 3 R win."""
    r = _long(_bars([(100, 107, 99, 106)]), ExitPolicy(target_r=0.6),
              cost_r=0.15)
    assert r.pnl_r == pytest.approx(0.45)


# --- excursion reporting ---------------------------------------------------

def test_mfe_records_the_best_the_trade_ever_saw():
    r = _long(_bars([(100, 125, 99, 105), (105, 106, 89, 90)]),
              ExitPolicy(target_r=5.0))
    assert r.mfe_r == pytest.approx(2.5)


def test_empty_bars_return_a_flat_scratch():
    r = _long(_bars([]), ExitPolicy(target_r=2.0))
    assert r.pnl_r == pytest.approx(0.0)
    assert r.bars_held == 0
