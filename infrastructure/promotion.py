"""Promotion gate - the written answer to "is this model good enough?".

Both of this project's costed studies (judas-swing, model-2022 pool attribution)
produced positive in-sample expectancy and negative out-of-sample expectancy.
Without a threshold fixed *before* the result is seen, that verdict gets argued
per conversation and the bar drifts toward whatever the run produced.

So the gate is code, not prose. It takes an in-sample and an out-of-sample trade
log, reports every criterion with its observed value and threshold, and returns
PROMOTE only if all of them pass. Nothing here knows about a model; it consumes
the same trade-log shape ``compute_stats`` does.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from infrastructure.stats import compute_stats


@dataclass(frozen=True)
class GateCriteria:
    """Thresholds a model must clear out-of-sample to be promoted.

    ``max_expectancy_decay`` is the fraction of in-sample expectancy allowed to
    disappear out-of-sample: 0.5 means OOS expectancy must be at least half of
    in-sample. ``max_loss_streak_fraction`` caps the longest losing run as a
    share of the sample - a 11-trade streak in 73 trades is not a strategy
    property, it is a sample too small to have one.
    """
    min_oos_trades: int = 30
    min_oos_expectancy_r: float = 0.0
    min_oos_profit_factor: float = 1.15
    max_expectancy_decay: float = 0.5
    max_loss_streak_fraction: float = 0.15


DEFAULT_CRITERIA = GateCriteria()


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    verdict: str
    criteria: list[CriterionResult]

    @property
    def failures(self) -> list[CriterionResult]:
        return [c for c in self.criteria if not c.passed]

    def describe(self) -> str:
        lines = [f"{self.verdict}"]
        for c in self.criteria:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        return "\n".join(lines)


def evaluate_gate(is_trades: pd.DataFrame, oos_trades: pd.DataFrame,
                  criteria: GateCriteria = DEFAULT_CRITERIA) -> GateResult:
    """Evaluate every criterion. No short-circuit - all failures are reported."""
    is_stats = compute_stats(is_trades)
    oos_stats = compute_stats(oos_trades)

    n = oos_stats['total_trades']
    oos_exp = oos_stats['expectancy_r']
    oos_pf = oos_stats['profit_factor']
    is_exp = is_stats['expectancy_r']
    streak = oos_stats['max_loss_streak']

    results = [
        CriterionResult(
            name='oos_sample',
            passed=n >= criteria.min_oos_trades,
            observed=n,
            threshold=criteria.min_oos_trades,
            detail=f"{n} out-of-sample trades, need >= {criteria.min_oos_trades}",
        ),
        CriterionResult(
            name='oos_expectancy',
            passed=oos_exp > criteria.min_oos_expectancy_r,
            observed=oos_exp,
            threshold=criteria.min_oos_expectancy_r,
            detail=f"OOS expectancy {oos_exp:+.4f} R, need > "
                   f"{criteria.min_oos_expectancy_r:+.4f} R",
        ),
        CriterionResult(
            name='oos_profit_factor',
            passed=oos_pf >= criteria.min_oos_profit_factor,
            observed=oos_pf,
            threshold=criteria.min_oos_profit_factor,
            detail=f"OOS profit factor {oos_pf:.3f}, need >= "
                   f"{criteria.min_oos_profit_factor:.2f}",
        ),
        _decay_criterion(is_exp, oos_exp, criteria),
        _streak_criterion(streak, n, criteria),
    ]

    passed = all(c.passed for c in results)
    return GateResult(passed=passed,
                      verdict='PROMOTE' if passed else 'REJECT',
                      criteria=results)


def _decay_criterion(is_exp: float, oos_exp: float,
                     criteria: GateCriteria) -> CriterionResult:
    """How much of the in-sample edge survived.

    A non-positive in-sample expectancy fails rather than passes: there is no
    edge to decay from, so the ratio is meaningless and treating that as a pass
    would let a model that never worked through on a technicality.
    """
    if is_exp <= 0:
        return CriterionResult(
            name='expectancy_decay',
            passed=False,
            observed=float('inf'),
            threshold=criteria.max_expectancy_decay,
            detail=f"in-sample expectancy {is_exp:+.4f} R is not positive - "
                   f"no edge to carry out-of-sample",
        )
    decay = (is_exp - oos_exp) / is_exp
    return CriterionResult(
        name='expectancy_decay',
        passed=decay <= criteria.max_expectancy_decay,
        observed=round(decay, 4),
        threshold=criteria.max_expectancy_decay,
        detail=f"{decay:.1%} of in-sample expectancy lost "
               f"({is_exp:+.4f} -> {oos_exp:+.4f} R), allowed "
               f"{criteria.max_expectancy_decay:.0%}",
    )


def _streak_criterion(streak: int, n: int,
                      criteria: GateCriteria) -> CriterionResult:
    """Longest losing run as a share of the out-of-sample sample."""
    if n == 0:
        return CriterionResult(
            name='loss_streak',
            passed=False,
            observed=float('inf'),
            threshold=criteria.max_loss_streak_fraction,
            detail="no out-of-sample trades to measure a streak on",
        )
    fraction = streak / n
    return CriterionResult(
        name='loss_streak',
        passed=fraction <= criteria.max_loss_streak_fraction,
        observed=round(fraction, 4),
        threshold=criteria.max_loss_streak_fraction,
        detail=f"longest losing run {streak}/{n} = {fraction:.1%} of the "
               f"sample, allowed {criteria.max_loss_streak_fraction:.0%}",
    )
