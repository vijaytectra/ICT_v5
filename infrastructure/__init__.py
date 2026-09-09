"""
Infrastructure
==============

Shared library machinery for the ICT trading project — the cross-cutting code
that is not a concept or a model:

- ``registry``    — the ``@concept`` registry + lineage graph (binds specs to code)
- ``loader``      — :class:`DataLoader` for the ES/NQ/YM CSVs
- ``report``      — reusable backtest report engine (``Mark``, ``TradeViz``, ``build_report``)
- ``execution``   — execution cost model + cost-aware fill/exit simulation
- ``stats``       — trade-log statistics (R-based), breakdowns, exit-reason table
- ``exits``       — exit-policy simulator: replay a trade under a different exit rule
- ``walkforward`` — expanding-window walk-forward selection and stitching
- ``promotion``   — the promotion gate: is this model good enough to trust?
- ``tripwire``    — leakage tripwire: does it still earn on shuffled bars?
- ``time_utils``  — session/time helpers

Concepts live in the top-level ``concepts`` package; full strategies live in the
top-level ``models`` package. Both compose this infrastructure.
"""

from infrastructure.loader import DataLoader
from infrastructure.execution import CostModel, ZERO_COST, cost_model_from_config
from infrastructure.promotion import (DEFAULT_CRITERIA, GateCriteria, GateResult,
                                      evaluate_gate)
from infrastructure.tripwire import TripwireResult, run_tripwire, shuffle_bars
from infrastructure.exits import ExitPolicy, ExitResult, simulate_exit

__version__ = "0.1.0"

__all__ = ["DataLoader", "CostModel", "ZERO_COST", "cost_model_from_config",
           "GateCriteria", "GateResult", "DEFAULT_CRITERIA", "evaluate_gate",
           "TripwireResult", "run_tripwire", "shuffle_bars",
           "ExitPolicy", "ExitResult", "simulate_exit"]
