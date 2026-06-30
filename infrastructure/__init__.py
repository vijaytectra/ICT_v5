"""
Infrastructure
==============

Shared library machinery for the ICT trading project — the cross-cutting code
that is not a concept or a model:

- ``registry``    — the ``@concept`` registry + lineage graph (binds specs to code)
- ``loader``      — :class:`DataLoader` for the ES/NQ/YM CSVs
- ``report``      — reusable backtest report engine (``Mark``, ``TradeViz``, ``build_report``)
- ``time_utils``  — session/time helpers

Concepts live in the top-level ``concepts`` package; full strategies live in the
top-level ``models`` package. Both compose this infrastructure.
"""

from infrastructure.loader import DataLoader

__version__ = "0.1.0"

__all__ = ["DataLoader"]
