"""
Composite trading models, organized by author/source.

- ict/      — ICT (Inner Circle Trader) models: 2022 model, FVG sweep, daily bias, ...
- romeo/    — Romeo's Candle Range Theory (CRT) models
- merchant/ — The Currency Merchant (Kishane) models

Each model composes primitives from the `concepts` package. The conceptual spec for every
model spec lives in its folder under `models/<author>/<name>/`; model classes should reference the
concept slug they implement.
"""
