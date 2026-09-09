# Experiment Configs & Infrastructure

Named YAML configs and the tooling that runs them. Three-layer system for parameterizing and comparing model configurations.

> Referenced from `CLAUDE.md`. Read this when running a single config, comparing runs, or doing a grid search.

---

## 1. Model config flags (Option C)

All knobs live in `Model2022.DEFAULT_CONFIG`. Pass a dict to `Model2022(config={...})` to override any key.

**Pool toggles** (all `bool`, default `True`):
`use_pdh_pdl`, `use_ons`, `use_chicago_prior`, `use_chicago_same_day`, `use_opening_range`, `use_reh_rel`

**Stop modes** (`stop_mode` key):
| Value | Description |
|---|---|
| `'wick'` | Stop at swept wick extreme, capped by `max_stop_pts` |
| `'atr'` | Stop at `entry ± stop_atr_mult × ATR(stop_atr_period)` on 2m bars, capped by `max_stop_pts` |
| `'fixed'` | Stop at `entry ± stop_fixed_pts`, capped by `max_stop_pts` |

**Other model keys**: `max_stop_pts`, `target_min_rr`, `killzones`, `max_bars_sweep_to_mss`, `eq_tolerance_pts`, `reh_lookback_days`, `swing_lookback`, `daily_lookback`, `dealing_range_days`

---

## 2. Named YAML configs + single-run backtest (Option A)

```
configs/model_2022/          # one file per named experiment
  baseline.yaml              # current production config
  with_bias.yaml
  pdh_pdl_only.yaml
  london_only.yaml
```

**YAML schema** (all keys optional — missing keys use defaults):
```yaml
name: my_experiment          # → results/my_experiment/ output dir
description: "human note"
start_date: "2024-11-01"
instrument: NQ
use_daily_bias: false        # false = try both directions per day
# any Model2022.DEFAULT_CONFIG key can go here:
stop_mode: wick
max_stop_pts: 20.0
use_pdh_pdl: true
killzones:
  - ["02:00", "05:00"]
  - ["07:00", "10:00"]
```

**Run a single config:**
```bash
.venv/bin/python backtests/fvg_sweep_backtest.py --config configs/model_2022/baseline.yaml
```

**Compare all runs:**
```bash
.venv/bin/python backtests/compare_runs.py                     # all results/ dirs
.venv/bin/python backtests/compare_runs.py baseline with_bias  # named only
.venv/bin/python backtests/compare_runs.py --sort sharpe --save
```

---

## 2b. Forex configs (EURUSD / GBPUSD / USDJPY)

`forex_eurusd.yaml`, `forex_gbpusd.yaml`, `forex_usdjpy.yaml` run the same model on
spot FX. Three things differ from the index-futures configs, all sourced from the
ICT specs in-repo rather than guessed:

**Killzones** - `killzones.md` frontmatter carries a separate forex NY window
(`ny_kz_forex_ny` 07:00-10:00, vs `ny_kz_indices_ny` 08:30-11:00). The forex configs
use London 02:00-05:00, NY 07:00-10:00 and London Close 10:00-12:00, dropping the
equities PM session (13:30-16:00). These agree with the 2020 GMT forex notes
(`strategies/notes/ict-forex-notes-mmari-2020.md` section 9) once converted to NY time.

**Pools** - spot FX has no CME pit and no RTH cash open, so `use_chicago_prior`,
`use_chicago_same_day` and `use_ons` (all globex/pit windows) are off. In their place
`use_asian_range` adds the prior 20:00-00:00 NY Asian range, whose stops the London
killzone raids first. The opening range is re-anchored from 09:30 NY (equities cash
open) to 03:00 NY (London open).

`use_asian_range` defaults to **False** in `Model2022.DEFAULT_CONFIG`, so index-futures
runs are unaffected.

**Price-scaled params** - `eq_tolerance_pts`, `max_stop_pts` and `stop_fixed_pts` are
index points on NQ/ES/YM and must be restated in price units for FX. Values were
measured on each pair's own 2024-2026 data (`eq_tolerance` ~ 0.5x median 15m bar
range, `max_stop` ~ 2.5x median 15m ATR14):

| Param | NQ | EURUSD | GBPUSD | USDJPY |
|---|---|---|---|---|
| `eq_tolerance_pts` | 5.0 | 0.00025 (2.5p) | 0.00030 (3.0p) | 0.035 (3.5p) |
| `max_stop_pts` | 20.0 | 0.0013 (13p) | 0.0017 (17p) | 0.20 (20p) |
| `dollars_per_point` | 20.0 | 100000 | 100000 | 640 |

Sanity check: as a fraction of price these land at 11-13 bps, against NQ's
20pt/20000 = 10 bps.

`dollars_per_point` is a backtest-runner key (see `DOLLARS_PER_POINT` in
`fvg_sweep_backtest.py`); FX values assume **1 standard lot** (100,000 base units), so
1 pip = $10 on the USD-quoted pairs. USDJPY is JPY-quoted and therefore rate-dependent
- 640 approximates 100000/156 and should be revisited if the rate moves materially.

**Windows note**: these scripts print non-ASCII characters, which the default cp1252
console cannot encode. Prefix runs with `PYTHONIOENCODING=utf-8`.

---

## 3. Automated grid search (Option B)

Edit `PARAM_GRID` in `backtests/grid_search.py` to define which parameters to sweep and what values to try. Uses `itertools.product` for the full cross-product. Data is loaded once and reused.

```bash
.venv/bin/python backtests/grid_search.py
.venv/bin/python backtests/grid_search.py --sort sharpe --out results/my_grid.csv
```

**Warning**: grid explodes fast. 2 bias × 3 stop modes × 4 max_stop × 2 PDH × 2 ONS × 2 OR × 2 REH = 192 runs. Use `BASE_CONFIG` to fix parameters that are already decided, and keep `PARAM_GRID` to the axes you're actively investigating.
