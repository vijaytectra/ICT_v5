# ICT Trading Research

Futures trading research project implementing ICT (Inner Circle Trader) concepts on ES, NQ, and YM continuous contracts.

## What's in here

| Layer | Location | Purpose |
|---|---|---|
| **Concepts** | `concepts/` | One folder per concept — spec (markdown + YAML frontmatter), code, tests, analysis, charts, backtests colocated |
| **Models** | `models/` | The strategy layer — full models + intermediate gates, each a folder with code + spec + backtests/configs/results/examples |
| **Infrastructure** | `infrastructure/` | Installable shared machinery: `DataLoader`, the `@concept` registry, the backtest report engine, time utils |
| **Strategy notes** | `strategies/notes/` | Human write-ups extracted from video/PDF sources |
| **Data** | `Data/` | 1m continuous contracts for ES, NQ, YM (~1.1 GB, gitignored) |

## Setup

```bash
pip install -e .
```

Data files are not in the repo. Run the acquisition pipeline in `Data/prep/` with a Databento API key.

## Quick start

```python
from infrastructure import DataLoader

loader = DataLoader(timeframe='5T', weeks=52, data_dir='Data')
nq = loader.read_NQ()   # returns DataFrame with DatetimeIndex
es = loader.read_ES()
ym = loader.read_YM()
```

Timeframes: `'1T'` `'5T'` `'15T'` `'1H'` `'4H'` `'D'`

## Running a backtest

```bash
python backtests/fvg_sweep_backtest.py
# → results/fvg_sweep_<date>.html  (interactive Plotly report)
```

## Concepts (semantic + code layer)

Each concept lives in its own folder under `concepts/<author>/<category>/<name>/`, colocating its `<slug>.md` spec (machine-parseable YAML frontmatter + Detection Rules — the spec for the code beside it) with its `<name>.py` implementation, `tests/`, and `analysis/`. Slugs are bound to detector code via the `@concept("<slug>")` decorator; the registry (`infrastructure/registry.py`) globs specs from `concepts/` and `models/`. Import any detector flat: `from concepts import find_fvgs, SwingPointScanner`.

Currently implemented detectors: `swing-points`, `fair-value-gap`, `daily-bias`, `model-2022` (and more — see the lineage graph below).

See [concepts/ict/README.md](concepts/ict/README.md) for the ICT concept index.

## Models

| Model | File | Status |
|---|---|---|
| ICT 2022 (FVG sweep) | `models/ict/model_2022/` | backtested |
| Sniper Model | `models/ict/sniper_model/` | implemented |
| Silver Bullet | `models/ict/silver_bullet/` | spec only |
| Unicorn | `models/ict/unicorn/` | spec only |
| Three Indians / Wolfe Wave | `models/ict/three_indians/` | spec only |

## Concept Lineage

How concepts compose into models. Re-generate after adding new concepts:

```bash
python infrastructure/lineage.py                  # print full diagram
python infrastructure/lineage.py model-2022       # single model's graph
python infrastructure/lineage.py --update-readme  # regenerate the section below
```

<!-- lineage-start -->
```mermaid
flowchart LR
    subgraph Concepts
        break_away_gap["break-away-gap"]
        change_in_state_of_delivery["change-in-state-of-delivery"]
        dark_pool["dark-pool"]
        draw_on_liquidity["draw-on-liquidity"]
        fair_value_gap["fair-value-gap"]
        killzones["killzones"]
        liquidity_sweep_stop_hunt["liquidity-sweep-stop-hunt"]
        market_structure_shift["market-structure-shift"]
        ohlc_candle_profiles["ohlc-candle-profiles"]
        order_block["order-block"]
        premium_discount["premium-discount"]
        relative_equal_highs_lows["relative-equal-highs-lows"]
        sessions_and_ranges["sessions-and-ranges"]
        smt_divergence["smt-divergence"]
        swing_points["swing-points"]
        swing_without_liquidity["swing-without-liquidity"]
        targets_and_exits["targets-and-exits"]
    end
    subgraph Intermediate
        daily_bias["daily-bias"]
    end
    subgraph Models
        model_2022["model-2022"]
        sniper_model["sniper-model"]
    end

    fair_value_gap --> break_away_gap
    swing_without_liquidity --> break_away_gap
    swing_points --> change_in_state_of_delivery
    draw_on_liquidity --> daily_bias
    premium_discount --> daily_bias
    ohlc_candle_profiles --> daily_bias
    swing_without_liquidity --> dark_pool
    fair_value_gap --> dark_pool
    swing_points --> draw_on_liquidity
    swing_points --> market_structure_shift
    break_away_gap --> market_structure_shift
    dark_pool --> market_structure_shift
    daily_bias --> model_2022
    fair_value_gap --> model_2022
    killzones --> model_2022
    liquidity_sweep_stop_hunt --> model_2022
    market_structure_shift --> model_2022
    relative_equal_highs_lows --> model_2022
    sessions_and_ranges --> model_2022
    targets_and_exits --> model_2022
    swing_points --> relative_equal_highs_lows
    daily_bias --> sniper_model
    liquidity_sweep_stop_hunt --> sniper_model
    smt_divergence --> sniper_model
    market_structure_shift --> sniper_model
```
<!-- lineage-end -->

## Disclaimer

For educational and research purposes only. Trading involves substantial risk of loss.
