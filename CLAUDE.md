# ICT Trading — Claude Context

## Project Overview

Futures trading research project implementing ICT (Inner Circle Trader) concepts on ES, NQ, and YM continuous contracts. The workflow is:
1. Extract a trading idea (from video/notes) using `/strategy-from-transcript`
2. Formalize rules into a model class (following the pattern below)
3. Backtest the model against historical 1m data
4. Analyze results and iterate

---

## Data

**Files** (all in `/Data/`):
| File | Instrument | Rows | Date Range |
|---|---|---|---|
| `es_futures_1m_cleaned.csv` | S&P 500 (ES) | ~5.4M | 2010-06-06 → 2025-11-17 |
| `nq_futures_1m_cleaned.csv` | Nasdaq (NQ) | ~5.2M | 2010-06-06 → 2025-11-17 |
| `ym_futures_1m_cleaned.csv` | Dow Jones (YM) | ~5.2M | 2010-06-06 → 2025-11-17 |

**Schema**: `timestamp, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol`

**Timestamps**: UTC. Market sessions in UTC:
- Asia: 18:00 – 00:00 (previous day's evening)
- London: 03:00 – 08:00
- New York: 13:30 – 20:00 (RTH); 08:00 – 13:30 (pre-market)
- Regular Trading Hours (RTH): 13:30 – 20:00 UTC

**Version control**: the cleaned CSVs and `raw/` downloads are **gitignored** (~1.1 GB, regenerable from the `Data/prep/` databento pipeline); `Data/prep/.env` (API key) is never committed; only `Data/prep/` is tracked. A 1000-row NQ slice lives in `infrastructure/tests/fixtures/` so tests run without the full dataset.

**Loading data** — use `DataLoader` from the installed `infrastructure` package (`pip install -e .` once):

```python
from infrastructure import DataLoader

loader = DataLoader(timeframe='5T', weeks=52, data_dir='Data')   # last 52 weeks of 5m bars
nq = loader.read_NQ()   # returns DataFrame with DatetimeIndex
es = loader.read_ES()
ym = loader.read_YM()

# Timeframe strings: '1T'=1m, '5T'=5m, '15T'=15m, '1H'=1h, '4H'=4h, 'D'=daily
# (legacy 'T'/'H' aliases are normalized internally for pandas >= 2.2)
```

`DataLoader` has no reliable default location — pass `data_dir` pointing at the repo's `Data/`. Scripts now live deep inside concept/model folders, so resolve the repo root by walking up to `pyproject.toml` (depth-independent) rather than counting `../`:
```python
import pathlib
_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
loader = DataLoader(timeframe='5T', data_dir=str(_ROOT / 'Data'))
```

---

## Project Structure

Key directories (`ls` the repo for the full tree):
- `concepts/` — **colocated semantic + code layer**: one folder per concept under `concepts/<author>/<category>/<name>/` (author = `ict`/`romeo`/`merchant`). **Everything for that concept lives in its folder**: spec (`<slug>.md`, markdown + YAML frontmatter), implementation (`<name>.py`), and whatever exists of `tests/`, `analysis/`, `backtests/`, `charts/`, `configs/`, `examples/`, `results/` (gitignored). Un-coded concepts are spec-only folders. The `concepts` package re-exports every detector's public API flat: `from concepts import find_fvgs, SwingPointScanner`.
- `models/` — the **model layer** (full strategies): `models/intermediate/<name>/` + `models/{ict,romeo,merchant}/<name>/`, each a folder holding `<name>.py` + its spec + the same colocated `tests/`/`analysis/`/`backtests/`/`configs/`/`examples/`/`results/` subfolders (see `models/README.md`). Un-built models are spec-only folders.
- `infrastructure/` — installable shared machinery (flat package): `registry.py` (`@concept` + lineage), `loader.py` (`DataLoader`), `report.py` (backtest report engine — `Mark`/`TradeViz`/`build_report`), `time_utils.py`. Also holds the repo-wide CLIs `lineage.py` / `fetch_dxy.py`, the reference docs `CHARTING.md` / `CHARTS.md`, and the cross-cutting test suite in `infrastructure/tests/` (+ `fixtures/`). Import: `from infrastructure import DataLoader`, `from infrastructure.registry import concept`.
- `strategies/notes/` — strategy write-ups · `analysis/` — *cross-cutting* findings only (per-concept findings live in that concept's `analysis/` folder; see `analysis/README.md`)
- `legacy/` (NOT imported). Example/chart/analysis scripts are colocated in each concept's/model's own `examples/`, `charts/`, `analysis/` subfolders — there is no top-level `examples/`.

Scripts inside concept/model folders resolve the repo root by walking up to `pyproject.toml` (`_ROOT`/`_HERE`), so they're depth-independent; backtest configs sit in `<folder>/backtests/configs/` and outputs in `<folder>/backtests/results/`.

**Three-layer architecture** — enforced by where files live:

| Layer | Location | Rule |
|---|---|---|
| **Concept** | `concepts/<author>/<category>/<name>/` | Implements one KB concept (a multi-slug module may carry several specs); no imports from other concepts except via `from concepts import ...` |
| **Intermediate** | `models/intermediate/<name>/` | Composes 2+ concepts into a reusable gate/signal (e.g. `daily_bias`) |
| **Model** | `models/<author>/<name>/` | Full trading strategy; composes concepts + intermediate models |

Each concept/model's `<slug>.md` is the **semantic layer**: one spec per concept, each with
frontmatter (`name`, `aliases`, `category`, `related`, `parameters`, `detection`) and a
"Detection Rules" section, sitting in the same folder as the code that implements it.
`infrastructure/registry.py` binds spec to code (it globs specs from `concepts/` and `models/`):

- Register a detector with `@concept("<slug>")`.
- Declare dependencies with `@concept("<slug>", depends_on=["other-slug", ...])` — this powers the lineage graph.
- `infrastructure/tests/test_registry_coverage.py` fails if a concept's `detection:` flag and registered detectors disagree.
- Pull default config from frontmatter with `infrastructure.registry.params("<slug>")`.
- Query the dependency tree: `infrastructure.registry.lineage("<slug>")` returns the transitive dict; `infrastructure.registry.mermaid("<slug>")` returns a Mermaid string.
- Regenerate the README diagram: `.venv/bin/python infrastructure/lineage.py --update-readme`.

> **Code and semantic layer stay in sync — always.** The `<slug>.md` spec (colocated in
> the concept's folder) is the spec; the `<name>.py` beside it is its implementation.
> **Whenever a detector's logic changes — especially a correction the user gives to
> detection rules, thresholds, timing, or a pool/validity condition — update the matching
> `<slug>.md` in the same folder, in the *same* change.**
> Reflect the new rule in the "Detection Rules" section (and `parameters:` frontmatter if
> a value changed). A code fix without the corresponding knowledge update is incomplete and
> leaves the spec lying. If a correction spans concepts (e.g. a liquidity rule touching the
> sweep, the opening range, and a model), update every affected file.

---

## Model Pattern & Adding a New Strategy

The `MarketSnapshot`/`generate_signal` model template and the step-by-step workflow for adding a strategy live in **`models/README.md`** — read it before writing a new model.

**Rule of exclusion** (applies to every model): if any required component is missing, `actionable=False` and no trade is taken. All checklist items must pass simultaneously.

---

## ICT Concepts Glossary

- **SMT divergence**: One correlated index makes a new extreme while others fail — signals reversal intent. Requires all three instruments (ES, NQ, YM) in the `correlated` dict.
- **Stop hunt / liquidity sweep**: Price sweeps beyond a prior high/low then reverses quickly.
- **Fair Value Gap (FVG)**: Three-candle imbalance: body of candle 1 and body of candle 3 don't overlap. Acts as magnet/reaction zone.
- **Draw on liquidity**: The next untested high/low pool price is attracted toward.
- **Market structure shift**: Break of a significant swing high/low on lower timeframe after a stop hunt.
- **OHLC / OLHC candle expectation**: Bearish day = Open→High→Low→Close sequence; Bullish = Open→Low→High→Close.
- **Devil's mark**: A prior flat open level that acts as a magnet.
- **Session liquidity pools**: Asia range high/low, London high/low, pre-market high/low — common stop hunt targets.

---

## Backtesting Conventions

**Framework**: Custom event loop (not backtesting.py or vectorbt — ICT models require multi-instrument simultaneous evaluation that those libraries don't handle cleanly).

**Position sizing**: Fixed 1 contract per trade. Instrument tick values:
- ES (micro MES): 1 tick = 0.25 pts = $1.25 | ES full: $12.50
- NQ (micro MNQ): 1 tick = 0.25 pts = $0.50 | NQ full: $5.00
- YM (micro MYM): 1 tick = 1 pt = $0.50 | YM full: $5.00

**Direction**: Both long and short signals are tested.

**Standard backtest outputs** (all models should produce):
1. Trade log CSV: `entry_time, exit_time, direction, entry_price, exit_price, stop, target, pnl, exit_reason`
2. Stats summary: total P&L, win rate, avg win/loss, profit factor, max drawdown, Sharpe ratio
3. Session breakdown: same stats sliced by Asia / London / NY
4. Equity curve plot

**Backtesting location**: runner scripts live in the `backtests/` subfolder of the concept/model they test (e.g. `models/ict/model_2022/backtests/fvg_sweep_backtest.py`); outputs are written to that folder's `backtests/results/` (gitignored — the `results/` ignore matches at any depth). A reusable engine, when extracted, belongs in `infrastructure/`.

**Charts**: after saving any chart to `/tmp/`, always open it immediately with `subprocess.run(['open', path])` (or `os.system(f'open {path}')`) so it launches in Preview without the user having to ask.

**Walk-forward approach**: Iterate day by day. Compute daily bias once at session open. Apply intraday signals on 5m bars within that day's session window. One trade per day (the model's rule of exclusion limits entries).

---

## Key Reference: Existing Plan

`.github/prompts/plan-nqBacktestFramework.prompt.md` contains the detailed backtest framework design — consult it when building or extending the backtesting engine.

---

## Experiment Infrastructure

Model config flags (`Model2022.DEFAULT_CONFIG`), named YAML configs, single-run backtests, `compare_runs.py`, and the `grid_search.py` sweep are all documented in **`models/ict/model_2022/backtests/configs/README.md`** — read it before running a config or grid search. (The configs and their runner now live with the model_2022 backtests.)

---

## Analysis Documents

Quantitative backtest findings for a concept live in that concept's own `analysis/` folder (e.g. `concepts/ict/pd_arrays/fair_value_gap/analysis/`); cross-cutting findings live in the top-level `analysis/`. The naming rule, required frontmatter, required sections, the never-edit-existing-files rule, and the what-goes-where table are in **`analysis/README.md`** — read and follow it whenever you write or update an analysis doc. Always add a row to that file's index table when you create a new analysis file.
