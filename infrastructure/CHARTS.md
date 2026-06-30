# Charts

Charting scripts and their image outputs are **colocated with the concept they
visualize**, not here. Each concept/model folder has its own `charts/` subfolder:

```
concepts/<author>/<category>/<name>/charts/<chart_script>.py
concepts/<author>/<category>/<name>/charts/<output-subdir>/<name>-<instrument>-<date>.png
models/<author>/<name>/charts/...
```

Examples:
```
concepts/ict/pd_arrays/order_block/charts/chart_ob.py                         → charts/ob/ob-DXY-1h-<date>.png
concepts/ict/daily_bias/dxy_weekly_bias/charts/chart_dxy_mss.py               → dxy-weekly-bias-DXY-<date>.png (beside script)
concepts/ict/market_structure/market_structure_shift/charts/chart_mss_type2.py → mss-type2-<date>.png (beside script)
concepts/merchant/price_action/break_away_gap/charts/chart_dxy_bag.py
```

Run a chart from the repo root, e.g.:
```
python concepts/ict/pd_arrays/order_block/charts/chart_ob.py
```
Each script resolves the repo root itself (walks up to `pyproject.toml`) and
writes its PNG next to itself.

## Naming

`<concept>-<instrument>-<date_from_instrument>.png` — where the date is the
**start date of the data shown**, not today. (`{instrument}` = `DXY`/`NQ`/`ES`/…)

## Style conventions

Follow [`infrastructure/CHARTING.md`](CHARTING.md) for all style rules (colors,
raid lines, dark background, etc.). Quick exploratory charts go to `/tmp/` first.
