"""
CISD chart — finds a clean bullish CISD example on NQ 5m and plots it.
Run: python concepts/ict/market_structure/change_in_state_of_delivery/charts/chart_cisd.py
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

from infrastructure import DataLoader
from concepts import SwingPointScanner, detect_cisd

DATA_DIR = str(_ROOT / 'Data')

# ── Load 5m NQ, last 4 weeks ──────────────────────────────────────────────────
loader = DataLoader(timeframe='5T', weeks=4, data_dir=DATA_DIR)
nq = loader.read_NQ()

# ── Hunt for a bullish CISD ───────────────────────────────────────────────────
# Strategy: scan each day, find swing lows, try CISD from each.
found = None
all_days = sorted(nq.index.normalize().unique(), reverse=True)

for day in all_days:
    day_bars = nq[nq.index.normalize() == day]
    if len(day_bars) < 30:
        continue

    scanner = SwingPointScanner(day_bars, lookback=2)
    scanner.identify_swings()
    if scanner.swing_lows is None or len(scanner.swing_lows) == 0:
        continue

    for sw_time in reversed(scanner.swing_lows.index.tolist()):
        cisd = detect_cisd(day_bars, direction='bullish',
                           sweep_time=sw_time,
                           max_leg_lookback=10,
                           max_bars_after_sweep=30)
        if cisd is None:
            continue
        # Require a meaningful multi-candle leg and at least 2 bars gap to CISD
        if cisd['leg_candle_count'] < 3:
            continue
        sw_idx   = day_bars.index.get_indexer([sw_time], method='nearest')[0]
        cisd_gap = cisd['bar_index'] - sw_idx
        if cisd_gap < 2:
            continue

        start  = max(0, sw_idx - 25)
        end    = min(len(day_bars), cisd['bar_index'] + 20)
        window_bars = day_bars.iloc[start:end]
        if len(window_bars) < 15:
            continue

        found = dict(bars=window_bars, sweep_time=sw_time, cisd=cisd)
        break

    if found:
        break

if found is None:
    print("No CISD found in the lookback window — try increasing weeks= in DataLoader.")
    sys.exit(1)

bars       = found['bars']
sweep_time = found['sweep_time']
cisd       = found['cisd']

sw_idx     = bars.index.get_indexer([sweep_time], method='nearest')[0]
sweep_low  = bars.iloc[sw_idx]['low']

print(f"Swing low : {sweep_time}  low={sweep_low:.2f}")
print(f"CISD fire : {cisd['cisd_time']}  close above {cisd['reference_level']:.2f}  "
      f"(leg={cisd['leg_candle_count']} candles)")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG      = '#0d1117'
BULL    = '#26a641'
BEAR    = '#f85149'
REF_C   = '#e3b341'   # gold — reference level
CISD_C  = '#58a6ff'   # blue — CISD trigger line

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.grid(False)

xs = np.arange(len(bars))
idx_map    = {ts: i for i, ts in enumerate(bars.index)}
sweep_x    = idx_map.get(sweep_time, bars.index.get_indexer([sweep_time], method='nearest')[0])

# Candles
W_BODY = 0.4
W_WICK = 0.05
for i, (ts, row) in enumerate(bars.iterrows()):
    is_bull = row['close'] >= row['open']
    color   = BULL if is_bull else BEAR
    body_lo = min(row['open'], row['close'])
    body_hi = max(row['open'], row['close'])
    ax.add_patch(plt.Rectangle((i - W_BODY/2, body_lo), W_BODY, body_hi - body_lo,
                                color=color, zorder=3))
    ax.plot([i, i], [row['low'], body_lo],  color=color, lw=1, zorder=2)
    ax.plot([i, i], [body_hi, row['high']], color=color, lw=1, zorder=2)

# Reference level (horizontal dashed)
ref = cisd['reference_level']
# Find where leg starts (first bar before sweep that is still bearish)
leg_start_x = sweep_x
for i in range(sweep_x, max(sweep_x - cisd['leg_candle_count'] - 1, 0) - 1, -1):
    if bars.iloc[i]['close'] < bars.iloc[i]['open']:
        leg_start_x = i
    else:
        break

ax.hlines(ref, leg_start_x, len(bars) - 1, colors=REF_C, lw=1.2,
          linestyles='--', label=f'CISD reference high  {ref:.2f}', zorder=4)
ax.text(len(bars) - 0.5, ref, f' {ref:.2f}', color=REF_C, va='center', fontsize=8)

# Sweep marker
if 0 <= sweep_x < len(bars):
    ax.annotate('swing low', xy=(sweep_x, bars.iloc[sweep_x]['low']),
                xytext=(sweep_x, bars.iloc[sweep_x]['low'] - (bars['high'].max() - bars['low'].min()) * 0.04),
                color=BEAR, fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color=BEAR, lw=1))

# CISD trigger bar
cisd_x = idx_map.get(cisd['cisd_time'])
if cisd_x is not None:
    ax.axvline(cisd_x, color=CISD_C, lw=1.2, linestyle=':', alpha=0.9,
               label=f'CISD close  {bars.iloc[cisd_x]["close"]:.2f}')
    ax.annotate('CISD', xy=(cisd_x, bars.iloc[cisd_x]['close']),
                xytext=(cisd_x + 1, bars.iloc[cisd_x]['close'] +
                        (bars['high'].max() - bars['low'].min()) * 0.025),
                color=CISD_C, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CISD_C, lw=1))

# Bar-index tick every 5
tick_xs    = xs[::5]
tick_labels = [bars.index[i].strftime('%H:%M') for i in tick_xs]
ax.set_xticks(tick_xs)
ax.set_xticklabels(tick_labels, color='#8b949e', fontsize=7, rotation=45)
ax.tick_params(axis='y', colors='#8b949e', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

ax.set_title(f'NQ 5m — Bullish CISD  ({bars.index[0].strftime("%Y-%m-%d")})',
             color='#e6edf3', fontsize=12, pad=10)
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=8)

out = '/tmp/cisd_chart.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved → {out}")
