"""
DXY Weekly Bias — Example Charts
=================================
Plots 6 example signals using the best parameters:
  swing_lookback=3, reaction_bars=2, reaction_type=mss

Each panel shows DXY weekly candles with:
  - Pool level (dashed gold) — the failure swing being watched
  - Raid candle (red/green X) — when DXY takes out the pool
  - MSS break level (dotted) — prior swing that gets broken
  - MSS confirmation bar (arrow) — the bar that confirms direction
  - NQ direction that week (correct ✓ / incorrect ✗)

Usage:
    python scripts/chart_dxy_weekly_bias.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ict import DataLoader
from ict.concepts.market_structure import SwingPointScanner, detect_mss
from ict.concepts.pool_validity import classify_swing_liquidity

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))

# ── Best params ───────────────────────────────────────────────────────────────
SW_LB        = 3
RX_BARS      = 2
MAX_REACTION = 4
PRE_BARS     = 20
WARMUP       = 52
N_EXAMPLES   = 6

# ── Load data ─────────────────────────────────────────────────────────────────
loader = DataLoader(timeframe='D', data_dir=DATA_DIR)
dxy_daily = loader.read_DXY('daily')
dxy_weekly = (
    dxy_daily.resample('W')
    .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    .dropna(subset=['open'])
)
dxy_weekly.index = dxy_weekly.index.tz_localize(None)

nq_loader = DataLoader(timeframe='W', data_dir=DATA_DIR)
nq_weekly = nq_loader.read_NQ()
nq_weekly.index = nq_weekly.index.tz_localize(None)

common = dxy_weekly.index.intersection(nq_weekly.index)
dxy_weekly = dxy_weekly.loc[common]
nq_weekly  = nq_weekly.loc[common]
nq_dir     = (nq_weekly['close'] > nq_weekly['open']).map({True: 'bullish', False: 'bearish'})

bar_index = {ts: i for i, ts in enumerate(dxy_weekly.index)}

# ── Pre-compute raids (same logic as backtest) ────────────────────────────────
scanner = SwingPointScanner(dxy_weekly, lookback=SW_LB)
scanner.identify_swings()
cl_highs = classify_swing_liquidity(scanner.swing_highs, 'high')
cl_lows  = classify_swing_liquidity(scanner.swing_lows,  'low')
vh = cl_highs[cl_highs['classification'] == 'failure_swing']
vl = cl_lows[cl_lows['classification']  == 'failure_swing']

raids = []
for side, swings, price_col, dxy_dir in [
    ('high', vh, 'swing_high_price', 'bearish'),
    ('low',  vl, 'swing_low_price',  'bullish'),
]:
    for ts, row in swings.iterrows():
        level    = row[price_col]
        pool_bar = bar_index.get(ts)
        if pool_bar is None:
            continue
        future = dxy_weekly.iloc[pool_bar + 1:]
        raided = future[future['high'] > level] if side == 'high' else future[future['low'] < level]
        if len(raided) == 0:
            continue
        raid_time = raided.index[0]
        raid_bar  = bar_index[raid_time]
        if raid_bar < WARMUP:
            continue

        start  = max(0, raid_bar - PRE_BARS)
        end    = min(raid_bar + MAX_REACTION + 1, len(dxy_weekly))
        window = dxy_weekly.iloc[start:end]

        mss = detect_mss(window, direction=dxy_dir, swing_lookback=1,
                         max_bars_after_sweep=MAX_REACTION, sweep_time=raid_time)
        if mss is None:
            continue
        mss_bar = bar_index.get(mss['break_time'])
        if mss_bar is None:
            continue
        offset = mss_bar - raid_bar
        if not (1 <= offset <= RX_BARS):
            continue

        eq_bias = 'bullish' if dxy_dir == 'bearish' else 'bearish'

        raids.append({
            'pool_side':   side,
            'pool_level':  level,
            'pool_time':   ts,
            'pool_bar':    pool_bar,
            'raid_time':   raid_time,
            'raid_bar':    raid_bar,
            'dxy_dir':     dxy_dir,
            'eq_bias':     eq_bias,
            'mss_time':    mss['break_time'],
            'mss_bar':     mss_bar,
            'mss_level':   mss['broken_level'],
        })

# Score each signal's regime accuracy now that we have the full raid list
all_raid_bars = sorted(r['raid_bar'] for r in raids)
for r in raids:
    later = [b for b in all_raid_bars if b > r['raid_bar']]
    regime_end   = min(later) if later else len(dxy_weekly)
    regime_weeks = dxy_weekly.index[r['mss_bar'] + 1 : regime_end]
    regime_nq    = nq_dir.reindex(regime_weeks).dropna()
    r['n_regime']  = len(regime_nq)
    r['n_correct'] = int((regime_nq == r['eq_bias']).sum())
    r['accuracy']  = r['n_correct'] / r['n_regime'] if r['n_regime'] else 0.0

n_total   = len(raids)
n_correct = sum(1 for r in raids if r['accuracy'] >= 0.5)
print(f"Found {n_total} signals. {n_correct}/{n_total} had >50% regime accuracy. Picking {N_EXAMPLES} examples.")

# Pick a spread across time and accuracy levels
examples = sorted(raids, key=lambda r: r['raid_bar'])[:N_EXAMPLES]

# ── Chart ─────────────────────────────────────────────────────────────────────
BG     = '#0d1117'
BULL   = '#26a641'
BEAR   = '#f85149'
GOLD   = '#e3b341'
BLUE   = '#58a6ff'
PURPLE = '#bc8cff'
GREY   = '#8b949e'

CONTEXT_BEFORE = 18   # weekly bars before pool to show
CONTEXT_AFTER  = 6    # weekly bars after MSS to show

fig, axes = plt.subplots(2, 3, figsize=(30, 16))
fig.patch.set_facecolor(BG)
fig.suptitle('DXY Weekly Bias — MSS Signal Examples  (sw_lb=3, rx_bars=2)',
             color='#e6edf3', fontsize=18, y=1.01)

def draw_candles(ax, bars):
    W = 0.4
    for i, (ts, row) in enumerate(bars.iterrows()):
        bull = row['close'] >= row['open']
        c    = BULL if bull else BEAR
        lo   = min(row['open'], row['close'])
        hi   = max(row['open'], row['close'])
        ax.add_patch(plt.Rectangle((i - W/2, lo), W, hi - lo, color=c, zorder=3))
        ax.plot([i, i], [row['low'],  lo], color=c, lw=0.8, zorder=2)
        ax.plot([i, i], [row['high'], hi], color=c, lw=0.8, zorder=2)

for ax, ex in zip(axes.flat, examples):
    ax.set_facecolor(BG)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.tick_params(colors=GREY, labelsize=7)

    # Slice the display window
    start_bar = max(0, ex['pool_bar'] - CONTEXT_BEFORE)
    end_bar   = min(len(dxy_weekly), ex['mss_bar'] + CONTEXT_AFTER + 1)
    bars      = dxy_weekly.iloc[start_bar:end_bar]
    local     = {ts: i for i, ts in enumerate(bars.index)}

    draw_candles(ax, bars)
    n = len(bars)

    pool_x  = local.get(ex['pool_time'])
    raid_x  = local.get(ex['raid_time'])
    mss_x   = local.get(ex['mss_time'])

    # Pool level — dashed gold from pool bar to right edge
    if pool_x is not None:
        ax.hlines(ex['pool_level'], pool_x, n - 1,
                  colors=GOLD, lw=1.2, linestyles='--', zorder=4, alpha=0.9)
        ax.text(pool_x, ex['pool_level'],
                f" {'H' if ex['pool_side']=='high' else 'L'} {ex['pool_level']:.2f}",
                color=GOLD, fontsize=10, va='bottom')

    # MSS break level — dotted blue
    if mss_x is not None:
        ax.hlines(ex['mss_level'], 0, mss_x,
                  colors=BLUE, lw=1.0, linestyles=':', zorder=4, alpha=0.8)

    # Raid candle marker
    if raid_x is not None:
        raid_row = bars.iloc[raid_x]
        y = raid_row['low'] if ex['dxy_dir'] == 'bearish' else raid_row['high']
        offset = -(bars['high'].max() - bars['low'].min()) * 0.035
        marker = 'v' if ex['dxy_dir'] == 'bearish' else '^'
        ax.plot(raid_x, y + offset * (1 if ex['dxy_dir']=='bearish' else -1),
                marker=marker, color=BEAR if ex['dxy_dir']=='bearish' else BULL,
                ms=8, zorder=5)
        ax.text(raid_x, y + offset * 2.2, 'RAID', color=GREY, fontsize=9,
                ha='center', va='top' if ex['dxy_dir']=='bearish' else 'bottom')

    # MSS confirmation arrow
    if mss_x is not None:
        mss_row = bars.iloc[mss_x]
        rng = bars['high'].max() - bars['low'].min()
        if ex['dxy_dir'] == 'bearish':
            ax.annotate('MSS', xy=(mss_x, mss_row['close']),
                        xytext=(mss_x + 0.8, mss_row['close'] - rng * 0.08),
                        color=BEAR, fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=BEAR, lw=1.5))
        else:
            ax.annotate('MSS', xy=(mss_x, mss_row['close']),
                        xytext=(mss_x + 0.8, mss_row['close'] + rng * 0.08),
                        color=BULL, fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=BULL, lw=1.5))

    # X-axis: label every 4th bar
    xs = np.arange(n)
    tick_xs = xs[::4]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([bars.index[i].strftime('%b\'%y') for i in tick_xs],
                       color=GREY, fontsize=9, rotation=45)
    ax.set_xlim(-0.5, n - 0.5)
    ax.tick_params(axis='y', labelsize=9)

    # Title
    acc_pct = ex['accuracy'] * 100
    color   = '#3fb950' if acc_pct >= 60 else (GOLD if acc_pct >= 50 else BEAR)
    ax.set_title(
        f"DXY {ex['dxy_dir']} MSS  [{ex['raid_time'].strftime('%Y-%m-%d')}]"
        f"   → NQ {ex['eq_bias']}"
        f"   regime acc: {acc_pct:.0f}% ({ex['n_correct']}/{ex['n_regime']} wks)",
        color=color, fontsize=12, pad=8
    )

# Legend
legend_items = [
    mpatches.Patch(color=GOLD,   label='Failure-swing pool'),
    mpatches.Patch(color=BLUE,   label='MSS break level'),
    mpatches.Patch(color=BEAR,   label='Bearish raid / MSS'),
    mpatches.Patch(color=BULL,   label='Bullish raid / MSS'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
           fontsize=8, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
out = '/tmp/dxy_weekly_bias_examples.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved → {out}")
