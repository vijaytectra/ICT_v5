"""
DXY Weekly Bias — Interactive Regime Chart (Plotly HTML)
=========================================================
Best params: ltc=ob, sw_lb=1, rel_tol=0.005, rx_bars=1, rx_type=mss, leg_lb=10

Shows DXY weekly candles + NQ weekly close with:
  - Regime shading (green=NQ bullish bias, red=NQ bearish bias)
  - Pool level lines (dashed) at each raid
  - Raid markers
  - MSS confirmation markers
  - Per-regime accuracy annotation

Usage:
    python analysis/dxy-weekly-bias/dxy_bias_regime_chart.py
"""

import os
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ict import DataLoader
from ict.concepts.market_structure import SwingPointScanner, detect_mss
from ict.concepts.hpfs import find_hpfs

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Data'))

# ── Best params ───────────────────────────────────────────────────────────────
LTC_TYPE      = "ob"
SW_LB         = 1
REL_TOL       = 0.005
RX_BARS       = 1
RX_TYPE       = "mss"
LEG_LB        = 10
HPFS_SCAN_FWD = 6
PRE_RAID_BARS = 20
MAX_REACTION  = 4
WARMUP_WEEKS  = 52

# ── Load data ─────────────────────────────────────────────────────────────────
loader = DataLoader(timeframe="D", data_dir=DATA_DIR)
dxy_daily = loader.read_DXY("daily")
dxy_weekly = (
    dxy_daily.resample("W")
    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    .dropna(subset=["open"])
)
dxy_weekly.index = dxy_weekly.index.tz_localize(None)

nq_loader = DataLoader(timeframe="W", data_dir=DATA_DIR)
nq_weekly = nq_loader.read_NQ()
nq_weekly.index = nq_weekly.index.tz_localize(None)

common = dxy_weekly.index.intersection(nq_weekly.index)
dxy_weekly = dxy_weekly.loc[common]
nq_weekly  = nq_weekly.loc[common]
nq_dir     = (nq_weekly["close"] > nq_weekly["open"]).map({True: "bullish", False: "bearish"})

bar_index = {ts: i for i, ts in enumerate(dxy_weekly.index)}
bars_ts   = list(dxy_weekly.index)

# ── Replicate raid precomputation (best params only) ──────────────────────────
def make_raid_record(pool_side, pool_level, pool_time, raid_time, raid_bar, dxy_dir, source):
    start  = max(0, raid_bar - PRE_RAID_BARS)
    end    = min(raid_bar + MAX_REACTION + 1, len(dxy_weekly))
    window = dxy_weekly.iloc[start:end]
    if len(window) < PRE_RAID_BARS // 2:
        return None
    mss_bar_offset = None
    mss = detect_mss(window, direction=dxy_dir, swing_lookback=1,
                     max_bars_after_sweep=MAX_REACTION, sweep_time=raid_time)
    if mss is not None:
        mb = bar_index.get(mss["break_time"])
        if mb is not None:
            mss_bar_offset = mb - raid_bar
    return {
        "source": source, "pool_side": pool_side, "pool_level": pool_level,
        "pool_time": pool_time, "raid_time": raid_time, "raid_bar": raid_bar,
        "mss_bar_offset": mss_bar_offset,
        "equity_bias": "bullish" if dxy_dir == "bearish" else "bearish",
    }

all_raids = []

# HPFS (ob)
for hpfs_dir, dxy_dir in [("bearish", "bearish"), ("bullish", "bullish")]:
    hpfs_df = find_hpfs(dxy_weekly, direction=hpfs_dir, ltc_type=LTC_TYPE, scan_forward=HPFS_SCAN_FWD)
    for _, row in hpfs_df[hpfs_df["raid_time"].notna()].iterrows():
        raid_time = row["raid_time"]
        raid_bar  = bar_index.get(raid_time)
        if raid_bar is None or raid_bar < WARMUP_WEEKS:
            continue
        rec = make_raid_record(
            "high" if hpfs_dir == "bearish" else "low",
            float(row["hpfs_level"]), row["hpfs_time"], raid_time, raid_bar, dxy_dir, "hpfs"
        )
        if rec:
            all_raids.append(rec)

# REH/REL
scanner = SwingPointScanner(dxy_weekly, lookback=SW_LB)
scanner.identify_swings()
for cluster in scanner.find_relative_equal_highs(tolerance=REL_TOL):
    pool_level = cluster["max_price"]
    pool_time  = sorted(cluster["timestamps"])[-1]
    pool_bar   = bar_index.get(pool_time)
    if pool_bar is None:
        continue
    future = dxy_weekly.iloc[pool_bar + 1:]
    raided = future[future["high"] > pool_level]
    if raided.empty:
        continue
    raid_time = raided.index[0]
    raid_bar  = bar_index[raid_time]
    if raid_bar < WARMUP_WEEKS:
        continue
    rec = make_raid_record("high", pool_level, pool_time, raid_time, raid_bar, "bearish", "reh")
    if rec:
        all_raids.append(rec)

for cluster in scanner.find_relative_equal_lows(tolerance=REL_TOL):
    pool_level = cluster["min_price"]
    pool_time  = sorted(cluster["timestamps"])[-1]
    pool_bar   = bar_index.get(pool_time)
    if pool_bar is None:
        continue
    future = dxy_weekly.iloc[pool_bar + 1:]
    raided = future[future["low"] < pool_level]
    if raided.empty:
        continue
    raid_time = raided.index[0]
    raid_bar  = bar_index[raid_time]
    if raid_bar < WARMUP_WEEKS:
        continue
    rec = make_raid_record("low", pool_level, pool_time, raid_time, raid_bar, "bullish", "rel")
    if rec:
        all_raids.append(rec)

# Deduplicate by (raid_bar, pool_side)
seen = {}
for r in sorted(all_raids, key=lambda x: x["raid_bar"]):
    key = (r["raid_bar"], r["pool_side"])
    if key not in seen:
        seen[key] = r
all_raids = sorted(seen.values(), key=lambda x: x["raid_bar"])

# Filter to confirmed MSS within rx_bars
shifts = []
for r in all_raids:
    o = r["mss_bar_offset"]
    if o is not None and 1 <= o <= RX_BARS:
        confirm_bar = r["raid_bar"] + o
        if confirm_bar >= WARMUP_WEEKS:
            shifts.append({**r, "confirm_bar": confirm_bar, "confirm_time": bars_ts[confirm_bar]})

shifts.sort(key=lambda s: s["confirm_bar"])

# Score regimes
for k, shift in enumerate(shifts):
    regime_start = shift["confirm_bar"] + 1
    regime_end   = shifts[k + 1]["confirm_bar"] if k + 1 < len(shifts) else len(dxy_weekly)
    regime_ts    = bars_ts[regime_start:regime_end]
    regime_nq    = nq_dir.reindex(regime_ts).dropna()
    n_correct    = int((regime_nq == shift["equity_bias"]).sum())
    shift["regime_start_ts"] = bars_ts[regime_start] if regime_start < len(bars_ts) else None
    shift["regime_end_ts"]   = bars_ts[regime_end - 1] if regime_end <= len(bars_ts) else bars_ts[-1]
    shift["n_weeks"]  = len(regime_nq)
    shift["n_correct"] = n_correct
    shift["accuracy"] = n_correct / len(regime_nq) if regime_nq.size else 0.0

print(f"Shifts found: {len(shifts)}")
for s in shifts:
    print(f"  {s['confirm_time'].date()}  {s['equity_bias']:<8}  {s['source']:<4}  "
          f"acc={s['accuracy']*100:.0f}% ({s['n_correct']}/{s['n_weeks']} wks)")

# ── Build Plotly figure ───────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],
    vertical_spacing=0.04,
    subplot_titles=["DXY Weekly (Candles)", "NQ Weekly (Close)"],
)

# DXY candlestick
fig.add_trace(go.Candlestick(
    x=dxy_weekly.index,
    open=dxy_weekly["open"], high=dxy_weekly["high"],
    low=dxy_weekly["low"],   close=dxy_weekly["close"],
    name="DXY", increasing_line_color="#26a641", decreasing_line_color="#f85149",
    increasing_fillcolor="#26a641", decreasing_fillcolor="#f85149",
    line_width=0.8,
), row=1, col=1)

# NQ close line
fig.add_trace(go.Scatter(
    x=nq_weekly.index, y=nq_weekly["close"],
    name="NQ close", line=dict(color="#58a6ff", width=1.2),
    mode="lines",
), row=2, col=1)

# Regime shading
BULL_FILL = "rgba(38,166,65,0.10)"
BEAR_FILL = "rgba(248,81,73,0.10)"

for k, shift in enumerate(shifts):
    if shift["regime_start_ts"] is None:
        continue
    color = BULL_FILL if shift["equity_bias"] == "bullish" else BEAR_FILL
    x0, x1 = shift["regime_start_ts"], shift["regime_end_ts"]
    acc_pct = shift["accuracy"] * 100
    label = f"{shift['equity_bias'][0].upper()} {acc_pct:.0f}% ({shift['n_correct']}/{shift['n_weeks']}w)"

    for row in [1, 2]:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=color, line_width=0,
            row=row, col=1,
        )

    # Regime label on top panel
    mid_ts = x0 + (x1 - x0) / 2
    dxy_range = dxy_weekly.loc[x0:x1]
    y_label = dxy_range["high"].max() * 1.001 if not dxy_range.empty else dxy_weekly["high"].max()
    fig.add_annotation(
        x=mid_ts, y=y_label,
        text=label,
        showarrow=False,
        font=dict(size=9, color="#26a641" if shift["equity_bias"] == "bullish" else "#f85149"),
        xanchor="center", yanchor="bottom",
        row=1, col=1,
    )

# Pool level lines + raid + MSS markers
for shift in shifts:
    raid_ts    = shift["raid_time"]
    confirm_ts = shift["confirm_time"]
    pool_ts    = shift["pool_time"]
    pool_lvl   = shift["pool_level"]
    is_bull    = shift["equity_bias"] == "bullish"
    color      = "#26a641" if is_bull else "#f85149"
    src_label  = shift["source"].upper()

    # Pool level dashed line (pool_time → raid_time)
    fig.add_shape(
        type="line", x0=pool_ts, x1=raid_ts, y0=pool_lvl, y1=pool_lvl,
        line=dict(color="#e3b341", width=1, dash="dash"),
        row=1, col=1,
    )

    # Raid marker
    raid_row = dxy_weekly.loc[raid_ts]
    raid_y   = raid_row["low"] if is_bull else raid_row["high"]
    fig.add_trace(go.Scatter(
        x=[raid_ts], y=[raid_y],
        mode="markers+text",
        marker=dict(symbol="triangle-up" if is_bull else "triangle-down",
                    size=10, color=color),
        text=[f"RAID<br>{src_label}"],
        textposition="bottom center" if is_bull else "top center",
        textfont=dict(size=8, color="#8b949e"),
        showlegend=False,
        hovertemplate=f"Raid {raid_ts.date()}<br>Pool: {pool_lvl:.3f}<br>Source: {src_label}<extra></extra>",
    ), row=1, col=1)

    # MSS confirmation marker
    conf_row = dxy_weekly.loc[confirm_ts]
    conf_y   = conf_row["close"]
    fig.add_trace(go.Scatter(
        x=[confirm_ts], y=[conf_y],
        mode="markers+text",
        marker=dict(symbol="star", size=12, color=color),
        text=["MSS"],
        textposition="top center" if is_bull else "bottom center",
        textfont=dict(size=9, color=color, family="Arial Black"),
        showlegend=False,
        hovertemplate=f"MSS {confirm_ts.date()}<br>→ NQ {shift['equity_bias']}<br>Regime acc: {shift['accuracy']*100:.0f}%<extra></extra>",
    ), row=1, col=1)

# NQ MSS dots (correct=green, wrong=red)
for shift in shifts:
    if shift["regime_start_ts"] is None:
        continue
    regime_ts = [t for t in bars_ts
                 if shift["regime_start_ts"] <= t <= shift["regime_end_ts"]]
    regime_nq = nq_weekly["close"].reindex(regime_ts).dropna()
    regime_dir = nq_dir.reindex(regime_ts).dropna()
    correct = regime_dir == shift["equity_bias"]
    colors  = ["#26a641" if c else "#f85149" for c in correct]
    fig.add_trace(go.Scatter(
        x=regime_nq.index, y=regime_nq.values,
        mode="markers",
        marker=dict(size=5, color=colors),
        showlegend=False,
        hovertemplate="NQ %{y:.0f}<extra></extra>",
    ), row=2, col=1)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="DXY Weekly Bias — HPFS + REH/REL → MSS Structural Regimes<br>"
             f"<sub>ltc=ob  sw_lb={SW_LB}  rel_tol={REL_TOL}  rx_bars={RX_BARS}  rx_type={RX_TYPE}</sub>",
        font=dict(size=16, color="#e6edf3"),
    ),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", size=10),
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    height=800,
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", font=dict(size=10)),
    hovermode="x unified",
)

for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
    fig.update_layout(**{axis: dict(
        gridcolor="#21262d",
        zerolinecolor="#21262d",
        tickfont=dict(color="#8b949e"),
    )})

out = "/tmp/dxy_bias_regimes.html"
fig.write_html(out, include_plotlyjs="cdn")
print(f"\nSaved → {out}")
os.system(f"open {out}")
