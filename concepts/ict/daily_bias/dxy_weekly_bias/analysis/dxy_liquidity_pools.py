"""
DXY Liquidity Pools Chart
=========================
Shows all detected HPFS and REH/REL pool levels on DXY daily candles.
Each pool is drawn as a horizontal line from formation to raid (or end of chart if unraided).
Confirmed signals (led to MSS within rx_bars) are highlighted.

Usage:
    python concepts/ict/daily_bias/dxy_weekly_bias/analysis/dxy_liquidity_pools.py
"""

import os
import sys
import pathlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

from infrastructure import DataLoader
from concepts import SwingPointScanner, detect_mss
from concepts import find_hpfs

DATA_DIR = str(_ROOT / 'Data')

# ── Params (same as best-fit backtest) ────────────────────────────────────────
LTC_TYPE      = "ob"
SW_LB         = 1
REL_TOL       = 0.005
RX_BARS       = 1
HPFS_SCAN_FWD = 6
PRE_RAID_BARS = 20
MAX_REACTION  = 4
WARMUP_WEEKS  = 52

# ── Load data ─────────────────────────────────────────────────────────────────
loader = DataLoader(timeframe="D", data_dir=DATA_DIR)
dxy_daily = loader.read_DXY("daily")
dxy_daily.index = dxy_daily.index.tz_localize(None)

dxy_weekly = (
    dxy_daily.resample("W")
    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    .dropna(subset=["open"])
)
dxy_weekly.index = dxy_weekly.index.tz_localize(None)

bar_index = {ts: i for i, ts in enumerate(dxy_weekly.index)}
bars_ts   = list(dxy_weekly.index)

# ── Collect all pools (with and without confirmation) ─────────────────────────
def check_confirmation(raid_bar, pool_side, dxy_dir):
    start  = max(0, raid_bar - PRE_RAID_BARS)
    end    = min(raid_bar + MAX_REACTION + 1, len(dxy_weekly))
    window = dxy_weekly.iloc[start:end]
    if len(window) < PRE_RAID_BARS // 2:
        return False
    raid_time = bars_ts[raid_bar]
    mss = detect_mss(window, direction=dxy_dir, swing_lookback=1,
                     max_bars_after_sweep=MAX_REACTION, sweep_time=raid_time)
    if mss is None:
        return False
    mb = bar_index.get(mss["break_time"])
    if mb is None:
        return False
    return 1 <= (mb - raid_bar) <= RX_BARS

pools = []

# ── HPFS pools ────────────────────────────────────────────────────────────────
for hpfs_dir, dxy_dir in [("bearish", "bearish"), ("bullish", "bullish")]:
    hpfs_df = find_hpfs(dxy_weekly, direction=hpfs_dir, ltc_type=LTC_TYPE, scan_forward=HPFS_SCAN_FWD)
    for _, row in hpfs_df.iterrows():
        hpfs_bar  = int(row["hpfs_bar"])
        hpfs_time = row["hpfs_time"]
        hpfs_lvl  = float(row["hpfs_level"])
        raid_time = row["raid_time"]   # may be NaT
        raid_bar  = bar_index.get(raid_time) if pd.notna(raid_time) else None

        is_warmup = (raid_bar is not None and raid_bar < WARMUP_WEEKS)
        confirmed = (
            not is_warmup
            and raid_bar is not None
            and check_confirmation(raid_bar, "high" if hpfs_dir == "bearish" else "low", dxy_dir)
        )

        pools.append({
            "source":    "hpfs",
            "side":      "high" if hpfs_dir == "bearish" else "low",
            "level":     hpfs_lvl,
            "form_time": hpfs_time,
            "raid_time": raid_time if pd.notna(raid_time) else None,
            "confirmed": confirmed,
            "dxy_dir":   dxy_dir,
        })

# ── REH pools ─────────────────────────────────────────────────────────────────
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
    raid_time = raided.index[0] if not raided.empty else None
    raid_bar  = bar_index[raid_time] if raid_time is not None else None
    confirmed = (
        raid_bar is not None
        and raid_bar >= WARMUP_WEEKS
        and check_confirmation(raid_bar, "high", "bearish")
    )
    pools.append({
        "source":    "reh",
        "side":      "high",
        "level":     pool_level,
        "form_time": pool_time,
        "raid_time": raid_time,
        "confirmed": confirmed,
        "dxy_dir":   "bearish",
    })

for cluster in scanner.find_relative_equal_lows(tolerance=REL_TOL):
    pool_level = cluster["min_price"]
    pool_time  = sorted(cluster["timestamps"])[-1]
    pool_bar   = bar_index.get(pool_time)
    if pool_bar is None:
        continue
    future = dxy_weekly.iloc[pool_bar + 1:]
    raided = future[future["low"] < pool_level]
    raid_time = raided.index[0] if not raided.empty else None
    raid_bar  = bar_index[raid_time] if raid_time is not None else None
    confirmed = (
        raid_bar is not None
        and raid_bar >= WARMUP_WEEKS
        and check_confirmation(raid_bar, "low", "bullish")
    )
    pools.append({
        "source":    "rel",
        "side":      "low",
        "level":     pool_level,
        "form_time": pool_time,
        "raid_time": raid_time,
        "confirmed": confirmed,
        "dxy_dir":   "bullish",
    })

print(f"Total pools: {len(pools)}")
confirmed_count = sum(1 for p in pools if p["confirmed"])
print(f"  Confirmed (led to MSS shift): {confirmed_count}")
print(f"  Unconfirmed: {len(pools) - confirmed_count}")
by_source = {}
for p in pools:
    by_source.setdefault(p["source"], 0)
    by_source[p["source"]] += 1
for src, cnt in sorted(by_source.items()):
    print(f"  {src}: {cnt}")

# ── Build figure ──────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.03,
    subplot_titles=["DXY Daily — All Liquidity Pools", "DXY Weekly — Pool Levels"],
)

# DXY Daily candlestick
fig.add_trace(go.Candlestick(
    x=dxy_daily.index,
    open=dxy_daily["open"], high=dxy_daily["high"],
    low=dxy_daily["low"],   close=dxy_daily["close"],
    name="DXY Daily",
    increasing_line_color="#3fb950", decreasing_line_color="#da3633",
    increasing_fillcolor="#3fb950",  decreasing_fillcolor="#da3633",
    line_width=0.5,
), row=1, col=1)

# DXY Weekly candlestick
fig.add_trace(go.Candlestick(
    x=dxy_weekly.index,
    open=dxy_weekly["open"], high=dxy_weekly["high"],
    low=dxy_weekly["low"],   close=dxy_weekly["close"],
    name="DXY Weekly",
    increasing_line_color="#26a641", decreasing_line_color="#f85149",
    increasing_fillcolor="#26a641",  decreasing_fillcolor="#f85149",
    line_width=0.8,
), row=2, col=1)

# ── Draw pool lines ───────────────────────────────────────────────────────────
# Color scheme:
#   confirmed HPFS bearish (DXY down → NQ up)  = cyan solid
#   confirmed HPFS bullish (DXY up → NQ down)   = orange solid
#   confirmed REH                                = cyan dashed
#   confirmed REL                                = orange dashed
#   unconfirmed                                  = gray dotted

def pool_color(p):
    if not p["confirmed"]:
        return "#484f58"    # gray
    if p["dxy_dir"] == "bearish":
        return "#58a6ff"    # blue (DXY bearish → NQ bullish)
    return "#f78166"        # orange (DXY bullish → NQ bearish)

def pool_dash(p):
    if not p["confirmed"]:
        return "dot"
    if p["source"] in ("reh", "rel"):
        return "dash"
    return "solid"

def pool_width(p):
    return 1.5 if p["confirmed"] else 0.6

chart_end = dxy_daily.index[-1]

for p in pools:
    x0 = p["form_time"]
    x1 = p["raid_time"] if p["raid_time"] is not None else chart_end
    lvl = p["level"]
    col = pool_color(p)
    dash = pool_dash(p)
    w    = pool_width(p)
    src  = p["source"].upper()
    conf_str = "✓" if p["confirmed"] else ""

    for row in [1, 2]:
        fig.add_shape(
            type="line",
            x0=x0, x1=x1, y0=lvl, y1=lvl,
            line=dict(color=col, width=w, dash=dash),
            row=row, col=1,
        )

    # Label only confirmed pools
    if p["confirmed"]:
        fig.add_annotation(
            x=x0, y=lvl,
            text=f"{src} {lvl:.2f}",
            showarrow=False, xanchor="right",
            font=dict(size=7, color=col),
            row=1, col=1,
        )

    # Raid dot on daily
    if p["raid_time"] is not None:
        is_high = p["side"] == "high"
        daily_w = dxy_daily.loc[
            p["raid_time"] - pd.Timedelta(days=8) : p["raid_time"] + pd.Timedelta(days=1)
        ]
        if not daily_w.empty:
            hit = daily_w[daily_w["high"] > lvl] if is_high else daily_w[daily_w["low"] < lvl]
            if not hit.empty:
                rd_ts = hit.index[-1]
                rd = dxy_daily.loc[rd_ts]
                fig.add_trace(go.Scatter(
                    x=[rd_ts],
                    y=[rd["high"] if is_high else rd["low"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down" if is_high else "triangle-up",
                        size=8 if p["confirmed"] else 5,
                        color=col,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>RAID</b> {rd_ts.date()}<br>{src} pool: {lvl:.3f}<br>{conf_str}<extra></extra>",
                ), row=1, col=1)

# ── Legend items (phantom traces) ────────────────────────────────────────────
for label, color, dash in [
    ("Confirmed – DXY Bearish (NQ↑)", "#58a6ff", "solid"),
    ("Confirmed – DXY Bullish (NQ↓)", "#f78166", "solid"),
    ("Confirmed REH/REL", "#58a6ff", "dash"),
    ("Unconfirmed Pool", "#484f58", "dot"),
]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=color, dash=dash, width=1.5),
        name=label, showlegend=True,
    ), row=1, col=1)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=f"DXY Liquidity Pools — HPFS ({LTC_TYPE.upper()}) + REH/REL<br>"
             f"<sub>Solid/dashed = confirmed (raid + MSS shift)  ·  Dotted gray = unconfirmed  ·  "
             f"Blue = DXY bearish (NQ bullish)  ·  Orange = DXY bullish (NQ bearish)</sub>",
        font=dict(size=14, color="#e6edf3"),
    ),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", size=10),
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    height=1400,
    legend=dict(
        bgcolor="#161b22", bordercolor="#30363d",
        font=dict(size=10), x=0.01, y=0.99,
    ),
    hovermode="x unified",
)

for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
    fig.update_layout(**{axis: dict(
        gridcolor="#21262d",
        zerolinecolor="#21262d",
        tickfont=dict(color="#8b949e"),
    )})

out = "/tmp/dxy_liquidity_pools.html"
fig.write_html(out, include_plotlyjs="cdn")
print(f"\nSaved → {out}")
os.system(f"open {out}")
