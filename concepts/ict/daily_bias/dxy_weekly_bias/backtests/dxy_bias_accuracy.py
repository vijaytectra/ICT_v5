"""
DXY Weekly Bias — Regime Accuracy (HPFS + REH/REL pools)
==========================================================

Pool sources:
  - HPFS (High Probability Failure Swing): LTC body/wick takes prior extreme,
    then first candle retracing inside the LTC body = HPFS level
  - REH/REL (Relative Equal Highs/Lows): clusters of swing extremes at
    approximately the same price — engineered stop shelves

After a pool raid, MSS or CISD on DXY confirms a directional SHIFT.
A REGIME persists from the confirmation bar until the NEXT confirmed shift
fires (regardless of which pool triggered it).

Metric: % of NQ weekly closes that match the shift direction, measured
across the full active regime between consecutive confirmed shifts.

Usage:
    python backtests/dxy_bias_accuracy.py
"""

import os
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
import sys
import itertools
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(_ROOT / 'src'))

from infrastructure import DataLoader
from concepts import SwingPointScanner, detect_mss, detect_cisd
from concepts import find_hpfs

DATA_DIR = str(_ROOT / 'Data')

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
loader = DataLoader(timeframe="D", data_dir=DATA_DIR)
dxy_daily = loader.read_DXY("daily")

dxy_weekly = (
    dxy_daily
    .resample("W")
    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    .dropna(subset=["open"])
)
dxy_weekly.index = dxy_weekly.index.tz_localize(None)

nq_loader = DataLoader(timeframe="W", data_dir=DATA_DIR)
nq_weekly = nq_loader.read_NQ()
nq_weekly.index = nq_weekly.index.tz_localize(None)

common_weeks = dxy_weekly.index.intersection(nq_weekly.index)
dxy_weekly   = dxy_weekly.loc[common_weeks]
nq_weekly    = nq_weekly.loc[common_weeks]

nq_direction = (nq_weekly["close"] > nq_weekly["open"]).map({True: "bullish", False: "bearish"})

print(f"DXY weekly: {dxy_weekly.index[0].date()} → {dxy_weekly.index[-1].date()}  ({len(dxy_weekly)} weeks)")
print(f"NQ  weekly: {nq_weekly.index[0].date()} → {nq_weekly.index[-1].date()}  ({len(nq_weekly)} weeks)\n")

# ── Parameter grid ────────────────────────────────────────────────────────────
LTC_TYPES         = ["rb", "ob"]         # HPFS variant
SW_LOOKBACKS      = [1, 2, 3]            # swing scanner lookback (shared HPFS scan_forward context + REH/REL)
REL_TOLERANCES    = [0.001, 0.002, 0.005]  # REH/REL clustering tolerance (% of price)
REACTION_BARS_ALL = [1, 2, 3, 4]
REACTION_TYPES    = ["mss", "cisd", "either"]
MAX_LEG_LOOKBACKS = [3, 5, 10]

WARMUP_WEEKS  = 52
MAX_REACTION  = max(REACTION_BARS_ALL)
PRE_RAID_BARS = 20
HPFS_SCAN_FWD = 6     # fixed HPFS scan_forward

bar_index = {ts: i for i, ts in enumerate(dxy_weekly.index)}
bars_ts   = list(dxy_weekly.index)

# ── Pool collection helpers ───────────────────────────────────────────────────

def collect_hpfs_raids(ltc_type: str, leg_lb: int) -> list:
    """Return all HPFS pool raids with MSS/CISD bar offsets."""
    records = []
    for hpfs_dir, dxy_dir in [("bearish", "bearish"), ("bullish", "bullish")]:
        hpfs_df = find_hpfs(dxy_weekly, direction=hpfs_dir, ltc_type=ltc_type, scan_forward=HPFS_SCAN_FWD)
        if hpfs_df.empty:
            continue
        raided = hpfs_df[hpfs_df["raid_time"].notna()].copy()
        for _, row in raided.iterrows():
            raid_time = row["raid_time"]
            raid_bar  = bar_index.get(raid_time)
            if raid_bar is None or raid_bar < WARMUP_WEEKS:
                continue
            rec = _make_raid_record(
                pool_side="high" if hpfs_dir == "bearish" else "low",
                pool_level=float(row["hpfs_level"]),
                pool_time=row["hpfs_time"],
                raid_time=raid_time,
                raid_bar=raid_bar,
                dxy_dir=dxy_dir,
                leg_lb=leg_lb,
                source="hpfs",
            )
            if rec:
                records.append(rec)
    return records


def collect_rel_raids(sw_lb: int, rel_tol: float, leg_lb: int) -> list:
    """Return all REH/REL pool raids with MSS/CISD bar offsets."""
    scanner = SwingPointScanner(dxy_weekly, lookback=sw_lb)
    scanner.identify_swings()

    records = []

    # REH — buy-side stops above equal highs → raid bearish DXY → NQ bullish
    for cluster in scanner.find_relative_equal_highs(tolerance=rel_tol):
        pool_level = cluster["max_price"]
        timestamps = sorted(cluster["timestamps"])
        pool_time  = timestamps[-1]   # pool established when last member forms
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
        rec = _make_raid_record(
            pool_side="high", pool_level=pool_level, pool_time=pool_time,
            raid_time=raid_time, raid_bar=raid_bar,
            dxy_dir="bearish", leg_lb=leg_lb, source="reh",
        )
        if rec:
            records.append(rec)

    # REL — sell-side stops below equal lows → raid bullish DXY → NQ bearish
    for cluster in scanner.find_relative_equal_lows(tolerance=rel_tol):
        pool_level = cluster["min_price"]
        timestamps = sorted(cluster["timestamps"])
        pool_time  = timestamps[-1]
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
        rec = _make_raid_record(
            pool_side="low", pool_level=pool_level, pool_time=pool_time,
            raid_time=raid_time, raid_bar=raid_bar,
            dxy_dir="bullish", leg_lb=leg_lb, source="rel",
        )
        if rec:
            records.append(rec)

    return records


def _make_raid_record(pool_side, pool_level, pool_time, raid_time, raid_bar,
                      dxy_dir, leg_lb, source) -> dict | None:
    """Run MSS + CISD on the raid window and return a record, or None if window is too short."""
    start  = max(0, raid_bar - PRE_RAID_BARS)
    end    = min(raid_bar + MAX_REACTION + 1, len(dxy_weekly))
    window = dxy_weekly.iloc[start:end]
    if len(window) < PRE_RAID_BARS // 2:
        return None

    mss_bar_offset = None
    mss = detect_mss(window, direction=dxy_dir, swing_lookback=1,
                     max_bars_after_sweep=MAX_REACTION, sweep_time=raid_time)
    if mss is not None:
        mss_bar = bar_index.get(mss["break_time"])
        if mss_bar is not None:
            mss_bar_offset = mss_bar - raid_bar

    cisd_bar_offset = None
    cisd = detect_cisd(window, direction=dxy_dir, sweep_time=raid_time,
                       max_leg_lookback=leg_lb, max_bars_after_sweep=MAX_REACTION)
    if cisd is not None:
        cisd_bar = bar_index.get(cisd["cisd_time"])
        if cisd_bar is not None:
            cisd_bar_offset = cisd_bar - raid_bar

    return {
        "source":           source,
        "pool_side":        pool_side,
        "pool_level":       pool_level,
        "pool_time":        pool_time,
        "raid_time":        raid_time,
        "raid_bar":         raid_bar,
        "mss_bar_offset":   mss_bar_offset,
        "cisd_bar_offset":  cisd_bar_offset,
        "equity_bias":      "bullish" if dxy_dir == "bearish" else "bearish",
    }


# ── Step 1: Pre-compute all pool raids ───────────────────────────────────────
print("Pre-computing pool raids...")
# key: (ltc_type, sw_lb, rel_tol, leg_lb) → list of raid records
precomputed = {}

for ltc_type, sw_lb, rel_tol, leg_lb in itertools.product(
    LTC_TYPES, SW_LOOKBACKS, REL_TOLERANCES, MAX_LEG_LOOKBACKS
):
    hpfs_raids = collect_hpfs_raids(ltc_type, leg_lb)
    rel_raids  = collect_rel_raids(sw_lb, rel_tol, leg_lb)
    all_raids  = hpfs_raids + rel_raids

    # Deduplicate: if HPFS and REH/REL trigger on the same raid_bar for the same
    # pool_side, keep the one with the earlier confirmation (more precise).
    seen = {}
    for r in sorted(all_raids, key=lambda x: x["raid_bar"]):
        key = (r["raid_bar"], r["pool_side"])
        if key not in seen:
            seen[key] = r
    deduped = sorted(seen.values(), key=lambda x: x["raid_bar"])

    precomputed[(ltc_type, sw_lb, rel_tol, leg_lb)] = deduped
    print(f"  ltc={ltc_type} sw_lb={sw_lb} tol={rel_tol:.3f} leg_lb={leg_lb:>2}: "
          f"{len(hpfs_raids)} hpfs + {len(rel_raids)} rel = {len(deduped)} raids")


# ── Step 2: Grid search — regime accuracy ─────────────────────────────────────
print("\nRunning grid search...")
results = []

for ltc_type, sw_lb, rel_tol, rx_bars, rx_type, leg_lb in itertools.product(
    LTC_TYPES, SW_LOOKBACKS, REL_TOLERANCES, REACTION_BARS_ALL, REACTION_TYPES, MAX_LEG_LOOKBACKS
):
    all_raids = precomputed[(ltc_type, sw_lb, rel_tol, leg_lb)]
    if not all_raids:
        continue

    # Filter to confirmed shifts only
    shifts = []
    for r in all_raids:
        confirm_bar = None
        confirm_type = None
        if rx_type in ("mss", "either"):
            o = r["mss_bar_offset"]
            if o is not None and 1 <= o <= rx_bars:
                confirm_bar  = r["raid_bar"] + o
                confirm_type = "mss"
        if confirm_bar is None and rx_type in ("cisd", "either"):
            o = r["cisd_bar_offset"]
            if o is not None and 1 <= o <= rx_bars:
                confirm_bar  = r["raid_bar"] + o
                confirm_type = "cisd"
        if confirm_bar is None or confirm_bar < WARMUP_WEEKS:
            continue
        shifts.append({
            **r,
            "confirm_bar":  confirm_bar,
            "confirm_type": confirm_type,
        })

    if not shifts:
        results.append({
            "ltc_type": ltc_type, "sw_lb": sw_lb, "rel_tol": rel_tol,
            "reaction_bars": rx_bars, "reaction_type": rx_type, "max_leg_lookback": leg_lb,
            "n_regimes": 0, "n_weeks": 0, "coverage_pct": 0.0,
            "accuracy_pct": 0.0, "eff_acc_pct": 0.0,
        })
        continue

    # Sort shifts chronologically
    shifts.sort(key=lambda s: s["confirm_bar"])

    # Build regime sequence: each shift's equity_bias holds from confirm_bar
    # until the next shift's confirm_bar
    regime_accs = []
    total_weeks = 0

    for k, shift in enumerate(shifts):
        regime_start = shift["confirm_bar"] + 1   # week AFTER confirmation
        regime_end   = shifts[k + 1]["confirm_bar"] if k + 1 < len(shifts) else len(dxy_weekly)

        if regime_start >= regime_end:
            continue

        regime_ts  = bars_ts[regime_start:regime_end]
        regime_nq  = nq_direction.reindex(regime_ts).dropna()
        if len(regime_nq) == 0:
            continue

        n_correct = int((regime_nq == shift["equity_bias"]).sum())
        regime_accs.append(n_correct / len(regime_nq))
        total_weeks += len(regime_nq)

    n_regimes  = len(regime_accs)
    n_total    = len(dxy_weekly) - WARMUP_WEEKS
    coverage   = total_weeks / n_total if n_total else 0

    if n_regimes == 0:
        accuracy = eff = 0.0
    else:
        accuracy = float(np.mean(regime_accs))
        eff      = accuracy * coverage

    results.append({
        "ltc_type":         ltc_type,
        "sw_lb":            sw_lb,
        "rel_tol":          rel_tol,
        "reaction_bars":    rx_bars,
        "reaction_type":    rx_type,
        "max_leg_lookback": leg_lb,
        "n_regimes":        n_regimes,
        "n_weeks":          total_weeks,
        "coverage_pct":     round(coverage * 100, 1),
        "accuracy_pct":     round(accuracy * 100, 1),
        "eff_acc_pct":      round(eff * 100, 1),
    })

results_df = pd.DataFrame(results).sort_values("eff_acc_pct", ascending=False)

# ── Console output ────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'ltc':<4}  {'sw':<3}  {'tol':>5}  {'rx':>2}  {'type':<6}  {'leg':>3}  "
      f"{'regimes':>7}  {'weeks':>6}  {'cov%':>5}  {'acc%':>5}  {'eff%':>5}")
print("-" * 110)
for _, row in results_df.head(25).iterrows():
    print(
        f"{row['ltc_type']:<4}  "
        f"{int(row['sw_lb']):<3}  "
        f"{row['rel_tol']:>5.3f}  "
        f"{int(row['reaction_bars']):>2}  "
        f"{row['reaction_type']:<6}  "
        f"{int(row['max_leg_lookback']):>3}  "
        f"{int(row['n_regimes']):>7}  "
        f"{int(row['n_weeks']):>6}  "
        f"{row['coverage_pct']:>5.1f}  "
        f"{row['accuracy_pct']:>5.1f}  "
        f"{row['eff_acc_pct']:>5.1f}"
    )
print("=" * 110)

csv_path = "/tmp/dxy_bias_grid.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nFull results → {csv_path}")

best = results_df.iloc[0]
print(f"\nBest combo:")
print(f"  ltc_type         = {best['ltc_type']}")
print(f"  sw_lb            = {int(best['sw_lb'])}")
print(f"  rel_tol          = {best['rel_tol']}")
print(f"  reaction_bars    = {int(best['reaction_bars'])}")
print(f"  reaction_type    = {best['reaction_type']}")
print(f"  max_leg_lookback = {int(best['max_leg_lookback'])}")
print(f"  → accuracy {best['accuracy_pct']:.1f}%  "
      f"coverage {best['coverage_pct']:.1f}%  "
      f"eff {best['eff_acc_pct']:.1f}%  "
      f"({int(best['n_regimes'])} regimes / {int(best['n_weeks'])} weeks)")

# ── Heatmap: accuracy vs reaction_bars for each reaction_type ─────────────────
BG = "#0d1117"
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("DXY Weekly Bias (HPFS + REH/REL) — Regime Accuracy % (sw_lb × reaction_bars)",
             color="#e6edf3", fontsize=13)

for ax, rtype in zip(axes, ["mss", "cisd", "either"]):
    ax.set_facecolor(BG)
    sub = results_df[results_df["reaction_type"] == rtype]
    pivot = (
        sub.groupby(["sw_lb", "reaction_bars"])["accuracy_pct"]
        .max()
        .unstack("reaction_bars")
    )
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=45, vmax=70, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color="#8b949e", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="#8b949e", fontsize=9)
    ax.set_xlabel("reaction_bars", color="#8b949e", fontsize=9)
    ax.set_ylabel("sw_lb", color="#8b949e", fontsize=9)
    ax.set_title(f"reaction_type = {rtype}", color="#e6edf3", fontsize=10)

    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            v = pivot.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.0f}%", ha="center", va="center",
                        color="black" if 48 < v < 67 else "white",
                        fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
png_path = "/tmp/dxy_bias_best.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Heatmap → {png_path}")
os.system(f"open {png_path}")
