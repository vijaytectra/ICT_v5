"""
Forex 1-minute Bid/Ask -> cleaned mid-price OHLCV
================================================

Merges a Dukascopy-style Bid file and Ask file for one FX pair into the
project's canonical 1m CSV format (see CLAUDE.md section  Data):

    timestamp,open,high,low,close,volume,spread,symbol

- OHLC is the **mid price**: (bid + ask) / 2, computed per field.
- `spread` is the close-to-close ask-bid spread, in price units (kept for
  realistic fill modelling later; DataLoader ignores it).
- Timestamps: the source header is `Time (EET)` and the exporter applies
  DST (EET winter UTC+2 / EEST summer UTC+3) - verified by the forex week
  always opening 00:00 local in both seasons. So we localize to
  Europe/Helsinki and convert to UTC, which is what every detector expects.

Usage:
    python Data/prep/forex_1m_to_cleaned.py EURUSD \
        "<path>/EURUSD_1 Min_Bid_....csv" "<path>/EURUSD_1 Min_Ask_....csv"

    # or point at a directory holding the standard-named exports:
    python Data/prep/forex_1m_to_cleaned.py --all "<dir>" --suffix 2024.01.01_2026.09.05
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_TZ = "Europe/Helsinki"          # EET/EEST with DST
TIME_COL = "Time (EET)"
TIME_FMT = "%Y.%m.%d %H:%M:%S"
OHLC = ["open", "high", "low", "close"]


def _pip(pair: str) -> float:
    """Smallest quoted increment.

    Index CFDs (USATECHIDXUSD, USA500IDXUSD, USA30IDXUSD) are quoted in index
    points, so their increment is 1.0 - reporting their spread in FX pips would
    inflate it by four orders of magnitude.
    """
    u = pair.upper()
    if "IDX" in u:
        return 1.0
    return 0.01 if u.endswith("JPY") else 0.0001


def _price_dp(pair: str) -> int:
    """Decimals to keep. Sources quote 3dp (indices, JPY) / 5dp; a mid adds one."""
    u = pair.upper()
    if "IDX" in u or u.endswith("JPY"):
        return 4
    return 6
DATA_DIR = Path(__file__).resolve().parents[1]


def _read_side(path: Path) -> pd.DataFrame:
    """Read one Bid or Ask export into a naive-indexed OHLCV frame."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = TIME_COL.strip().lower()
    if time_col not in df.columns:
        raise ValueError(f"{path.name}: expected a '{TIME_COL}' column, got {list(df.columns)}")
    ts = pd.to_datetime(df[time_col], format=TIME_FMT)
    df = df.drop(columns=[time_col])
    df.index = pd.Index(ts, name="timestamp")
    missing = [c for c in OHLC + ["volume"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df[OHLC + ["volume"]].astype("float64")


def _to_utc(idx: pd.Index) -> pd.Index:
    """Localize naive EET/EEST timestamps to UTC, DST-aware."""
    try:
        return idx.tz_localize(SRC_TZ).tz_convert("UTC")
    except Exception as exc:  # ambiguous / nonexistent DST times
        print(f"  ! strict DST localize failed ({exc}); retrying with shift/infer", file=sys.stderr)
        return idx.tz_localize(
            SRC_TZ, ambiguous="infer", nonexistent="shift_forward"
        ).tz_convert("UTC")


def _repair_ohlc(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Clamp high/low so every bar satisfies low <= open,close <= high.

    The source Bid/Ask exports carry occasional sub-pip inconsistencies (an
    open printed just outside the bar's own high/low). Left alone these create
    phantom gaps in FVG detection, so we widen high/low to contain the bar.
    """
    hi = df[OHLC].max(axis=1)
    lo = df[OHLC].min(axis=1)
    n_bad = int(((df["high"] < hi) | (df["low"] > lo)).sum())
    if n_bad:
        worst = float((hi - df["high"]).clip(lower=0).max())
        worst = max(worst, float((df["low"] - lo).clip(lower=0).max()))
        print(f"[{pair}]   repaired {n_bad:,} bars with inconsistent OHLC "
              f"(max {worst:.7f} = {worst / _pip(pair):.3f} pips)", flush=True)
    df["high"] = hi
    df["low"] = lo
    return df


def convert(pair: str, bid_path: Path, ask_path: Path, out_path: Path) -> pd.DataFrame:
    print(f"[{pair}] reading bid  {bid_path.name}", flush=True)
    bid = _read_side(bid_path)
    print(f"[{pair}] reading ask  {ask_path.name}", flush=True)
    ask = _read_side(ask_path)

    if len(bid) != len(ask) or not bid.index.equals(ask.index):
        common = bid.index.intersection(ask.index)
        print(f"[{pair}] ! bid/ask index mismatch - intersecting "
              f"({len(bid):,}/{len(ask):,} -> {len(common):,})", file=sys.stderr)
        bid, ask = bid.loc[common], ask.loc[common]

    out = pd.DataFrame(index=bid.index)
    for c in OHLC:
        out[c] = (bid[c] + ask[c]) / 2.0
    out["volume"] = bid["volume"] + ask["volume"]
    out["spread"] = ask["close"] - bid["close"]

    out.index = _to_utc(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    dp = _price_dp(pair)
    out[OHLC] = out[OHLC].round(dp)
    out["spread"] = out["spread"].round(dp)
    out["volume"] = out["volume"].round(2)
    out["symbol"] = pair

    out = _repair_ohlc(out, pair)

    # Post-repair integrity assertion: nothing may be outside its own bar.
    assert (out["high"] >= out["low"]).all(), "high < low after repair"
    assert out["open"].between(out["low"], out["high"]).all(), "open outside bar"
    assert out["close"].between(out["low"], out["high"]).all(), "close outside bar"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index_label="timestamp")

    print(f"[{pair}] wrote {out_path}")
    print(f"[{pair}]   rows   : {len(out):,}")
    print(f"[{pair}]   range  : {out.index.min()} -> {out.index.max()}  (UTC)")
    pip = _pip(pair)
    print(f"[{pair}]   spread : median {out['spread'].median() / pip:.2f} pips  "
          f"p95 {out['spread'].quantile(0.95) / pip:.2f} pips")
    return out


def _resolve_pair(dirpath: Path, pair: str, suffix: str) -> tuple[Path, Path]:
    bid = dirpath / f"{pair}_1 Min_Bid_{suffix}.csv"
    ask = dirpath / f"{pair}_1 Min_Ask_{suffix}.csv"
    for p in (bid, ask):
        if not p.exists():
            raise FileNotFoundError(p)
    return bid, ask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs="?", help="e.g. EURUSD")
    ap.add_argument("bid", nargs="?", type=Path)
    ap.add_argument("ask", nargs="?", type=Path)
    ap.add_argument("--all", dest="src_dir", type=Path,
                    help="directory containing the standard-named exports")
    ap.add_argument("--suffix", default="2024.01.01_2026.09.05",
                    help="date-range suffix used in the filenames")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--tag", default="",
                    help="appended to the output filename, e.g. --tag 2015 gives "
                         "usatechidxusd_2015_1m_cleaned.csv; use it to keep a second "
                         "date range for the same instrument side by side")
    args = ap.parse_args()

    if args.src_dir:
        for pair in [p.strip().upper() for p in args.pairs.split(",") if p.strip()]:
            bid, ask = _resolve_pair(args.src_dir, pair, args.suffix)
            tag = f"_{args.tag}" if args.tag else ""
            convert(pair, bid, ask, args.out_dir / f"{pair.lower()}{tag}_1m_cleaned.csv")
            print()
        return 0

    if not (args.pair and args.bid and args.ask):
        ap.error("give PAIR BID ASK, or use --all <dir>")
    tag = f"_{args.tag}" if args.tag else ""
    convert(args.pair.upper(), args.bid, args.ask,
            args.out_dir / f"{args.pair.lower()}{tag}_1m_cleaned.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
