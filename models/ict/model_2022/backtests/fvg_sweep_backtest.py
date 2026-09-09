#!/usr/bin/env python3
"""
FVG Sweep Model Backtester — ICT 2022 Model on NQ
==================================================

Walk-forward backtest on NQ futures. Configurable via YAML (--config) or
run with defaults (no bias gate, wick stop, all pools).

Usage:
  python fvg_sweep_backtest.py                          # uses defaults
  python fvg_sweep_backtest.py --config configs/model_2022/baseline.yaml

Config keys (all optional — missing keys use model/backtest defaults):
  name            str    experiment name → results/<name>/ output dir
  description     str    human note, not used in code
  start_date      str    backtest start, e.g. '2024-11-01'
  instrument      str    'NQ', 'ES', 'YM', 'EURUSD', 'GBPUSD' or 'USDJPY'
  dollars_per_point float USD per 1.00 price move; defaults per DOLLARS_PER_POINT
  be_trigger_r    float  Move the stop to breakeven once the trade runs this many R (0 = off)
  entry_timeframe str    Entry/MSS bar size, default '2min' (the ICT Silver Bullet cases use 1min)
  partials        list   Partial-take schedule [[fraction, R], ...]; the remainder is the runner
  partial_be      bool   Move the stop to breakeven once the first partial fills (default True)
  eod_exit_hour   int    NY hour of the forced flat (default 16)
  use_spread      bool   Charge the recorded per-bar spread on entry and exit (default False)
  spread_multiplier float Scale that recorded spread (default 1.0)
  commission_points float Round-turn commission in price points (default 0.0)
  slippage_points float  Adverse slippage in price points applied to stop fills (default 0.0)
  limit_expires_with_killzone bool Cancel an unfilled limit when its killzone closes (default True)
  cancel_on_target bool  Abandon the setup if the target trades before the fill (default True)
  use_daily_bias  bool   True = use bias gate; False = try both directions (default False)
  <model keys>    any    any key in Model2022.DEFAULT_CONFIG is forwarded to the model

Outputs (written to results/<name>/):
  *_trades.csv   — full trade log
  *_equity.png   — equity curve
  *_report.html  — interactive Plotly multi-timeframe report
"""

import argparse
import os
import pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_trades import (
    localize_to, eod_utc,
    simulate_fill, simulate_exit, calc_pnl_pts, CostModel,
    compute_stats, print_stats, plot_equity_curve,
)

from infrastructure import DataLoader
from models.ict.model_2022 import Model2022, Model2022Snapshot
from infrastructure.report import Mark, TradeViz, build_report

# ─── Constants ───────────────────────────────────────────────────────────────

# USD per 1.00 of price movement, per instrument.
#   Index futures: the contract multiplier (NQ $20/pt, ES $50/pt, YM $5/pt).
#   Spot FX: 1 standard lot = 100,000 base units, so a 1.00 move in a
#   USD-quoted pair (EURUSD, GBPUSD) is $100,000 -> 1 pip = $10. USDJPY is
#   JPY-quoted, so its USD value depends on the rate; 640 approximates one
#   standard lot near USDJPY 156 (100000/156). Override per config.
DOLLARS_PER_POINT = {
    'NQ': 20.0, 'ES': 50.0, 'YM': 5.0,
    'EURUSD': 100000.0, 'GBPUSD': 100000.0, 'USDJPY': 640.0,
    # Index CFDs are quoted in index points; 1.0 keeps P&L in points so the
    # figures do not depend on a chosen lot size.
    'USATECHIDXUSD': 1.0, 'USA500IDXUSD': 1.0, 'USA30IDXUSD': 1.0,
}
NQ_DOLLARS_PER_POINT = DOLLARS_PER_POINT['NQ']  # back-compat alias

# Decimals kept in the trade log. Index futures quote in points (2dp is
# plenty); FX quotes 5dp (3dp for JPY) and a stop is ~0.0013, so rounding
# to 2dp would collapse every price and P&L column to zero.
PRICE_DP = {
    'NQ': 2, 'ES': 2, 'YM': 2,
    'EURUSD': 6, 'GBPUSD': 6, 'USDJPY': 4,
    'USATECHIDXUSD': 2, 'USA500IDXUSD': 2, 'USA30IDXUSD': 2,
}
DEFAULT_PRICE_DP = 6
DEFAULT_START_DATE   = '2024-11-01'
DEFAULT_INSTRUMENT   = 'NQ'
WARMUP_WEEKS         = 14
# Session close used for the forced flat. Configurable per run via the
# 'eod_exit_hour' key; this remains the default so behaviour is unchanged.
# Interpreted in America/New_York by eod_utc(), which converts to the frame's
# timezone - the constant is a NY wall-clock hour, not a UTC hour.
DEFAULT_EOD_EXIT_HOUR = 16
EOD_EXIT_HOUR        = DEFAULT_EOD_EXIT_HOUR  # back-compat alias

DATA_DIR     = str(_ROOT / 'Data')
BASE_OUT_DIR = str(_HERE / 'results' / 'runs')

# Keys consumed by the backtest runner; everything else goes to Model2022(config=...)
BACKTEST_KEYS = {'name', 'description', 'start_date', 'instrument', 'use_daily_bias',
                 'dollars_per_point', 'price_dp', 'be_trigger_r', 'entry_timeframe',
                 'partials', 'partial_be', 'eod_exit_hour',
                 'use_spread', 'spread_multiplier', 'commission_points',
                 'slippage_points', 'limit_expires_with_killzone',
                 'cancel_on_target'}


# ─── Snapshot builder ────────────────────────────────────────────────────────

_TF_CACHE: dict = {}


def _resampled(df_1m: pd.DataFrame, entry_tf: str) -> dict:
    """Resample the whole history once and cache it.

    build_snapshot used to resample raw 1m bars inside the per-session loop:
    6 days to the entry timeframe, 15 to 15m, and 45 each to daily and 4h, for
    every session. Over a multi-year run that recomputes the same bars ~100
    times. Resampling once and slicing per session is identical output because
    every rule here (2m/15m/4h/D) divides a day evenly and pandas anchors bins
    to midnight, so a slice of the resampled frame equals a resample of the
    slice. test_build_snapshot_matches_per_session_resample pins that.
    """
    key = (id(df_1m), len(df_1m), entry_tf)
    hit = _TF_CACHE.get(key)
    if hit is not None:
        return hit
    entry_agg = {'open': 'first', 'high': 'max', 'low': 'min',
                 'close': 'last', 'volume': 'sum'}
    if 'spread' in df_1m.columns:
        # Mean, not sum: spread is a price distance carried per bar so execution
        # can be charged from recorded history rather than a fabricated constant.
        entry_agg['spread'] = 'mean'
    ohlcv = {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}
    out = {
        'entry': df_1m.resample(entry_tf).agg(entry_agg).dropna(subset=['close']),
        '15min': df_1m.resample('15min').agg(ohlcv).dropna(subset=['close']),
        'D':     df_1m.resample('D').agg(ohlcv).dropna(subset=['close']),
        '4h':    df_1m.resample('4h').agg(ohlcv).dropna(subset=['close']),
    }
    _TF_CACHE.clear()          # one dataset per worker; never grow unbounded
    _TF_CACHE[key] = out
    return out


def build_snapshot(df_1m: pd.DataFrame, session_ts: pd.Timestamp,
                   entry_tf: str = '2min') -> Model2022Snapshot:
    d_naive = pd.Timestamp(session_ts.date())
    d = d_naive.tz_localize('UTC') if df_1m.index.tz is not None else d_naive
    tf = _resampled(df_1m, entry_tf)
    return Model2022Snapshot(
        df_2m=tf['entry'].loc[d - pd.Timedelta(days=5): d + pd.Timedelta(days=1)],
        df_15m=tf['15min'].loc[d - pd.Timedelta(days=14): d + pd.Timedelta(days=1)],
        df_daily=tf['D'].loc[d - pd.Timedelta(days=45): d - pd.Timedelta(minutes=1)],
        session_date=session_ts,
        df_4h=tf['4h'].loc[d - pd.Timedelta(days=45): d - pd.Timedelta(minutes=1)],
    )


# ─── Plotly report adapter ───────────────────────────────────────────────────

C_ENTRY, C_STOP, C_TARGET = '#1f77b4', '#c0392b', '#1a8a4a'
C_FVG, C_SWEEP, C_MSS, C_BIAS = '#f39c12', '#8e44ad', '#16a085', '#7f8c8d'


def _window(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    lo = localize_to(lo, df.index)
    hi = localize_to(hi, df.index)
    return df.loc[lo:hi]


def trade_panels(snap: Model2022Snapshot, session_ts: pd.Timestamp) -> dict:
    d = pd.Timestamp(session_ts.date())
    session_day_bar = (
        snap.df_2m
        .resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                            'close': 'last', 'volume': 'sum'})
        .dropna(subset=['close'])
    )
    d_loc = localize_to(d, session_day_bar.index)
    session_day_bar = session_day_bar[session_day_bar.index.normalize() == d_loc]
    daily_display = pd.concat([snap.df_daily.tail(30), session_day_bar])
    daily_display = daily_display[~daily_display.index.duplicated(keep='last')]
    return {
        'daily': daily_display,
        '15m':   _window(snap.df_15m, d - pd.Timedelta(days=5), d + pd.Timedelta(days=1)),
        '2m':    _window(snap.df_2m,  d, d + pd.Timedelta(days=1)),
    }


def signal_to_marks(signal: dict, fill: dict, exit_info: dict) -> list:
    direction = signal['direction']
    marks = []
    g = 'Entry/Stop/Target'
    marks += [
        Mark('level',  g, '2m', f"Entry {signal['entry']:.2f}",   C_ENTRY,  price=signal['entry']),
        Mark('level',  g, '2m', f"Stop {signal['stop']:.2f}",     C_STOP,   dash='dash', price=signal['stop']),
        Mark('level',  g, '2m', f"Target {signal['target']:.2f}", C_TARGET, dash='dash', price=signal['target']),
        Mark('marker', g, '2m', "Fill", C_ENTRY,
             time=fill['fill_time'], price=fill['fill_price'], symbol='circle'),
    ]
    if exit_info:
        col = (C_TARGET if exit_info['exit_reason'] == 'target'
               else C_STOP if exit_info['exit_reason'] == 'stop' else C_BIAS)
        marks.append(Mark('marker', g, '2m', f"Exit ({exit_info['exit_reason']})", col,
                          time=exit_info['exit_time'], price=exit_info['exit_price'], symbol='x'))

    if signal.get('fvg_top') is not None:
        marks.append(Mark('zone', 'FVG', '2m', "FVG", C_FVG,
                          y0=signal['fvg_bottom'], y1=signal['fvg_top'],
                          t0=signal['entry_time'],
                          t1=(exit_info['exit_time'] if exit_info else None)))

    if signal.get('sweep_time') is not None:
        lbl = f"Sweep {signal.get('swept_pool') or ''}".strip()
        marks.append(Mark('marker', 'Sweep', '2m', lbl, C_SWEEP,
                          time=signal['sweep_time'], price=signal['sweep_level'],
                          symbol='x', default_on=True))
        marks.append(Mark('level', 'Sweep', '15m', f"Swept pool {signal['sweep_level']:.2f}",
                          C_SWEEP, dash='dot', price=signal['sweep_level'], default_on=True))
        cluster = signal.get('swept_cluster') or {}
        source  = cluster.get('source', '')
        price   = cluster.get('price', signal['sweep_level'])
        if source in ('OR-high', 'OR-low'):
            or_high = price if source == 'OR-high' else cluster.get('range_high', price)
            or_low  = price if source == 'OR-low'  else cluster.get('range_low',  price)
            t0, t1  = cluster.get('range_start'), cluster.get('range_end')
            if t0 is not None and t1 is not None:
                marks.append(Mark('zone', 'Sweep', '2m', 'Opening Range', C_SWEEP,
                                  y0=or_low, y1=or_high, t0=t0, t1=t1, default_on=True))
            for ts in cluster.get('timestamps', []):
                marks.append(Mark('marker', 'Sweep', '2m', f"Pool: {source} {price:.2f}",
                                  C_SWEEP, time=ts, price=price, symbol='diamond', default_on=True))
        elif source in ('PDH', 'PDL'):
            for ts in cluster.get('timestamps', []):
                marks.append(Mark('marker', 'Sweep', 'daily', f"Pool: {source} {price:.2f}",
                                  C_SWEEP, time=ts, price=price, symbol='diamond', default_on=True))
        else:
            for ts in cluster.get('timestamps', []):
                marks.append(Mark('marker', 'Sweep', '15m', f"Pool: {source} {price:.2f}",
                                  C_SWEEP, time=ts, price=price, symbol='diamond', default_on=True))

    if signal.get('bms_time') is not None:
        sym = 'triangle-down' if direction == 'short' else 'triangle-up'
        marks.append(Mark('marker', 'MSS', '2m', f"MSS {signal['bms_swing_level']:.2f}", C_MSS,
                          time=signal['bms_time'], price=signal['bms_swing_level'],
                          symbol=sym, default_on=True))
        marks.append(Mark('level', 'MSS', '2m', "MSS swing", C_MSS, dash='dot',
                          price=signal['bms_swing_level'], default_on=True))
    return marks


# ─── Main loop ───────────────────────────────────────────────────────────────

def killzone_expiry(sweep_time, killzones, tz_ref) -> "pd.Timestamp | None":
    """When the killzone that contained the sweep closes.

    entry-sequence.md cancels an unfilled limit at that boundary. Windows are
    NY wall-clock, so the comparison is made in America/New_York and the result
    converted back to the frame's timezone.
    """
    if sweep_time is None:
        return None
    ts = pd.Timestamp(sweep_time)
    if ts.tzinfo is None:
        return None
    ny = ts.tz_convert('America/New_York')
    mins = ny.hour * 60 + ny.minute
    for start, end in killzones:
        sh, sm = map(int, str(start).split(':'))
        eh, em = map(int, str(end).split(':'))
        if sh * 60 + sm <= mins < eh * 60 + em:
            close = ny.normalize() + pd.Timedelta(hours=eh, minutes=em)
            return close.tz_convert(tz_ref) if tz_ref is not None else close.tz_localize(None)
    return None


def classify_exit(reason: str, pnl_pts: float) -> str:
    """Exit route, kept separate from trade outcome.

    A target-hit rate and a profitable-trade rate are different measurements:
    an EOD flat can be either, and a breakeven exit is neither a win nor a loss.
    Collapsing them into one "win rate" is what made a book that is 40%
    profitable read as 7.5%.
    """
    if reason == 'target':
        return 'TP'
    if reason == 'stop':
        return 'SL'
    if reason == 'breakeven':
        return 'BREAKEVEN'
    if reason == 'eod':
        return 'EOD_PROFIT' if pnl_pts > 0 else 'EOD_LOSS'
    return 'OTHER'


def _kz_label(ts) -> str:
    """NY-hour label for the sweep, so losses can be grouped by session."""
    if ts is None:
        return ''
    try:
        ny = ts.tz_convert('America/New_York')
    except Exception:
        return ''
    h = ny.hour + ny.minute / 60
    if 2 <= h < 5:    return 'London 02-05'
    if 7 <= h < 10:   return 'NY 07-10'
    if 10 <= h < 12:  return 'LDN close 10-12'
    if 13.5 <= h < 16: return 'NY PM'
    return f'other {ny:%H:%M}'


def run_backtest(
    run_config: dict = None,
    verbose: bool = True,
    save_outputs: bool = True,
    df_1m_cache: pd.DataFrame = None,
) -> dict:
    """
    Run the backtest and return a stats dict.

    Args:
        run_config:   Flat dict of config overrides (BACKTEST_KEYS + model keys).
                      None uses all defaults.
        verbose:      Print progress and stats to stdout.
        save_outputs: Write CSV, PNG, and HTML to results dir.
        df_1m_cache:  Pre-loaded 1m DataFrame to skip re-loading (for grid search).

    Returns:
        dict with 'name', 'config', 'n_trades', 'n_signals', and all compute_stats keys.
    """
    cfg = run_config or {}

    name           = cfg.get('name', 'baseline')
    start_date     = cfg.get('start_date', DEFAULT_START_DATE)
    instrument     = cfg.get('instrument', DEFAULT_INSTRUMENT).upper()
    usd_per_point  = float(cfg.get('dollars_per_point',
                                   DOLLARS_PER_POINT.get(instrument, 1.0)))
    price_dp       = int(cfg.get('price_dp',
                                 PRICE_DP.get(instrument, DEFAULT_PRICE_DP)))
    be_trigger_r   = float(cfg.get('be_trigger_r', 0.0))
    entry_tf       = str(cfg.get('entry_timeframe', '2min'))
    partials       = cfg.get('partials') or None
    partial_be     = bool(cfg.get('partial_be', True))
    # Both default True: they are entry-sequence.md rules the engine was not
    # honouring, not new behaviour. 24% of index trades were filling after
    # their killzone had closed, up to 5 hours late, for -8.1R.
    kz_expiry      = bool(cfg.get('limit_expires_with_killzone', True))
    no_chase       = bool(cfg.get('cancel_on_target', True))
    eod_hour       = int(cfg.get('eod_exit_hour', DEFAULT_EOD_EXIT_HOUR))
    costs          = CostModel(
        use_spread=bool(cfg.get('use_spread', False)),
        spread_multiplier=float(cfg.get('spread_multiplier', 1.0)),
        commission_points=float(cfg.get('commission_points', 0.0)),
        slippage_points=float(cfg.get('slippage_points', 0.0)),
    )
    use_daily_bias = cfg.get('use_daily_bias', False)

    model_config = {k: v for k, v in cfg.items() if k not in BACKTEST_KEYS}
    model        = Model2022(config=model_config)

    out_dir = os.path.join(BASE_OUT_DIR, name)
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    stem = f"fvg_sweep_{instrument.lower()}"

    # Directions to evaluate per day
    directions = (None,) if use_daily_bias else ('bearish', 'bullish')

    # ── Load data ────────────────────────────────────────────────────────────
    start_dt     = pd.Timestamp(start_date)
    warmup_start = start_dt - pd.Timedelta(weeks=WARMUP_WEEKS)
    weeks_needed = int((pd.Timestamp.now() - warmup_start).days / 7) + 4

    if df_1m_cache is not None:
        df_1m = df_1m_cache
    else:
        if verbose:
            print(f"Loading {instrument} 1m data (~{weeks_needed} weeks from {warmup_start.date()})...")
        loader = DataLoader(timeframe='1min', weeks=weeks_needed, data_dir=DATA_DIR)
        df_1m  = getattr(loader, f'read_{instrument}')()
        if verbose:
            print(f"  Loaded {len(df_1m):,} rows  {df_1m.index[0]} → {df_1m.index[-1]}")

    raw_idx_full = df_1m.index
    if raw_idx_full.tz is None:
        raw_idx_full = raw_idx_full.tz_localize('UTC')
    cutoff_utc = start_dt.tz_localize('UTC')
    df_bt = (df_1m[raw_idx_full >= cutoff_utc] if df_1m.index.tz is not None
             else df_1m[df_1m.index >= start_dt])

    raw_idx = df_bt.index
    if raw_idx.tz is None:
        raw_idx = raw_idx.tz_localize('UTC')
    ny_idx = raw_idx.tz_convert('America/New_York')
    session_mask_arr = ((ny_idx.hour * 60 + ny_idx.minute >= 8 * 60 + 30) &
                        (ny_idx.hour * 60 + ny_idx.minute <  11 * 60))
    trading_days = sorted(set(ny_idx.date[session_mask_arr]))
    if verbose:
        print(f"  {len(trading_days)} trading days  {trading_days[0]} → {trading_days[-1]}\n")

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    trade_log  = []
    viz_trades = []
    n_signals  = 0
    n_no_fill  = 0
    n_errors   = 0
    # Why non-actionable sessions died. With target_mode='pool_only' this is the
    # only way to see how many setups were complete but found no real opposing
    # pool at the R:R floor - i.e. how many genuine draws the market offered.
    funnel: dict = {}

    for i, day in enumerate(trading_days):
        session_ts = pd.Timestamp(day)
        try:
            snap = build_snapshot(df_1m, session_ts, entry_tf)
        except Exception as exc:
            n_errors += 1
            if verbose:
                print(f"  ERROR {day}: {exc}")
            continue

        for forced_dir in directions:
            try:
                signal = model.generate_signal(snap, force_direction=forced_dir)
            except Exception as exc:
                n_errors += 1
                if verbose:
                    print(f"  ERROR {day} {forced_dir}: {exc}")
                continue

            if not signal['actionable']:
                reason = next((c for c in reversed(signal.get('checks') or [])
                               if c.startswith('Rule of exclusion')), 'other')
                funnel[reason] = funnel.get(reason, 0) + 1
                continue
            n_signals += 1

            expiry = (killzone_expiry(signal.get('sweep_time'), model.config['killzones'],
                                      snap.df_2m.index.tz)
                      if kz_expiry else None)
            fill = simulate_fill(snap.df_2m, signal, session_ts, eod_hour, costs=costs,
                                 expiry=expiry, cancel_on_target=no_chase)
            if fill is None:
                n_no_fill += 1
                continue

            exit_info = simulate_exit(snap.df_2m, signal, fill, session_ts, eod_hour,
                                      be_trigger_r=be_trigger_r,
                                      partials=partials, partial_be=partial_be,
                                      costs=costs)
            pnl_pts   = exit_info.get('pnl_pts')
            if pnl_pts is None:
                pnl_pts = calc_pnl_pts(signal['direction'], fill['fill_price'],
                                       exit_info['exit_price'])
            # Spread and slippage are already inside the executed prices; the
            # recorded figures below are for attribution. Commission is a flat
            # round-turn charge and is deducted here.
            spread_cost = float(exit_info.get('spread_cost_pts', 0.0))
            slip_cost   = float(exit_info.get('slippage_cost_pts', 0.0))
            commission  = costs.commission_points
            pnl_pts    -= commission
            risk_pts_raw = abs(signal['stop'] - signal['entry'])
            r_multiple  = (pnl_pts / risk_pts_raw) if risk_pts_raw > 0 else 0.0
            exit_class  = classify_exit(exit_info['exit_reason'], pnl_pts)
            pnl_usd   = round(pnl_pts * usd_per_point, 2)

            trade_log.append({
                'session_date': str(day),
                'direction':    signal['direction'],
                'entry_price':  round(fill['fill_price'], price_dp),
                'entry_time':   str(fill['fill_time']),
                'stop':         round(signal['stop'], price_dp),
                'target':       round(signal['target'], price_dp),
                'exit_price':   round(exit_info['exit_price'], price_dp),
                'exit_time':    str(exit_info['exit_time']),
                'exit_reason':  exit_info['exit_reason'],
                'risk_pts':     round(abs(signal['stop'] - signal['entry']), price_dp),
                'reward_pts':   round(abs(signal['target'] - signal['entry']), price_dp),
                'pnl_pts':      round(pnl_pts, price_dp),
                'pnl_usd':      pnl_usd,
                # Diagnostics: which pool was raided, when, and how far price ran
                # each way before the exit. Needed to tell a setup that never
                # worked from one that worked and then gave it back.
                'exit_class':   exit_class,
                'R':            round(r_multiple, 3),
                'holding_minutes': round(
                    (pd.Timestamp(exit_info['exit_time']) -
                     pd.Timestamp(fill['fill_time'])).total_seconds() / 60.0, 1),
                'spread_cost_pts': round(spread_cost, price_dp),
                'commission_pts':  round(commission, price_dp),
                'slippage_cost_pts': round(slip_cost, price_dp),
                'total_cost_pts':  round(spread_cost + slip_cost + commission, price_dp),
                'swept_pool':   signal.get('swept_pool'),
                # The raided pool price and the swing the MSS broke, so a trade
                # can be reconstructed from the log without re-running.
                'sweep_level':  signal.get('sweep_level'),
                'mss_level':    signal.get('bms_swing_level'),
                **{k: (round(signal[k], price_dp) if isinstance(signal.get(k), float)
                       else signal.get(k))
                   for k in ('sweep_pool_level', 'sweep_penetration_pts',
                             'sweep_bar_range', 'sweep_wick_pts',
                             'sweep_close_inside',
                             'mss_size_pts', 'mss_body_pts', 'mss_range_pts',
                             'nearest_pool_price')},
                **{k: (round(signal[k], 4) if isinstance(signal.get(k), float)
                       else signal.get(k))
                   for k in ('sweep_wick_pct', 'mss_body_pct', 'mss_body_mult',
                             'mss_opposing_wick_pct', 'nearest_pool_rr',
                             'real_target_rr', 'planned_rr',
                             'fvg_size_pts', 'fvg_size_atr')},
                'mss_to_sweep_bars':    signal.get('mss_to_sweep_bars'),
                'entry_array':          signal.get('entry_array'),
                'synthetic_target_used': signal.get('synthetic_target_used'),
                'n_live_target_pools':  signal.get('n_live_target_pools'),
                'sweep_time':   str(signal.get('sweep_time')),
                'killzone':     _kz_label(signal.get('sweep_time')),
                'mae_pts':      round(exit_info.get('mae_pts', 0.0), price_dp),
                'mfe_pts':      round(exit_info.get('mfe_pts', 0.0), price_dp),
                'mfe_r':        round(exit_info.get('mfe_pts', 0.0) /
                                      abs(signal['stop'] - signal['entry']), 2)
                                if signal['stop'] != signal['entry'] else 0.0,
                'bars_held':    exit_info.get('bars_held', 0),
            })

            if save_outputs:
                viz_trades.append(TradeViz(
                    label=f"{day}  {signal['direction'].upper()}  ${pnl_usd:,.0f}",
                    panels=trade_panels(snap, session_ts),
                    marks=signal_to_marks(signal, fill, exit_info),
                    pnl=pnl_usd,
                ))

        if verbose and (i + 1) % 50 == 0:
            pnl_so_far = sum(t['pnl_usd'] for t in trade_log)
            print(f"  Day {i+1:>3}/{len(trading_days)}  "
                  f"signals={n_signals}  trades={len(trade_log)}  "
                  f"P&L=${pnl_so_far:,.0f}")

    if verbose:
        print(f"\n── Summary ──────────────────────────────────────────")
        print(f"  Days evaluated : {len(trading_days)}")
        print(f"  Signals fired  : {n_signals}")
        print(f"  No-fill        : {n_no_fill}")
        print(f"  Filled trades  : {len(trade_log)}")
        print(f"  Errors         : {n_errors}")
        if funnel:
            print("  Rejections:")
            for reason, cnt in sorted(funnel.items(), key=lambda kv: -kv[1]):
                print(f"    {cnt:>5}  {reason}")

    if not trade_log:
        if verbose:
            print("\nNo trades executed.")
        return {'name': name, 'config': cfg, 'n_trades': 0,
                'n_signals': n_signals, 'funnel': funnel}

    trades = pd.DataFrame(trade_log)
    stats  = compute_stats(trades, price_dp=price_dp)

    if verbose:
        print()
        print_stats(stats, f'{name}  ({instrument}, {start_date}→)', price_dp=price_dp)
        trades['month'] = pd.to_datetime(trades['session_date']).dt.to_period('M')
        print(f"\n  {'Month':<10} {'Trades':>6}  {'Wins':>4}  {'P&L pts':>9}  {'P&L $':>9}")
        print(f"  {'-'*45}")
        for month, grp in trades.groupby('month'):
            w = (grp['exit_reason'] == 'target').sum()
            print(f"  {str(month):<10} {len(grp):>6}  {w:>4}  "
                  f"{grp['pnl_pts'].sum():>9.{price_dp}f}  {grp['pnl_usd'].sum():>9,.0f}")

    if save_outputs:
        log_path = os.path.join(out_dir, f'{stem}_trades.csv')
        trades.to_csv(log_path, index=False)
        if verbose:
            print(f"\nTrade log → {log_path}")

        chart_path = os.path.join(out_dir, f'{stem}_equity.png')
        plot_equity_curve(
            trades, chart_path,
            title=f'{name} — {instrument}  {start_date}→  (1 unit, ${usd_per_point:g}/pt)',
        )
        if verbose:
            print(f"Equity curve → {chart_path}")

        report_stats = {
            'Trades':          stats['total_trades'],
            'Win rate':        f"{stats['win_rate']}%  ({stats['wins']}W / {stats['losses']}L / {stats['eods']} EOD)",
            'Total P&L':       f"${stats['total_pnl_usd']:,.2f}  ({stats['total_pnl_pts']:.{price_dp}f} pts)",
            'Avg R:R (setup)': stats['avg_rr'],
            'Profit factor':   stats['profit_factor'],
            'Max drawdown':    f"${stats['max_dd_usd']:,.2f}",
            'Sharpe':          stats['sharpe_ratio'],
        }
        report_path = os.path.join(out_dir, f'{stem}_report.html')
        build_report(
            trades=viz_trades,
            panel_order=['daily', '15m', '2m'],
            out_path=report_path,
            title=f'{name} — {instrument}  {start_date}→',
            equity=trades['pnl_usd'].cumsum(),
            stats=report_stats,
        )
        if verbose:
            print(f"Interactive report → {report_path}")

    # The trade list rides along so callers can slice by date without a
    # re-run: sessions are independent (no carried position, no equity-based
    # sizing), so a date slice of this log equals a backtest of that range.
    return {'name': name, 'config': cfg, 'n_trades': len(trade_log),
            'n_signals': n_signals, 'trades': trade_log,
            'funnel': funnel, **stats}


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FVG Sweep backtest — ICT 2022 Model')
    parser.add_argument('--config', default=None,
                        help='Path to a YAML experiment config file')
    args = parser.parse_args()

    run_cfg = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            run_cfg = yaml.safe_load(f) or {}
        print(f"Config: {args.config}  ({run_cfg.get('name', '?')})")

    run_backtest(run_cfg)
