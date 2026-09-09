#!/usr/bin/env python3
"""
Judas Swing Backtester
======================

Walk-forward backtest of the ICT Judas Swing model. One evaluation per
(day, session): the London Judas at 02:00 NY and the New York Judas at the
08:20 CME open are separate setups with separate pools, separate deadlines and
separate exit times, so they are scored separately.

Usage:
  python judas_backtest.py --config configs/judas_fx.yaml
  python judas_backtest.py --config configs/judas_fx.yaml --quiet

Config keys consumed by the runner (everything else is forwarded to the model):
  name              str    experiment name -> results/runs/<name>/
  description       str    human note
  start_date        str    backtest start, e.g. '2024-05-01'
  end_date          str    optional backtest end (walk-forward splits)
  instrument        str    DataLoader key: 'EURUSD', 'USATECHIDXUSD', 'NQ', ...
  dollars_per_point float  USD per 1.00 price move
  price_dp          int    decimals kept in the trade log
  entry_timeframe   str    entry/reversal bar size, default '2min'
  use_daily_bias    bool   True = bias gate; False = evaluate both sides
  be_trigger_r      float  move stop to breakeven after this many R (0 = off)
  partials          list   [[fraction, R], ...] partial-take schedule
  partial_be        bool   breakeven after the first partial fills (default True)
  one_trade_per_day bool   keep only the first signal of the day (default True)

Outputs (results/runs/<name>/):
  judas_<instrument>_trades.csv   full trade log
  judas_<instrument>_equity.png   equity curve
  judas_<instrument>_report.html  interactive multi-timeframe report
"""

import argparse
import os
import pathlib as _pl
import sys

import pandas as pd

from typing import Optional

_HERE = _pl.Path(__file__).resolve().parent
_ROOT = next(p for p in _pl.Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())

# Only the equity plot is borrowed from the 2022 model's runner; fills, exits,
# costs and statistics now come from infrastructure/, so both models can be
# scored the same way.
sys.path.insert(0, str(_ROOT / 'models' / 'ict' / 'model_2022' / 'backtests'))
from plot_trades import localize_to, plot_equity_curve      # noqa: E402

from infrastructure import DataLoader                       # noqa: E402
from infrastructure.execution import (CostModel, ZERO_COST,  # noqa: E402
                                      cost_model_from_config, simulate_entry,
                                      simulate_exit, trade_costs)
from infrastructure import stats as stats_mod               # noqa: E402
from infrastructure.report import Mark, TradeViz, build_report  # noqa: E402
from models.ict.judas_swing import JudasSwingModel, JudasSnapshot  # noqa: E402


# ─── Constants ───────────────────────────────────────────────────────────────

DOLLARS_PER_POINT = {
    'NQ': 20.0, 'ES': 50.0, 'YM': 5.0,
    # Spot FX, 1 standard lot: a 1.00 move on a USD-quoted pair is $100,000.
    'EURUSD': 100000.0, 'GBPUSD': 100000.0, 'AUDUSD': 100000.0,
    # USD-base pairs are quoted in the counter currency; these approximate one
    # standard lot at 2024-2026 rates. Override per config for precision.
    'USDJPY': 640.0, 'USDCHF': 125000.0, 'USDCAD': 72000.0,
    # Index CFDs: 1 contract = 1 unit of the index.
    'USATECHIDXUSD': 1.0, 'USA500IDXUSD': 1.0, 'USA30IDXUSD': 1.0,
}
PRICE_DP = {
    'NQ': 2, 'ES': 2, 'YM': 2,
    'USDJPY': 4, 'USATECHIDXUSD': 3, 'USA500IDXUSD': 3, 'USA30IDXUSD': 3,
}
DEFAULT_PRICE_DP = 6
WARMUP_WEEKS = 10

# Forced-flat hour, NY clock. The Judas model normally exits on its session's own
# `exit_ny` (12:00 after a London setup, 16:00 after a New York one), which is
# ICT's "collapse intraday positions from 10:00-11:00" rule applied per session.
# `eod_exit_hour` in a config overrides that for every session; 16 is the value
# the previous engine hardcoded, so leaving it unset preserves prior behaviour.
DEFAULT_EOD_EXIT_HOUR = 16
EXIT_TZ = 'America/New_York'

DATA_DIR = str(_ROOT / 'Data')
BASE_OUT_DIR = str(_HERE / 'results' / 'runs')

BACKTEST_KEYS = {
    'name', 'description', 'start_date', 'end_date', 'instrument',
    'dollars_per_point', 'price_dp', 'entry_timeframe', 'use_daily_bias',
    'be_trigger_r', 'partials', 'partial_be', 'one_trade_per_day',
    'pip_scaled', 'costs', 'eod_exit_hour',
}

C_ENTRY, C_STOP, C_TARGET = '#1f77b4', '#c0392b', '#1a8a4a'
C_FVG, C_JUDAS, C_REV = '#f39c12', '#8e44ad', '#16a085'


# Price-based model params are quoted in points on an index and pips on FX, so a
# single config cannot be shared across instruments unless it says which. With
# `pip_scaled: true` the *_pts keys are written in pips (or index points) and
# scaled here by the instrument's own tick size, so one doctrine file runs on all
# nine instruments without nine hand-tuned copies drifting apart.
PIP_SCALED_KEYS = ('stop_buffer_pts', 'max_stop_pts', 'min_stop_pts')


COST_PIP_SCALED_KEYS = ('fixed_spread', 'slippage_entry', 'slippage_exit',
                        'slippage_stop')


def apply_pip_scaling_costs(cfg: dict, instrument: str) -> dict:
    """Scale the `costs:` section's price-unit fields when pip_scaled is set.

    Slippage written as "1" means one pip on FX and one index point on a CFD,
    the same convention the model's own *_pts parameters use.
    """
    section = (cfg or {}).get('costs')
    if not section or not cfg.get('pip_scaled'):
        return cfg
    pip = DataLoader(timeframe='1min').pip(instrument)
    scaled = dict(section)
    for k in COST_PIP_SCALED_KEYS:
        if k in scaled and scaled[k] is not None:
            scaled[k] = float(scaled[k]) * pip
    return {**cfg, 'costs': scaled}


def apply_pip_scaling(cfg: dict, instrument: str) -> dict:
    if not cfg.get('pip_scaled'):
        return cfg
    pip = DataLoader(timeframe='1min').pip(instrument)
    out = dict(cfg)
    for k in PIP_SCALED_KEYS:
        if k in out and out[k] is not None:
            out[k] = float(out[k]) * pip
    return out


# ─── Snapshot ────────────────────────────────────────────────────────────────

_AGG = {'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'}
# The per-bar spread is the only record of what crossing the book cost at that
# moment. It has to be carried through every resample or the execution model
# silently falls back to mid prices. Averaged, not summed: a position is opened
# at one instant inside the bar, not across all of it.
_SPREAD_AGG = 'mean'


def precompute_frames(df_1m: pd.DataFrame, entry_tf: str = '2min') -> dict:
    """Resample the whole series once, so a session snapshot is just a slice.

    Resample boundaries are absolute, so slicing a whole-series resample gives
    exactly the bars a per-session resample would - at a fraction of the cost.
    Building 900 snapshots the naive way re-resamples 45 days of 1-minute bars
    900 times over, which is what made the parameter grid unusable.
    """
    agg = dict(_AGG)
    if 'spread' in df_1m.columns:
        agg['spread'] = _SPREAD_AGG
    return {
        'entry': df_1m.resample(entry_tf).agg(agg).dropna(subset=['close']),
        '15m': df_1m.resample('15min').agg(agg).dropna(subset=['close']),
        'daily': df_1m.resample('D').agg(agg).dropna(subset=['close']),
        '4h': df_1m.resample('4h').agg(agg).dropna(subset=['close']),
    }


def build_snapshot(df_1m: pd.DataFrame, session_ts: pd.Timestamp,
                   entry_tf: str = '2min',
                   frames: dict | None = None) -> JudasSnapshot:
    """Slice the bars one session needs.

    The entry frame starts 3 days back so the Asian range (prior 20:00-00:00 NY)
    and the prior-day pools sit inside it; daily and 4H go back 45 days for the
    bias read and the five-day ADR. Daily and 4H stop one minute before the
    session date so nothing from the day being traded leaks into its own bias.

    `frames` is the output of precompute_frames(); pass it when evaluating many
    sessions.
    """
    d_naive = pd.Timestamp(session_ts.date())
    d = d_naive.tz_localize('UTC') if df_1m.index.tz is not None else d_naive
    f = frames if frames is not None else precompute_frames(df_1m, entry_tf)

    return JudasSnapshot(
        df_entry=f['entry'].loc[d - pd.Timedelta(days=3): d + pd.Timedelta(days=1)],
        df_15m=f['15m'].loc[d - pd.Timedelta(days=14): d + pd.Timedelta(days=1)],
        df_daily=f['daily'].loc[d - pd.Timedelta(days=45): d - pd.Timedelta(minutes=1)],
        session_date=session_ts,
        df_4h=f['4h'].loc[d - pd.Timedelta(days=45): d - pd.Timedelta(minutes=1)],
    )


def session_eod(session_ts: pd.Timestamp, signal: dict, index,
                eod_exit_hour: Optional[int] = None) -> pd.Timestamp:
    """The forced-flat timestamp for one signal, as an explicit NY wall time.

    Resolution order:
      1. `eod_exit_hour` from the config, if given - applies to every session.
      2. the session profile's own `exit_ny` (12:00 for London, 16:00 for NY).
      3. DEFAULT_EOD_EXIT_HOUR.

    The hour is built in `America/New_York` and then converted to the bar
    index's timezone, so DST is handled by the zoneinfo database rather than by
    an offset assumption. The result is clamped to the end of the session day:
    an exit hour can never push a position past the day it was opened in.
    """
    if eod_exit_hour is not None:
        hh, mm = int(eod_exit_hour), 0
    else:
        raw = str(signal.get('exit_ny') or f'{DEFAULT_EOD_EXIT_HOUR:02d}:00')
        parts = raw.split(':')
        hh, mm = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    if not 0 <= hh <= 23 or not 0 <= mm <= 59:
        raise ValueError(f"EOD exit time {hh:02d}:{mm:02d} is not a valid NY wall time")

    day = pd.Timestamp(session_ts.date())
    eod = pd.Timestamp(f'{day.date()} {hh:02d}:{mm:02d}:00', tz=EXIT_TZ)
    day_end = pd.Timestamp(f'{day.date()} 23:59:59', tz=EXIT_TZ)
    eod = min(eod, day_end)
    return eod.tz_convert(index.tz) if index.tz is not None else eod.tz_localize(None)


def execute_signal(snap: JudasSnapshot, signal: dict, session_ts: pd.Timestamp,
                   cost: CostModel = ZERO_COST,
                   usd_per_point: float = 1.0,
                   eod_exit_hour: Optional[int] = None,
                   be_trigger_r: float = 0.0,
                   partials=None, partial_be: bool = True) -> Optional[dict]:
    """Turn one signal into a fully costed trade-log row, or None if unfilled.

    This is the single execution path: the runner, the parameter grid and the
    basket runner all call it, so a change to fills or costs cannot apply to one
    and not the others.
    """
    df = snap.df_entry
    deadline = (None if signal.get('entry_style') == 'market'
                else localize_to(signal['entry_deadline'], df.index))
    fill = simulate_entry(df, signal, deadline, cost)
    if fill is None:
        return None

    eod = session_eod(session_ts, signal, df.index, eod_exit_hour)
    ex = simulate_exit(df, signal, fill, eod, cost,
                       be_trigger_r=be_trigger_r,
                       partials=partials, partial_be=partial_be)

    risk = abs(float(signal['stop']) - float(signal['entry']))
    costs = trade_costs(ex, signal['direction'], fill['fill_mid'], cost, usd_per_point)
    held = (pd.Timestamp(ex['exit_time']) - pd.Timestamp(fill['fill_time']))

    return {
        'session_date': str(session_ts.date()),
        'weekday': session_ts.day_name()[:3],
        'session': signal['session'],
        'direction': signal['direction'],
        'entry_time': str(fill['fill_time']),
        # When the setup became actionable, as distinct from when the limit
        # filled. The gap between the two is what says whether a fill leaned
        # on the bar that confirmed the pattern (see f2_exposure.py).
        'signal_time': str(signal['entry_time']),
        'sweep_to_fill_min': (pd.Timestamp(fill['fill_time'])
                              - pd.Timestamp(signal['sweep_time'])
                              ).total_seconds() / 60.0,
        'exit_time': str(ex['exit_time']),
        'entry_price': fill['fill_price'],
        'entry_mid': fill['fill_mid'],
        'exit_price': ex['exit_price'],
        'exit_mid': ex['exit_mid'],
        'stop': float(signal['stop']),
        'target': float(signal['target']),
        'risk_pts': risk,
        'reward_pts': abs(float(signal['target']) - float(signal['entry'])),
        'planned_rr': float(signal.get('rr', 0.0)),
        'pnl_pts': ex['pnl_pts'],
        'pnl_r': (ex['pnl_pts'] / risk) if risk else 0.0,
        'pnl_usd': ex['pnl_pts'] * usd_per_point,
        'exit_reason': ex['exit_reason'],
        'raw_exit_reason': ex['raw_exit_reason'],
        'holding_minutes': held.total_seconds() / 60.0,
        'bars_held': ex['bars_held'],
        'mae_pts': ex['mae_pts'],
        'mfe_pts': ex['mfe_pts'],
        'mae_r': (ex['mae_pts'] / risk) if risk else 0.0,
        'mfe_r': (ex['mfe_pts'] / risk) if risk else 0.0,
        'swept_pool': signal.get('swept_pool'),
        'sweep_time': str(signal.get('sweep_time')),
        'confirmation': signal.get('confirmation'),
        'entry_style': signal.get('entry_style', 'limit'),
        'judas_depth': signal.get('judas_depth_class'),
        'adr5': signal.get('adr5'),
        **costs,
        '_fill': fill,
        '_exit': ex,
    }


# ─── Report adapter ──────────────────────────────────────────────────────────

def _window(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return df.loc[localize_to(lo, df.index):localize_to(hi, df.index)]


def trade_panels(snap: JudasSnapshot, session_ts: pd.Timestamp) -> dict:
    d = pd.Timestamp(session_ts.date())
    day_bar = (snap.df_entry
               .resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                   'close': 'last', 'volume': 'sum'})
               .dropna(subset=['close']))
    d_loc = localize_to(d, day_bar.index)
    day_bar = day_bar[day_bar.index.normalize() == d_loc]
    daily = pd.concat([snap.df_daily.tail(30), day_bar])
    daily = daily[~daily.index.duplicated(keep='last')]
    return {
        'daily': daily,
        '15m': _window(snap.df_15m, d - pd.Timedelta(days=5), d + pd.Timedelta(days=1)),
        'entry': _window(snap.df_entry, d - pd.Timedelta(days=1), d + pd.Timedelta(days=1)),
    }


def signal_to_marks(signal: dict, fill: dict, exit_info: dict) -> list:
    g = 'Entry/Stop/Target'
    marks = [
        Mark('level', g, 'entry', f"Entry {signal['entry']:.5f}", C_ENTRY,
             price=signal['entry']),
        Mark('level', g, 'entry', f"Stop {signal['stop']:.5f}", C_STOP,
             dash='dash', price=signal['stop']),
        Mark('level', g, 'entry', f"Target {signal['target']:.5f}", C_TARGET,
             dash='dash', price=signal['target']),
        Mark('marker', g, 'entry', "Fill", C_ENTRY, time=fill['fill_time'],
             price=fill['fill_price'], symbol='circle'),
    ]
    if exit_info:
        col = (C_TARGET if exit_info['exit_reason'] in ('target', 'partial')
               else C_STOP if exit_info['exit_reason'] == 'stop' else '#7f8c8d')
        marks.append(Mark('marker', g, 'entry', f"Exit ({exit_info['exit_reason']})",
                          col, time=exit_info['exit_time'],
                          price=exit_info['exit_price'], symbol='x'))

    marks.append(Mark('zone', 'Entry array', 'entry', "Entry array", C_FVG,
                      y0=signal['fvg_bottom'], y1=signal['fvg_top'],
                      t0=signal['entry_time'],
                      t1=(exit_info['exit_time'] if exit_info else None)))

    lbl = f"Judas raid: {signal['swept_pool']}"
    marks += [
        Mark('marker', 'Judas', 'entry', lbl, C_JUDAS, time=signal['sweep_time'],
             price=signal['judas_extreme'], symbol='x', default_on=True),
        Mark('level', 'Judas', 'entry', f"Raided pool {signal['sweep_level']:.5f}",
             C_JUDAS, dash='dot', price=signal['sweep_level'], default_on=True),
        Mark('level', 'Judas', '15m', f"Raided pool {signal['sweep_level']:.5f}",
             C_JUDAS, dash='dot', price=signal['sweep_level'], default_on=True),
        Mark('level', 'Judas', 'entry', f"Session open {signal['anchor_price']:.5f}",
             '#7f8c8d', dash='dot', price=signal['anchor_price']),
    ]

    sym = 'triangle-up' if signal['direction'] == 'long' else 'triangle-down'
    marks += [
        Mark('marker', 'Reversal', 'entry',
             f"{signal['confirmation']} {signal['bms_swing_level']:.5f}", C_REV,
             time=signal['bms_time'], price=signal['bms_swing_level'],
             symbol=sym, default_on=True),
        Mark('level', 'Reversal', 'entry', f"{signal['confirmation']} level", C_REV,
             dash='dot', price=signal['bms_swing_level'], default_on=True),
    ]
    return marks


# ─── Main loop ───────────────────────────────────────────────────────────────

def run_backtest(run_config: dict = None, verbose: bool = True,
                 save_outputs: bool = True,
                 df_1m_cache: pd.DataFrame = None) -> dict:
    cfg = run_config or {}

    name = cfg.get('name', 'judas_baseline')
    start_date = cfg.get('start_date', '2024-05-01')
    end_date = cfg.get('end_date')
    instrument = cfg.get('instrument', 'EURUSD').upper()
    usd_per_point = float(cfg.get('dollars_per_point',
                                  DOLLARS_PER_POINT.get(instrument, 1.0)))
    price_dp = int(cfg.get('price_dp', PRICE_DP.get(instrument, DEFAULT_PRICE_DP)))
    entry_tf = str(cfg.get('entry_timeframe', '2min'))
    use_daily_bias = bool(cfg.get('use_daily_bias', True))
    be_trigger_r = float(cfg.get('be_trigger_r', 0.0))
    partials = cfg.get('partials') or None
    partial_be = bool(cfg.get('partial_be', True))
    one_per_day = bool(cfg.get('one_trade_per_day', True))
    eod_exit_hour = cfg.get('eod_exit_hour')
    if eod_exit_hour is not None:
        eod_exit_hour = int(eod_exit_hour)
    cost = cost_model_from_config(apply_pip_scaling_costs(cfg, instrument))

    model_config = apply_pip_scaling(
        {k: v for k, v in cfg.items() if k not in BACKTEST_KEYS}, instrument)
    model = JudasSwingModel(config=model_config)
    sessions = model.config['sessions']
    directions = (None,) if use_daily_bias else ('bullish', 'bearish')

    out_dir = os.path.join(BASE_OUT_DIR, name)
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)
    stem = f"judas_{instrument.lower()}"

    # ── Data ─────────────────────────────────────────────────────────────────
    start_dt = pd.Timestamp(start_date)
    warmup_start = start_dt - pd.Timedelta(weeks=WARMUP_WEEKS)
    if df_1m_cache is not None:
        df_1m = df_1m_cache
    else:
        weeks = int((pd.Timestamp.now() - warmup_start).days / 7) + 4
        if verbose:
            print(f"Loading {instrument} 1m data (~{weeks} weeks)...")
        loader = DataLoader(timeframe='1min', weeks=weeks, data_dir=DATA_DIR)
        df_1m = loader.read(instrument)
        if verbose:
            print(f"  {len(df_1m):,} rows  {df_1m.index[0]} -> {df_1m.index[-1]}")

    idx = df_1m.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    ny_idx = idx.tz_convert('America/New_York')
    in_range = ny_idx >= start_dt.tz_localize('America/New_York')
    if end_date:
        in_range &= ny_idx <= pd.Timestamp(end_date).tz_localize('America/New_York')
    trading_days = sorted(set(ny_idx.date[in_range]))
    if not trading_days:
        raise ValueError(f"No bars in {start_date} -> {end_date or 'end'}")
    if verbose:
        print(f"  {len(trading_days)} days  {trading_days[0]} -> {trading_days[-1]}\n")

    # ── Walk forward ─────────────────────────────────────────────────────────
    trade_log, viz_trades = [], []
    n_signals = n_no_fill = n_errors = 0

    frames = precompute_frames(df_1m, entry_tf)

    for i, day in enumerate(trading_days):
        session_ts = pd.Timestamp(day)
        try:
            snap = build_snapshot(df_1m, session_ts, entry_tf, frames)
        except Exception as exc:
            n_errors += 1
            if verbose:
                print(f"  ERROR {day}: {exc}")
            continue
        if len(snap.df_entry) < 50 or len(snap.df_daily) < 10:
            continue

        traded_today = False
        for session in sessions:
            if traded_today and one_per_day:
                break
            for forced in directions:
                try:
                    signal = model.generate_signal(snap, session=session,
                                                   force_direction=forced)
                except Exception as exc:
                    n_errors += 1
                    if verbose:
                        print(f"  ERROR {day} {session} {forced}: {exc}")
                    continue
                if not signal['actionable']:
                    continue
                n_signals += 1

                row = execute_signal(
                    snap, signal, session_ts, cost=cost,
                    usd_per_point=usd_per_point, eod_exit_hour=eod_exit_hour,
                    be_trigger_r=be_trigger_r,
                    partials=partials, partial_be=partial_be)
                if row is None:
                    n_no_fill += 1
                    continue
                fill, exit_info = row.pop('_fill'), row.pop('_exit')
                trade_log.append(row)
                traded_today = True

                if save_outputs:
                    viz_trades.append(TradeViz(
                        label=f"{day} {signal['session']} "
                              f"{signal['direction'].upper()} "
                              f"{row['pnl_r']:+.2f}R",
                        panels=trade_panels(snap, session_ts),
                        marks=signal_to_marks(signal, fill, exit_info),
                        pnl=row['pnl_usd'],
                    ))
                break  # one direction per session

        if verbose and (i + 1) % 100 == 0:
            print(f"  Day {i+1:>4}/{len(trading_days)}  signals={n_signals}  "
                  f"trades={len(trade_log)}  "
                  f"P&L=${sum(t['pnl_usd'] for t in trade_log):,.0f}")

    if verbose:
        print(f"\n-- Summary --------------------------------------")
        print(f"  Days evaluated : {len(trading_days)}")
        print(f"  Signals fired  : {n_signals}")
        print(f"  No-fill        : {n_no_fill}")
        print(f"  Filled trades  : {len(trade_log)}")
        print(f"  Errors         : {n_errors}")

    if not trade_log:
        if verbose:
            print("\nNo trades executed.")
        return {'name': name, 'config': cfg, 'n_trades': 0, 'n_signals': n_signals}

    trades = pd.DataFrame(trade_log)
    stats_mod.validate_trade_log(trades)
    stats = stats_mod.compute_stats(trades, cost_model=cost)

    if verbose:
        stats_mod.print_stats(
            stats, f'{name}  ({instrument}, {start_date}->{end_date or "end"})')
        print("\nExit reasons")
        print(stats_mod.exit_reason_table(trades).to_string(index=False))
        for col in ('session', 'direction', 'swept_pool', 'weekday',
                    'judas_depth', 'month', 'year'):
            table = stats_mod.breakdown(trades, col)
            if len(table):
                print(f"\nBy {col}")
                print(table.to_string(index=False))

    if save_outputs:
        log_path = os.path.join(out_dir, f'{stem}_trades.csv')
        trades.round({c: price_dp for c in trades.columns
                      if trades[c].dtype.kind == 'f'}).to_csv(log_path, index=False)
        chart_path = os.path.join(out_dir, f'{stem}_equity.png')
        plot_equity_curve(trades, chart_path,
                          title=f'{name} - {instrument} (1 unit, ${usd_per_point:g}/pt)')
        report_path = os.path.join(out_dir, f'{stem}_report.html')
        build_report(
            trades=viz_trades, panel_order=['daily', '15m', 'entry'],
            out_path=report_path, title=f'{name} - {instrument} Judas Swing',
            equity=trades['pnl_r'].cumsum(),
            stats={
                'Trades': stats['total_trades'],
                'Profitable': f"{stats['win_pct']}%  ({stats['wins']}W / "
                              f"{stats['losses']}L / {stats['breakeven']}BE)",
                'Target hit': f"{stats['target_hit_pct']}%",
                'Expectancy': f"{stats['expectancy_r']:+.3f} R",
                'Total R': stats['total_r'],
                'Profit factor': stats['profit_factor'],
                'Max drawdown': f"{stats['max_dd_r']:.2f} R",
                'Sharpe / Sortino': f"{stats['sharpe']} / {stats['sortino']}",
                'Cost model': stats['cost_model'],
                'Execution cost': f"${stats['exec_cost_usd']:,.2f}",
            },
        )
        if verbose:
            print(f"\nTrade log  -> {log_path}")
            print(f"Equity     -> {chart_path}")
            print(f"Report     -> {report_path}")

    return {'name': name, 'config': cfg, 'n_trades': len(trade_log),
            'n_signals': n_signals, **stats}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Judas Swing backtest')
    ap.add_argument('--config', default=None, help='YAML experiment config')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    run_cfg = {}
    if args.config:
        import yaml
        path = args.config if os.path.isabs(args.config) else str(_HERE / args.config)
        with open(path) as f:
            run_cfg = yaml.safe_load(f) or {}
        print(f"Config: {path}  ({run_cfg.get('name', '?')})")

    run_backtest(run_cfg, verbose=not args.quiet, save_outputs=not args.no_save)
