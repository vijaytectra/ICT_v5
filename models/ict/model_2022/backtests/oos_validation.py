"""E6 on everything: the 2015-2023 index history it has never seen, plus six FX
pairs, against the 2024-2026 indices it was built on.

E6 = ONS + REH/REL pools, 2.0R target floor, FVG/OB/breaker entry arrays, near-
edge entry, wick stop with a structural buffer, daily bias gate, spread charged.
Every one of those choices was made by searching ~25 configurations on the
2024-2026 index data, so that period is in-sample and proves nothing on its own.

2015-2023 is the real test: same three instruments, nine untouched years,
roughly 2,250 sessions against the 600 the design was fitted to.

Price-scaled params are re-measured on each dataset by the same formula used
throughout (eq ~ 0.5x median 15m range, max_stop ~ 2.5x median 15m ATR,
buffer ~ 1.5x median 2m ATR). That is a volatility recalibration, not a
performance tune - Nasdaq traded near 7,600 in that era against 29,500 now, so
frozen point values would fail for arithmetic reasons that say nothing about
the strategy.
"""
import os, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = r"C:\personal\ICT_v5"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "ict", "model_2022", "backtests"))

E6 = dict(
    use_daily_bias=True,
    killzones=[('02:00','05:00'), ('08:30','11:00'), ('13:30','16:00')],
    use_ons=True, use_reh_rel=True,
    use_pdh_pdl=False, use_chicago_prior=False, use_chicago_same_day=False,
    use_opening_range=False, use_asian_range=False,
    fvg_entry_depth='edge', entry_arrays=['fvg','ob','breaker'],
    array_lookback_bars=300,
    stop_mode='wick', target_mode='liquidity', target_min_rr=2.0,
    use_spread=True, commission_points=0.0, slippage_points=0.0,
)

# (symbol, start_date, eq, max_stop, buffer, $/pt, price_dp, group)
JOBS = [
 ('USATECHIDXUSD_2015','2015-05-01',  4.37,  24.60,  4.88, 1.0, 2, 'OOS 2015-2023'),
 ('USA500IDXUSD_2015', '2015-05-01',  1.25,   7.03,  1.38, 1.0, 2, 'OOS 2015-2023'),
 ('USA30IDXUSD_2015',  '2015-05-01', 10.52,  59.27, 11.77, 1.0, 2, 'OOS 2015-2023'),
 ('USATECHIDXUSD',     '2024-05-01', 17.0,   95.0,  19.0,  1.0, 2, 'in-sample 2024-26'),
 ('USA500IDXUSD',      '2024-05-01',  3.0,   17.0,   3.3,  1.0, 2, 'in-sample 2024-26'),
 ('USA30IDXUSD',       '2024-05-01', 19.0,  110.0,  21.0,  1.0, 2, 'in-sample 2024-26'),
 ('EURUSD','2024-05-01',0.00024,0.00133,0.00026,100000.0,6,'FX 2024-26'),
 ('GBPUSD','2024-05-01',0.00031,0.00171,0.00034,100000.0,6,'FX 2024-26'),
 ('USDJPY','2024-05-01',0.04425,0.24759,0.04768,   640.0,4,'FX 2024-26'),
 ('USDCHF','2024-05-01',0.00022,0.00119,0.00023,100000.0,6,'FX 2024-26'),
 ('USDCAD','2024-05-01',0.00025,0.00130,0.00026,100000.0,6,'FX 2024-26'),
 ('AUDUSD','2024-05-01',0.00022,0.00117,0.00023,100000.0,6,'FX 2024-26'),
]

def _run(job):
    sym, start, eq, ms, buf, dpp, dp, grp = job
    import pandas as pd
    from fvg_sweep_backtest import run_backtest, DATA_DIR, WARMUP_WEEKS
    from infrastructure import DataLoader
    s0 = pd.Timestamp(start)
    weeks = int((pd.Timestamp.now() - (s0 - pd.Timedelta(weeks=WARMUP_WEEKS))).days/7)+4
    df = DataLoader(timeframe='1min', weeks=weeks, data_dir=DATA_DIR).read(sym)
    cfg = {**E6, 'instrument': sym, 'start_date': start, 'name': 'x',
           'eq_tolerance_pts': eq, 'max_stop_pts': ms, 'stop_fixed_pts': ms,
           'stop_buffer_pts': buf, 'dollars_per_point': dpp, 'price_dp': dp}
    try:
        st = run_backtest(run_config=cfg, verbose=False, save_outputs=False, df_1m_cache=df)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return {'sym': sym, 'group': grp, 'error': f"{type(exc).__name__}: {exc}"}
    return {'sym': sym, 'group': grp, 'trades': st.get('total_trades'),
            'prof%': st.get('profitable_pct'), 'tp%': st.get('win_rate'),
            'avg_R': st.get('avg_r'), 'total_R': st.get('total_r'),
            'PF': st.get('profit_factor'), 'net': st.get('total_pnl_usd'),
            'maxDD': st.get('max_dd_usd'), 'sharpe': st.get('sharpe_ratio'),
            'sortino': st.get('sortino_ratio'), 'maxloss_streak': st.get('max_loss_streak')}

if __name__ == '__main__':
    print(f"{len(JOBS)} runs", flush=True)
    out=[]; dest=os.path.join(os.path.dirname(__file__),'oos.json')
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(_run,j): j for j in JOBS}
        for i,f in enumerate(as_completed(futs),1):
            try: r=f.result()
            except Exception as e: r={'sym':futs[f][0],'group':futs[f][7],'error':str(e)}
            out.append(r)
            print(f"[{i}/{len(JOBS)}] {r['sym']:<20} {r['group']:<18} "
                  f"n={r.get('trades')} R={r.get('total_R')} PF={r.get('PF')} {r.get('error','')}", flush=True)
            json.dump(out, open(dest,'w'), indent=1)
    print("DONE", flush=True)
