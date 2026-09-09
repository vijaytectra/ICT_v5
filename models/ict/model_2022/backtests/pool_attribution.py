"""Drop the pools that are structurally mismatched, and see what the freed
sessions give back.

Post-mortem of the 148-trade index baseline: total +5.7R, of which the ONS
(overnight 05:00-09:15 NY) pool alone is +25.3R across all three instruments,
while Chicago pit (-12.1R), the 09:30 opening range (-8.3R) and PDH/PDL (-6.1R)
are pure loss.

The timing is mechanical, not coincidental. Each pool can only be swept after
its own formed_at, so each family occupies a fixed slot in the day:

    PDL/REH/REL/Chicago-prior   02:00-05:00   (live from session open)
    ONS                         from 09:16    (formed 09:15)
    OR                          from 10:00
    Chicago same-day            from 13:46    (formed 12:00)

_detect_sweep takes the EARLIEST valid sweep and the model takes one setup per
session, so an early pool consumes the session and the 09:16 ONS setup is never
evaluated. Removing the early pools should therefore not only drop their losses
but also RAISE the ONS trade count - the opposite of a filter shrinking the
sample. That is the specific prediction this run tests.
"""
import os, sys, json, itertools
from concurrent.futures import ProcessPoolExecutor

ROOT = r"C:\personal\ICT_v5"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "ict", "model_2022", "backtests"))

BASE = dict(
    start_date='2024-05-01', use_daily_bias=True,
    killzones=[('02:00','05:00'), ('08:30','11:00'), ('13:30','16:00')],
    fvg_entry_depth='edge', stop_mode='wick', target_min_rr=3.0,
    use_spread=True, commission_points=0.0, slippage_points=0.0,
)
INSTR = {
 'USATECHIDXUSD': dict(eq_tolerance_pts=17.0, max_stop_pts=95.0, stop_fixed_pts=95.0,
                       stop_buffer_pts=19.0, dollars_per_point=1.0, price_dp=2),
 'USA500IDXUSD':  dict(eq_tolerance_pts=3.0,  max_stop_pts=17.0, stop_fixed_pts=17.0,
                       stop_buffer_pts=3.3,  dollars_per_point=1.0, price_dp=2),
 'USA30IDXUSD':   dict(eq_tolerance_pts=19.0, max_stop_pts=110.0, stop_fixed_pts=110.0,
                       stop_buffer_pts=21.0, dollars_per_point=1.0, price_dp=2),
}

ALL_ON  = dict(use_pdh_pdl=True, use_reh_rel=True, use_ons=True,
               use_chicago_prior=True, use_chicago_same_day=True,
               use_opening_range=True, opening_range_start='09:30',
               opening_range_minutes=30, use_asian_range=False)
ONS_ONLY = dict(use_pdh_pdl=False, use_reh_rel=False, use_ons=True,
                use_chicago_prior=False, use_chicago_same_day=False,
                use_opening_range=False, use_asian_range=False)
ONS_REH  = dict(ONS_ONLY, use_reh_rel=True)

VARIANTS = {
 'C0 all pools (baseline)':  ALL_ON,
 'C1 ONS only':              ONS_ONLY,
 'C2 ONS + REH/REL':         ONS_REH,
 'C3 ONS only + Asian':      dict(ONS_ONLY, use_asian_range=True),
 'C4 ONS only, pool target': dict(ONS_ONLY, target_mode='pool_only', target_min_rr=1.0),
 'C5 ONS only, NY kz only':  dict(ONS_ONLY, killzones=[('08:30','11:00')]),
}

def _run(job):
    sym, var = job
    import pandas as pd
    from fvg_sweep_backtest import run_backtest, DATA_DIR, WARMUP_WEEKS
    from infrastructure import DataLoader
    start = pd.Timestamp(BASE['start_date'])
    weeks = int((pd.Timestamp.now() - (start - pd.Timedelta(weeks=WARMUP_WEEKS))).days/7)+4
    df = DataLoader(timeframe='1min', weeks=weeks, data_dir=DATA_DIR).read(sym)
    cfg = {**BASE, **INSTR[sym], **VARIANTS[var], 'instrument': sym, 'name': 'x'}
    try:
        st = run_backtest(run_config=cfg, verbose=False, save_outputs=False, df_1m_cache=df)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return {'sym': sym, 'variant': var, 'error': f"{type(exc).__name__}: {exc}"}
    return {'sym': sym, 'variant': var, 'trades': st.get('total_trades'),
            'prof%': st.get('profitable_pct'), 'tp%': st.get('win_rate'),
            'avg_R': st.get('avg_r'), 'total_R': st.get('total_r'),
            'PF': st.get('profit_factor'), 'net': st.get('total_pnl_usd'),
            'maxDD': st.get('max_dd_usd'), 'sharpe': st.get('sharpe_ratio')}

if __name__ == '__main__':
    jobs = list(itertools.product(INSTR, VARIANTS))
    print(f"{len(jobs)} runs", flush=True)
    out=[]
    with ProcessPoolExecutor(max_workers=9) as ex:
        for i, r in enumerate(ex.map(_run, jobs), 1):
            out.append(r)
            print(f"[{i}/{len(jobs)}] {r['sym']} {r['variant']} "
                  f"n={r.get('trades')} R={r.get('total_R')} PF={r.get('PF')} {r.get('error','')}", flush=True)
            json.dump(out, open(os.path.join(os.path.dirname(__file__),'ons.json'),'w'), indent=1)
    print("DONE", flush=True)
