"""
Data Loading Utilities
======================

Handles loading and resampling of futures data from CSV files.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Optional, Literal


def _normalize_freq(tf: str) -> str:
    """Normalize legacy pandas offset aliases for pandas >= 2.2.

    Keeps the project's documented timeframe API ('1T', '5T', '15T', '1H', '4H')
    working after pandas removed the 'T' (minute) and 'H' (hour) aliases in
    favour of 'min' and 'h'.
    """
    tf = re.sub(r'(\d*)T$', r'\1min', tf)
    tf = re.sub(r'(\d*)H$', r'\1h', tf)
    return tf


class DataLoader:
    """
    Load and resample futures data from CSV files.
    
    Supports NQ, ES and YM futures, six spot FX pairs (EURUSD, GBPUSD, USDJPY,
    AUDUSD, USDCHF, USDCAD) and three index CFDs, with flexible timeframe
    resampling.
    """
    
    FILENAMES = {
        'NQ': 'nq_futures_1m_cleaned.csv',
        'ES': 'es_futures_1m_cleaned.csv',
        'YM': 'ym_futures_1m_cleaned.csv',
        'EURUSD': 'eurusd_1m_cleaned.csv',
        'GBPUSD': 'gbpusd_1m_cleaned.csv',
        'USDJPY': 'usdjpy_1m_cleaned.csv',
        'AUDUSD': 'audusd_1m_cleaned.csv',
        'USDCHF': 'usdchf_1m_cleaned.csv',
        'USDCAD': 'usdcad_1m_cleaned.csv',
        # Dukascopy index CFDs - cash-index proxies for the NQ/ES/YM futures
        # the 2022 model was originally built on.
        'USATECHIDXUSD': 'usatechidxusd_1m_cleaned.csv',   # Nasdaq 100 (NQ)
        'USA500IDXUSD':  'usa500idxusd_1m_cleaned.csv',    # S&P 500 (ES)
        'USA30IDXUSD':   'usa30idxusd_1m_cleaned.csv',     # Dow 30 (YM)
        # 2015-2023 history for the same three indices, kept as separate keys so
        # the out-of-sample period can be loaded without disturbing 2024-2026.
        'USATECHIDXUSD_2015': 'usatechidxusd_2015_1m_cleaned.csv',
        'USA500IDXUSD_2015':  'usa500idxusd_2015_1m_cleaned.csv',
        'USA30IDXUSD_2015':   'usa30idxusd_2015_1m_cleaned.csv',
        # The two ranges stitched into one continuous series. The files abut
        # without overlap (2015 ends 2023-12-29 21:13Z, 2024 starts
        # 2024-01-01 23:00Z), so the concatenation is a join, not a merge.
        'USATECHIDXUSD_ALL': ['usatechidxusd_2015_1m_cleaned.csv',
                              'usatechidxusd_1m_cleaned.csv'],
        'USA500IDXUSD_ALL':  ['usa500idxusd_2015_1m_cleaned.csv',
                              'usa500idxusd_1m_cleaned.csv'],
        'USA30IDXUSD_ALL':   ['usa30idxusd_2015_1m_cleaned.csv',
                              'usa30idxusd_1m_cleaned.csv'],
        # Dukascopy's dollar-index CFD. Verified against the ICE basket
        # reconstructed from the five component pairs held here: 5m return
        # correlation 0.960, and -0.882 against EURUSD. This is the inverse leg
        # of the original 2016 USDX SMT teaching.
        'DOLLARIDXUSD': 'dollaridxusd_1m_cleaned.csv',
        'DXY_DAILY': 'dxy_daily.csv',
        'DXY_1H': 'dxy_1h.csv',
    }

    # Smallest quoted increment, used to scale price-based model params across
    # instruments. Index futures are quoted in points; FX in pips (JPY-quoted
    # pairs use 0.01, everything else 0.0001).
    PIP = {
        'NQ': 1.0, 'ES': 1.0, 'YM': 1.0,
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
        'AUDUSD': 0.0001, 'USDCHF': 0.0001, 'USDCAD': 0.0001,
        'USATECHIDXUSD': 1.0, 'USA500IDXUSD': 1.0, 'USA30IDXUSD': 1.0,
        'USATECHIDXUSD_2015': 1.0, 'USA500IDXUSD_2015': 1.0, 'USA30IDXUSD_2015': 1.0,
        'USATECHIDXUSD_ALL': 1.0, 'USA500IDXUSD_ALL': 1.0, 'USA30IDXUSD_ALL': 1.0,
        'DOLLARIDXUSD': 0.01,
        'USDCHF': 0.0001, 'USDCAD': 0.0001, 'AUDUSD': 0.0001,
    }

    FX_SYMBOLS = ('EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD')
    INDEX_SYMBOLS = ('USATECHIDXUSD', 'USA500IDXUSD', 'USA30IDXUSD')
    INDEX_SYMBOLS_ALL = ('USATECHIDXUSD_ALL', 'USA500IDXUSD_ALL', 'USA30IDXUSD_ALL')
    FUTURES_SYMBOLS = ('NQ', 'ES', 'YM')

    # SMT correlation groups. Members must be positively correlated with each
    # other; `inverse` members move against the group and are compared with the
    # sign flipped. Measured 5m correlations on this dataset are in the values.
    SMT_GROUPS = {
        # NQ/ES +0.92, ES/YM +0.95, NQ/YM +0.83 (2015-2023)
        'indices': {'members': ('USATECHIDXUSD_ALL', 'USA500IDXUSD_ALL',
                                'USA30IDXUSD_ALL'), 'inverse': ()},
        'indices_2024': {'members': ('USATECHIDXUSD', 'USA500IDXUSD',
                                     'USA30IDXUSD'), 'inverse': ()},
        # EURUSD/GBPUSD +0.77; DXY -0.88 vs EURUSD, -0.78 vs GBPUSD
        'fx_dollar': {'members': ('EURUSD', 'GBPUSD', 'DOLLARIDXUSD'),
                      'inverse': ('DOLLARIDXUSD',)},
    }

    def __init__(self, timeframe: str, weeks: Optional[int] = None, data_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            timeframe: Pandas resample string ('5T', '15T', '1H', 'D', etc.)
            weeks: Number of weeks to load (None = all data)
            data_dir: Path to data directory (defaults to standard location)
        """
        self.timeframe = timeframe
        self.weeks = weeks
        
        if data_dir is None:
            self.data_dir = Path.home() / 'Projects' / 'Data'
        else:
            self.data_dir = Path(data_dir)
    
    def _load_and_resample(self, symbol: str) -> pd.DataFrame:
        """
        Internal method to load and resample data.
        
        Args:
            symbol: Instrument key from FILENAMES (e.g. 'NQ', 'EURUSD')
        
        Returns:
            Resampled OHLCV DataFrame with datetime index
        """
        entry = self.FILENAMES[symbol]
        names = [entry] if isinstance(entry, str) else list(entry)

        frames = []
        for name in names:
            filepath = self.data_dir / name
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            part = pd.read_csv(filepath)
            part['timestamp'] = pd.to_datetime(part['timestamp'])
            frames.append(part.set_index('timestamp'))

        df = pd.concat(frames).sort_index() if len(frames) > 1 else frames[0].sort_index()
        # Multi-file symbols stitch two date ranges that abut rather than
        # overlap; dedupe defensively so a re-export with one shared bar cannot
        # silently double a candle.
        if len(frames) > 1:
            df = df[~df.index.duplicated(keep='first')]

        # Filter by weeks if specified
        if self.weeks is not None:
            cutoff_date = df.index.max() - pd.Timedelta(weeks=self.weeks)
            df = df[df.index >= cutoff_date]
        
        # Resample to specified timeframe.
        # `spread` is the ask-bid distance recorded per source bar; it is carried
        # through as a mean so the backtest can charge a real, per-bar execution
        # cost instead of a fabricated constant. Averaging (not summing) is what
        # keeps it a price distance rather than an accumulation.
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        if 'spread' in df.columns:
            agg['spread'] = 'mean'

        resampled = df.resample(_normalize_freq(self.timeframe)).agg(agg)
        return resampled
    
    def read_NQ(self) -> pd.DataFrame:
        """Load Nasdaq 100 futures (NQ) data."""
        return self._load_and_resample('NQ')
    
    def read_ES(self) -> pd.DataFrame:
        """Load S&P 500 futures (ES) data."""
        return self._load_and_resample('ES')
    
    def read_YM(self) -> pd.DataFrame:
        """Load Dow Jones futures (YM) data."""
        return self._load_and_resample('YM')
    
    def read(self, symbol: str) -> pd.DataFrame:
        """Load any instrument by key, e.g. read('EURUSD'). Case-insensitive."""
        key = symbol.upper()
        if key not in self.FILENAMES:
            raise KeyError(
                f"Unknown symbol {symbol!r}. Known: {sorted(self.FILENAMES)}"
            )
        return self._load_and_resample(key)

    def read_EURUSD(self) -> pd.DataFrame:
        """Load EUR/USD spot FX data (mid price)."""
        return self._load_and_resample('EURUSD')

    def read_GBPUSD(self) -> pd.DataFrame:
        """Load GBP/USD spot FX data (mid price)."""
        return self._load_and_resample('GBPUSD')

    def read_USDJPY(self) -> pd.DataFrame:
        """Load USD/JPY spot FX data (mid price)."""
        return self._load_and_resample('USDJPY')

    def read_AUDUSD(self) -> pd.DataFrame:
        """Load AUD/USD spot FX data (mid price)."""
        return self._load_and_resample('AUDUSD')

    def read_USDCHF(self) -> pd.DataFrame:
        """Load USD/CHF spot FX data (mid price)."""
        return self._load_and_resample('USDCHF')

    def read_USDCAD(self) -> pd.DataFrame:
        """Load USD/CAD spot FX data (mid price)."""
        return self._load_and_resample('USDCAD')

    def read_USATECHIDXUSD(self) -> pd.DataFrame:
        """Load the Nasdaq-100 index CFD (cash proxy for NQ)."""
        return self._load_and_resample('USATECHIDXUSD')

    def read_USA500IDXUSD(self) -> pd.DataFrame:
        """Load the S&P-500 index CFD (cash proxy for ES)."""
        return self._load_and_resample('USA500IDXUSD')

    def read_USA30IDXUSD(self) -> pd.DataFrame:
        """Load the Dow-30 index CFD (cash proxy for YM)."""
        return self._load_and_resample('USA30IDXUSD')

    def read_fx(self) -> dict:
        """Load all three FX pairs.

        Returns:
            Dict of symbol -> DataFrame for EURUSD, GBPUSD and USDJPY.

        Note: EURUSD and GBPUSD are positively correlated, so they can be fed
        straight to detect_smt. USDJPY is USD-base and therefore *inversely*
        correlated to the other two - invert it (or use it as a dollar proxy)
        before treating it as a positively correlated leg.
        """
        return {s: self._load_and_resample(s) for s in self.FX_SYMBOLS}

    def read_smt_group(self, group: str) -> dict:
        """Load one SMT correlation group, aligned to a common bar index.

        SMT compares the *same time-of-day candle* across instruments (the
        KB's `correlated-pairs-smt` criterion c2 and its most-cited failure
        mode). That is only sound on a shared index, so this drops any bar not
        present in every member rather than forward-filling one - a fabricated
        bar would manufacture divergences that never traded.

        Returns:
            {'frames': {symbol: DataFrame}, 'inverse': (symbols,)}
        """
        if group not in self.SMT_GROUPS:
            raise KeyError(f"Unknown SMT group {group!r}. "
                           f"Known: {sorted(self.SMT_GROUPS)}")
        spec = self.SMT_GROUPS[group]
        frames = {s: self._load_and_resample(s).dropna(subset=['open', 'high',
                                                               'low', 'close'])
                  for s in spec['members']}

        common = None
        for df in frames.values():
            common = df.index if common is None else common.intersection(df.index)
        frames = {s: df.loc[common] for s, df in frames.items()}
        return {'frames': frames, 'inverse': spec['inverse']}

    def pip(self, symbol: str) -> float:
        """Smallest quoted increment for `symbol` - use to scale point-based
        model params (eq_tolerance_pts, max_stop_pts, ...) across instruments."""
        key = symbol.upper()
        if key not in self.PIP:
            raise KeyError(f"No pip size known for {symbol!r}. Known: {sorted(self.PIP)}")
        return self.PIP[key]

    def read_DXY(self, resolution: str = 'daily') -> pd.DataFrame:
        """Load DXY (US Dollar Index) data.

        Args:
            resolution: 'daily' (2010–present) or '1h' (last ~730 days).
        """
        key = 'DXY_DAILY' if resolution == 'daily' else 'DXY_1H'
        filepath = self.data_dir / {'DXY_DAILY': 'dxy_daily.csv', 'DXY_1H': 'dxy_1h.csv'}[key]
        if not filepath.exists():
            raise FileNotFoundError(
                f"DXY data not found at {filepath}. "
                "Run infrastructure/fetch_dxy.py to download."
            )
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        df = df.sort_index()
        if self.weeks is not None:
            cutoff = df.index.max() - pd.Timedelta(weeks=self.weeks)
            df = df[df.index >= cutoff]
        return df

    def read_all(self) -> dict:
        """
        Load all available futures data.
        
        Returns:
            Dict with keys 'NQ', 'ES', 'YM' containing respective DataFrames
        """
        return {
            'NQ': self.read_NQ(),
            'ES': self.read_ES(),
            'YM': self.read_YM()
        }


# Maintain backward compatibility with old class name
class csvReader(DataLoader):
    """Deprecated: Use DataLoader instead. Maintained for backward compatibility."""
    pass
