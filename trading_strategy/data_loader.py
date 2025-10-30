import os
import pandas as pd
from typing import List, Dict, Optional
from functools import lru_cache
from pathlib import Path
import hashlib
import pickle

class DataLoader:
    """Optimized data loader with caching"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path) if isinstance(base_path, str) else base_path
        self.timeframes = ['5m','15m','30m','1h','4h','1d']
        
        # Memory cache for frequently accessed data
        self._memory_cache: Dict[str, pd.DataFrame] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Disk cache directory
        self._cache_dir = Path('.cache')
        self._cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, pair: str, tf: str, start_date: str, end_date: str) -> str:
        """Generate cache key for data request"""
        key_str = f"{pair}_{tf}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        # Check memory cache first
        if cache_key in self._memory_cache:
            self._cache_hits += 1
            return self._memory_cache[cache_key].copy()
        
        # Check disk cache
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                # Add to memory cache
                self._memory_cache[cache_key] = df
                self._cache_hits += 1
                return df.copy()
            except Exception:
                pass
        
        self._cache_misses += 1
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame):
        """Save data to cache"""
        # Save to memory cache (limit size to prevent memory issues)
        if len(self._memory_cache) < 20:  # Max 20 cached datasets
            self._memory_cache[cache_key] = df.copy()
        
        # Save to disk cache
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        try:
            df.to_pickle(cache_file)
        except Exception:
            pass

    def load_pair_data(self, pair: str,
                       timeframes: List[str] = None,
                       start_date: str = None,
                       end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load pair data with caching optimization
        """
        if timeframes is None:
            timeframes = self.timeframes

        pair_data = {}
        
        for tf in timeframes:
            # Try to load from cache
            cache_key = self._get_cache_key(pair, tf, start_date, end_date)
            cached_df = self._load_from_cache(cache_key)
            
            if cached_df is not None:
                pair_data[tf] = cached_df
                continue
            
            # Load from disk
            dfs = []
            ym_months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m")
            
            for ym in ym_months:
                # Try multiple path formats
                paths = [
                    self.base_path / "data" / "raw" / pair / tf / f"{pair}_{tf}_{ym}.parquet",
                    self.base_path / pair / tf / f"{pair}_{tf}_{ym}.parquet",
                    Path("data/raw") / pair / tf / f"{pair}_{tf}_{ym}.parquet"
                ]
                
                for path in paths:
                    if path.exists():
                        try:
                            dfm = pd.read_parquet(path)
                            dfs.append(dfm)
                            break
                        except Exception as e:
                            continue
            
            if dfs:
                # Optimize concatenation
                df = pd.concat(dfs, ignore_index=False, copy=False)
                
                # Optimize sorting (check if already sorted)
                if 'start_time' in df.columns:
                    if not df['start_time'].is_monotonic_increasing:
                        df.sort_values('start_time', inplace=True)
                    df.set_index('start_time', inplace=True)
                
                # Filter by date range
                df = df[start_date:end_date]
                
                # Optimize dtypes to reduce memory
                df = self._optimize_dtypes(df)
                
                # Save to cache
                self._save_to_cache(cache_key, df)
                
                pair_data[tf] = df
            else:
                cols = ['open','high','low','close','volume','turnover']
                pair_data[tf] = pd.DataFrame(columns=cols)
        
        return pair_data
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory"""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    def clear_cache(self):
        """Clear all caches"""
        self._memory_cache.clear()
        for cache_file in self._cache_dir.glob('*.pkl'):
            cache_file.unlink()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self._memory_cache)
        }
