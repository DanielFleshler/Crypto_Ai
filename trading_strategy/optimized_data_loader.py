"""
Optimized Data Loader with Caching and Performance Enhancements
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta
import concurrent.futures
from functools import lru_cache
import pyarrow.parquet as pq
import pyarrow as pa


class OptimizedDataLoader:
    """
    High-performance data loader with:
    - Multi-level caching (memory + disk)
    - Parallel loading
    - Memory-mapped files
    - Incremental updates
    - Data prefetching
    """
    
    def __init__(self, base_path: str, cache_size_mb: int = 1000):
        self.base_path = Path(base_path)
        self.cache_path = self.base_path / '.cache'
        self.cache_path.mkdir(exist_ok=True)
        
        # Memory cache with size limit
        self.cache_size_mb = cache_size_mb
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        self.cache_sizes: Dict[str, int] = {}
        self.total_cache_size = 0
        
        # Precomputed file index for fast lookups
        self.file_index = self._build_file_index()
        
        # Thread pool for parallel loading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def _build_file_index(self) -> Dict[str, Dict[str, List[Path]]]:
        """Build index of available data files for fast lookup."""
        index = {}
        data_path = self.base_path / 'data' / 'raw'
        
        if not data_path.exists():
            return index
            
        for pair_dir in data_path.iterdir():
            if not pair_dir.is_dir():
                continue
                
            pair = pair_dir.name
            index[pair] = {}
            
            for tf_dir in pair_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                    
                timeframe = tf_dir.name
                parquet_files = sorted(tf_dir.glob('*.parquet'))
                index[pair][timeframe] = parquet_files
                
        return index
    
    def _get_cache_key(self, pair: str, timeframe: str, 
                      start_date: str, end_date: str) -> str:
        """Generate cache key for data request."""
        key_str = f"{pair}_{timeframe}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _manage_cache_size(self):
        """Evict oldest cache entries if size limit exceeded."""
        if self.total_cache_size <= self.cache_size_mb * 1024 * 1024:
            return
            
        # Sort by access time (implement LRU)
        sorted_keys = sorted(self.memory_cache.keys(), 
                           key=lambda k: self.cache_access_times.get(k, 0))
        
        while self.total_cache_size > self.cache_size_mb * 1024 * 1024 and sorted_keys:
            key = sorted_keys.pop(0)
            size = self.cache_sizes.get(key, 0)
            del self.memory_cache[key]
            del self.cache_sizes[key]
            self.total_cache_size -= size
    
    @lru_cache(maxsize=128)
    def _read_parquet_optimized(self, file_path: Path, 
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Read parquet file with optimizations."""
        # Use pyarrow for better performance
        table = pq.read_table(
            file_path,
            columns=columns,
            use_pandas_metadata=True
        )
        
        # Convert to pandas with optimizations
        df = table.to_pandas(
            date_as_object=False,
            use_threads=True,
            ignore_metadata=False
        )
        
        return df
    
    def load_pair_data_parallel(self, pair: str, timeframes: List[str],
                              start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes in parallel."""
        # Check if pair exists in index
        if pair not in self.file_index:
            return {tf: pd.DataFrame() for tf in timeframes}
        
        # Create loading tasks
        futures = {}
        for tf in timeframes:
            future = self.executor.submit(
                self._load_single_timeframe,
                pair, tf, start_date, end_date
            )
            futures[tf] = future
        
        # Collect results
        results = {}
        for tf, future in futures.items():
            try:
                results[tf] = future.result(timeout=30)
            except Exception as e:
                print(f"Error loading {pair} {tf}: {e}")
                results[tf] = pd.DataFrame()
                
        return results
    
    def _load_single_timeframe(self, pair: str, timeframe: str,
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Load data for a single timeframe with caching."""
        # Check memory cache first
        cache_key = self._get_cache_key(pair, timeframe, start_date, end_date)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_path / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                # Add to memory cache
                self._add_to_memory_cache(cache_key, df)
                return df
            except Exception:
                pass
        
        # Load from source files
        df = self._load_from_source(pair, timeframe, start_date, end_date)
        
        # Cache the result
        if not df.empty:
            self._add_to_memory_cache(cache_key, df)
            # Save to disk cache
            df.to_pickle(cache_file)
            
        return df
    
    def _load_from_source(self, pair: str, timeframe: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from source parquet files."""
        if pair not in self.file_index or timeframe not in self.file_index[pair]:
            return pd.DataFrame()
        
        # Get relevant files based on date range
        files = self._get_relevant_files(pair, timeframe, start_date, end_date)
        
        if not files:
            return pd.DataFrame()
        
        # Load files in parallel
        dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._read_parquet_optimized, f) for f in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    df = future.result()
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading file: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        # Efficient concatenation
        df = pd.concat(dfs, ignore_index=False, sort=False)
        
        # Optimize memory usage
        df = self._optimize_dtypes(df)
        
        # Sort and filter by date range
        if 'start_time' in df.columns:
            df.sort_values('start_time', inplace=True)
            df.set_index('start_time', inplace=True)
        
        # Filter to exact date range
        df = df[start_date:end_date]
        
        return df
    
    def _get_relevant_files(self, pair: str, timeframe: str,
                          start_date: str, end_date: str) -> List[Path]:
        """Get list of files that contain data for the requested period."""
        all_files = self.file_index[pair][timeframe]
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        relevant_files = []
        for file in all_files:
            # Extract date from filename (assuming format: PAIR_TF_YYYY-MM.parquet)
            try:
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    year_month = parts[2]
                    file_date = pd.to_datetime(year_month + '-01')
                    file_end = file_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                    
                    # Check if file period overlaps with requested period
                    if file_date <= end and file_end >= start:
                        relevant_files.append(file)
            except Exception:
                continue
                
        return relevant_files
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        # Optimize numeric columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
                
        return df
    
    def _add_to_memory_cache(self, key: str, df: pd.DataFrame):
        """Add DataFrame to memory cache with size management."""
        # Calculate size
        size = df.memory_usage(deep=True).sum()
        
        # Add to cache
        self.memory_cache[key] = df
        self.cache_sizes[key] = size
        self.total_cache_size += size
        
        # Manage cache size
        self._manage_cache_size()
    
    def preload_data(self, pairs: List[str], timeframes: List[str],
                    start_date: str, end_date: str):
        """Preload data for multiple pairs/timeframes in background."""
        futures = []
        
        for pair in pairs:
            future = self.executor.submit(
                self.load_pair_data_parallel,
                pair, timeframes, start_date, end_date
            )
            futures.append(future)
            
        # Don't wait for completion - let it load in background
        return futures
    
    def get_incremental_update(self, pair: str, timeframe: str,
                             last_timestamp: datetime) -> pd.DataFrame:
        """Get only new data since last timestamp."""
        # This would connect to a real-time data feed in production
        # For now, we'll simulate by loading recent data
        end_date = datetime.now()
        start_date = last_timestamp
        
        return self._load_single_timeframe(
            pair, timeframe, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    def clear_cache(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.cache_sizes.clear()
        self.total_cache_size = 0
        
        # Clear disk cache
        for cache_file in self.cache_path.glob('*.pkl'):
            cache_file.unlink()
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
