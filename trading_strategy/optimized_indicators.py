"""
Optimized Technical Indicators
Vectorized calculations for maximum performance
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from numba import jit


class OptimizedIndicators:
    """
    High-performance indicator calculations using NumPy vectorization and Numba JIT
    """
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_sma_numba(values: np.ndarray, period: int) -> np.ndarray:
        """Fast Simple Moving Average using Numba"""
        result = np.empty_like(values)
        result[:period-1] = np.nan
        
        for i in range(period-1, len(values)):
            result[i] = np.mean(values[i-period+1:i+1])
        
        return result
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'close', period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average (vectorized)"""
        return df[column].rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'close', period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average (vectorized)"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (vectorized)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Vectorized TR calculation
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Use EMA for ATR (faster than SMA)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_swing_points_numba(highs: np.ndarray, lows: np.ndarray, 
                                    strength: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast swing point detection using Numba
        Returns: (swing_highs, swing_lows) as boolean arrays
        """
        n = len(highs)
        swing_highs = np.zeros(n, dtype=np.bool_)
        swing_lows = np.zeros(n, dtype=np.bool_)
        
        for i in range(strength, n - strength):
            # Check for swing high
            is_swing_high = True
            for j in range(1, strength + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs[i] = True
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, strength + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows[i] = True
        
        return swing_highs, swing_lows
    
    @staticmethod
    def detect_swing_points(df: pd.DataFrame, strength: int = 2) -> pd.DataFrame:
        """
        Detect swing points using optimized Numba implementation
        """
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs, swing_lows = OptimizedIndicators.calculate_swing_points_numba(
            highs, lows, strength
        )
        
        result_df = df.copy()
        result_df['swing_high'] = swing_highs
        result_df['swing_low'] = swing_lows
        result_df['swing_high_price'] = np.where(swing_highs, highs, np.nan)
        result_df['swing_low_price'] = np.where(swing_lows, lows, np.nan)
        
        return result_df
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def detect_fvg_numba(highs: np.ndarray, lows: np.ndarray, 
                        min_gap_pct: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Fair Value Gap detection using Numba
        Returns: (fvg_indices, fvg_bullish) where fvg_bullish is boolean array
        """
        n = len(highs)
        fvg_indices = []
        fvg_bullish = []
        
        for i in range(2, n):
            # Bullish FVG: gap between candle[i-2].low and candle[i].high
            gap_up = lows[i-2] - highs[i]
            if gap_up > 0:
                gap_pct = gap_up / highs[i]
                if gap_pct >= min_gap_pct:
                    fvg_indices.append(i)
                    fvg_bullish.append(True)
            
            # Bearish FVG: gap between candle[i-2].high and candle[i].low
            gap_down = lows[i] - highs[i-2]
            if gap_down > 0:
                gap_pct = gap_down / lows[i]
                if gap_pct >= min_gap_pct:
                    fvg_indices.append(i)
                    fvg_bullish.append(False)
        
        return np.array(fvg_indices), np.array(fvg_bullish)
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """Calculate RSI (vectorized)"""
        delta = df[column].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'close', 
                                 period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands (vectorized)"""
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, column: str = 'close',
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (vectorized)"""
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (vectorized)"""
        direction = np.sign(df['close'].diff())
        obv = (direction * df['volume']).cumsum()
        return obv
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_pivots_numba(highs: np.ndarray, lows: np.ndarray, 
                              closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate pivot points (vectorized with Numba)
        Returns: (pivot, resistance, support)
        """
        n = len(highs)
        pivots = np.empty(n)
        resistances = np.empty(n)
        supports = np.empty(n)
        
        for i in range(1, n):
            pivot = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
            pivots[i] = pivot
            resistances[i] = 2 * pivot - lows[i-1]
            supports[i] = 2 * pivot - highs[i-1]
        
        # First value is NaN
        pivots[0] = np.nan
        resistances[0] = np.nan
        supports[0] = np.nan
        
        return pivots, resistances, supports
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, price_bins: int = 50) -> pd.DataFrame:
        """Calculate Volume Profile (optimized)"""
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, price_bins + 1)
        
        # Assign each candle to a bin based on close price
        df['price_bin'] = pd.cut(df['close'], bins=bins, labels=False, include_lowest=True)
        
        # Group by bin and sum volume
        volume_profile = df.groupby('price_bin')['volume'].sum().reset_index()
        volume_profile['price_level'] = bins[:-1] + (bins[1] - bins[0]) / 2
        
        return volume_profile
    
    @staticmethod
    def batch_calculate_indicators(df: pd.DataFrame, 
                                  indicators: list = None) -> pd.DataFrame:
        """
        Calculate multiple indicators in batch for efficiency
        
        Args:
            df: OHLCV DataFrame
            indicators: List of indicator names to calculate
            
        Returns:
            DataFrame with all indicators added
        """
        if indicators is None:
            indicators = ['sma_20', 'ema_50', 'atr_14', 'rsi_14']
        
        result_df = df.copy()
        
        for indicator in indicators:
            if indicator.startswith('sma_'):
                period = int(indicator.split('_')[1])
                result_df[indicator] = OptimizedIndicators.calculate_sma(df, period=period)
            
            elif indicator.startswith('ema_'):
                period = int(indicator.split('_')[1])
                result_df[indicator] = OptimizedIndicators.calculate_ema(df, period=period)
            
            elif indicator.startswith('atr_'):
                period = int(indicator.split('_')[1])
                result_df[indicator] = OptimizedIndicators.calculate_atr(df, period=period)
            
            elif indicator.startswith('rsi_'):
                period = int(indicator.split('_')[1])
                result_df[indicator] = OptimizedIndicators.calculate_rsi(df, period=period)
            
            elif indicator == 'obv':
                result_df['obv'] = OptimizedIndicators.calculate_obv(df)
            
            elif indicator == 'swing_points':
                result_df = OptimizedIndicators.detect_swing_points(result_df)
        
        return result_df


class IndicatorCache:
    """
    Cache for indicator calculations to avoid redundant computation
    """
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_key(self, df_hash: str, indicator: str, **params) -> str:
        """Generate cache key"""
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{df_hash}_{indicator}_{param_str}"
    
    def get(self, key: str) -> Optional[pd.Series]:
        """Get indicator from cache"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key].copy()
        return None
    
    def set(self, key: str, value: pd.Series):
        """Store indicator in cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value.copy()
        self.access_count[key] = 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()


# Global indicator cache
_indicator_cache = IndicatorCache()


def get_cached_indicator(df: pd.DataFrame, indicator_func, **kwargs) -> pd.Series:
    """
    Get indicator with caching
    
    Args:
        df: DataFrame
        indicator_func: Function to calculate indicator
        **kwargs: Parameters for indicator function
        
    Returns:
        Calculated or cached indicator series
    """
    # Generate hash of dataframe
    df_hash = pd.util.hash_pandas_object(df.index).sum()
    
    # Generate cache key
    func_name = indicator_func.__name__
    cache_key = _indicator_cache.get_key(str(df_hash), func_name, **kwargs)
    
    # Try to get from cache
    cached_result = _indicator_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Calculate and cache
    result = indicator_func(df, **kwargs)
    _indicator_cache.set(cache_key, result)
    
    return result
