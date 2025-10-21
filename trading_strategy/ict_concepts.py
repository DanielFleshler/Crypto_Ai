# Phase 1.3: ICT Concepts Detection (FVG, OB, IFVG, BB, OTE)
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ICTConcept:
    timestamp: datetime
    concept_type: str
    start_price: float
    end_price: float
    timeframe: str
    strength: float
class ICTConceptsDetector:
    """
    Detects ICT (Inner Circle Trader) concepts:
    - FVG (Fair Value Gap)
    - IFVG (Institutional Fair Value Gap) 
    - OB (Order Block)
    - BB (Breaker Block)
    - OTE (Optimal Trade Entry)
    """
    
    def __init__(self):
        self.concepts = []
        
    def detect_fvg(self, df: pd.DataFrame, min_gap_percent: float = 0.1) -> List[ICTConcept]:
        """
        Detect Fair Value Gaps (FVG)
        FVG occurs when there's a gap between candle 1 high/low and candle 3 low/high
        
        Args:
            df: OHLC DataFrame
            min_gap_percent: Minimum gap size as percentage of price
            
        Returns:
            List of ICTConcept objects for FVGs
        """
        fvgs = []
        
        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1] 
            candle_3 = df.iloc[i]
            
            # Bullish FVG: Candle 1 high < Candle 3 low
            if candle_1['high'] < candle_3['low']:
                gap_size = candle_3['low'] - candle_1['high']
                gap_percent = (gap_size / candle_1['high']) * 100
                
                if gap_percent >= min_gap_percent:
                    fvgs.append(ICTConcept(
                        timestamp=df.index[i],
                        concept_type='FVG_BULLISH',
                        start_price=candle_1['high'],
                        end_price=candle_3['low'],
                        timeframe='current',
                        strength=min(gap_percent / 2, 1.0)  # Cap at 1.0
                    ))
            
            # Bearish FVG: Candle 1 low > Candle 3 high  
            elif candle_1['low'] > candle_3['high']:
                gap_size = candle_1['low'] - candle_3['high']
                gap_percent = (gap_size / candle_1['low']) * 100
                
                if gap_percent >= min_gap_percent:
                    fvgs.append(ICTConcept(
                        timestamp=df.index[i],
                        concept_type='FVG_BEARISH',
                        start_price=candle_3['high'],
                        end_price=candle_1['low'],
                        timeframe='current',
                        strength=min(gap_percent / 2, 1.0)
                    ))
        
        return fvgs
    
    def detect_order_blocks(self, df: pd.DataFrame, swing_points: pd.DataFrame) -> List[ICTConcept]:
        """
        Detect Order Blocks (OB)
        OB is the last candle before a strong move in opposite direction
        
        Args:
            df: OHLC DataFrame  
            swing_points: DataFrame with swing highs/lows marked
            
        Returns:
            List of ICTConcept objects for Order Blocks
        """
        order_blocks = []
        
        # Get swing highs and lows
        swing_highs = swing_points[swing_points['swing_high'] == True]
        swing_lows = swing_points[swing_points['swing_low'] == True]
        
        # Find bullish order blocks (before swing lows)
        for swing_idx in swing_lows.index:
            swing_low_price = swing_lows.loc[swing_idx, 'swing_low_price']
            
            # Look for the last bearish candle before the swing low
            for i in range(df.index.get_loc(swing_idx) - 1, max(0, df.index.get_loc(swing_idx) - 10), -1):
                candle = df.iloc[i]
                
                # Check if it's a bearish candle that created the low
                if candle['close'] < candle['open'] and candle['low'] <= swing_low_price * 1.01:
                    order_blocks.append(ICTConcept(
                        timestamp=df.index[i],
                        concept_type='OB_BULLISH',
                        start_price=candle['low'],
                        end_price=candle['high'], 
                        timeframe='current',
                        strength=0.7
                    ))
                    break
        
        # Find bearish order blocks (before swing highs)
        for swing_idx in swing_highs.index:
            swing_high_price = swing_highs.loc[swing_idx, 'swing_high_price']
            
            # Look for the last bullish candle before the swing high
            for i in range(df.index.get_loc(swing_idx) - 1, max(0, df.index.get_loc(swing_idx) - 10), -1):
                candle = df.iloc[i]
                
                # Check if it's a bullish candle that created the high
                if candle['close'] > candle['open'] and candle['high'] >= swing_high_price * 0.99:
                    order_blocks.append(ICTConcept(
                        timestamp=df.index[i],
                        concept_type='OB_BEARISH',
                        start_price=candle['low'],
                        end_price=candle['high'],
                        timeframe='current', 
                        strength=0.7
                    ))
                    break
        
        return order_blocks
    
    def detect_breaker_blocks(self, df: pd.DataFrame, order_blocks: List[ICTConcept]) -> List[ICTConcept]:
        """
        Detect Breaker Blocks (BB)
        BB is an Order Block that has been broken and now acts as support/resistance
        
        Args:
            df: OHLC DataFrame
            order_blocks: List of previously identified Order Blocks
            
        Returns:
            List of ICTConcept objects for Breaker Blocks
        """
        breaker_blocks = []
        
        for ob in order_blocks:
            ob_timestamp = ob.timestamp
            ob_start_idx = df.index.get_loc(ob_timestamp)
            
            # Check if OB was broken in subsequent candles
            for i in range(ob_start_idx + 1, len(df)):
                candle = df.iloc[i]
                
                if ob.concept_type == 'OB_BULLISH':
                    # Check if price broke below the bullish OB
                    if candle['low'] < ob.start_price:
                        breaker_blocks.append(ICTConcept(
                            timestamp=ob.timestamp,
                            concept_type='BB_BEARISH',
                            start_price=ob.start_price,
                            end_price=ob.end_price,
                            timeframe='current',
                            strength=0.8
                        ))
                        break
                        
                elif ob.concept_type == 'OB_BEARISH':
                    # Check if price broke above the bearish OB
                    if candle['high'] > ob.end_price:
                        breaker_blocks.append(ICTConcept(
                            timestamp=ob.timestamp,
                            concept_type='BB_BULLISH',
                            start_price=ob.start_price,
                            end_price=ob.end_price,
                            timeframe='current',
                            strength=0.8
                        ))
                        break
        
        return breaker_blocks
    
    def calculate_ote_levels(self, start_price: float, end_price: float) -> Dict[str, float]:
        """
        Calculate OTE (Optimal Trade Entry) levels using Fibonacci
        OTE zone is typically between 62% - 79% retracement
        
        Args:
            start_price: Start of the move
            end_price: End of the move
            
        Returns:
            Dict with Fibonacci levels
        """
        price_diff = end_price - start_price
        
        return {
            'ote_start': start_price + (price_diff * 0.62),  # 62% 
            'ote_end': start_price + (price_diff * 0.79),    # 79%
            'fib_50': start_price + (price_diff * 0.50),     # 50%
            'fib_618': start_price + (price_diff * 0.618),   # 61.8%
            'fib_786': start_price + (price_diff * 0.786)    # 78.6%
        }
    
    def detect_ote_zones(self, df: pd.DataFrame, swing_points: pd.DataFrame, 
                        lookback: int = 20) -> List[ICTConcept]:
        """
        Detect OTE (Optimal Trade Entry) zones
        
        Args:
            df: OHLC DataFrame
            swing_points: DataFrame with swing points
            lookback: Number of periods to look back for swings
            
        Returns:
            List of ICTConcept objects for OTE zones
        """
        ote_zones = []
        
        # Get recent swing points
        recent_swings = []
        swing_highs = swing_points[swing_points['swing_high'] == True].tail(lookback)
        swing_lows = swing_points[swing_points['swing_low'] == True].tail(lookback)
        
        # Create OTE zones for recent significant moves
        for idx, high_row in swing_highs.iterrows():
            for idx2, low_row in swing_lows.iterrows():
                # Check if swings are within reasonable distance (10 periods)
                # Convert to numeric positions for comparison
                idx_pos = swing_highs.index.get_loc(idx)
                idx2_pos = swing_lows.index.get_loc(idx2)
                if abs(idx_pos - idx2_pos) <= 10:  # Swings within 10 periods
                    start_price = low_row['swing_low_price']
                    end_price = high_row['swing_high_price']
                    
                    if end_price > start_price:  # Valid bullish move
                        ote_levels = self.calculate_ote_levels(start_price, end_price)
                        
                        ote_zones.append(ICTConcept(
                            timestamp=max(idx, idx2),
                            concept_type='OTE_BULLISH',
                            start_price=ote_levels['ote_start'],
                            end_price=ote_levels['ote_end'],
                            timeframe='current',
                            strength=0.9
                        ))
        
        return ote_zones

print("🎯 ICTConceptsDetector Class Created")
print("✅ FVG (Fair Value Gap) detection")
print("✅ Order Block (OB) identification")  
print("✅ Breaker Block (BB) detection")
print("✅ OTE (Optimal Trade Entry) zones")
print("📊 Ready for ICT pattern analysis")