# Phase 1.2: Market Structure Detection (BOS/CHoCH)
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class MarketStructure:
    timestamp: datetime
    structure_type: str
    price: float
    timeframe: str
    strength: float
class MarketStructureDetector:
    """
    Detects market structure changes: BOS (Break of Structure) and CHoCH (Change of Character)
    Based on Higher Highs, Higher Lows, Lower Highs, Lower Lows
    """
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.structures = []
        
    def detect_swing_points(self, df: pd.DataFrame, strength: int = 2) -> pd.DataFrame:
        """
        Detect swing highs and lows
        
        Args:
            df: OHLC DataFrame
            strength: Number of candles on each side to confirm swing
            
        Returns:
            DataFrame with swing points marked
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        high_values = df['high'].values
        low_values = df['low'].values
        
        for i in range(strength, len(df) - strength):
            # Check for swing high
            is_swing_high = True
            for j in range(strength):
                if (high_values[i] <= high_values[i - j - 1] or 
                    high_values[i] <= high_values[i + j + 1]):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                df.iloc[i, df.columns.get_loc('swing_high')] = True
                df.iloc[i, df.columns.get_loc('swing_high_price')] = high_values[i]
            
            # Check for swing low  
            is_swing_low = True
            for j in range(strength):
                if (low_values[i] >= low_values[i - j - 1] or 
                    low_values[i] >= low_values[i + j + 1]):
                    is_swing_low = False
                    break
                    
            if is_swing_low:
                df.iloc[i, df.columns.get_loc('swing_low')] = True
                df.iloc[i, df.columns.get_loc('swing_low_price')] = low_values[i]
        
        return df
    
    def detect_market_structure(self, df: pd.DataFrame) -> List[MarketStructure]:
        """
        Detect BOS and CHoCH based on swing points
        
        Args:
            df: DataFrame with swing points
            
        Returns:
            List of MarketStructure objects
        """
        structures = []
        
        # Get swing points
        swing_highs = df[df['swing_high'] == True].copy()
        swing_lows = df[df['swing_low'] == True].copy()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return structures
        
        # Analyze trend changes
        prev_trend = None  # 'BULLISH' or 'BEARISH'
        
        # Combine and sort swing points by time
        all_swings = []
        
        for idx, row in swing_highs.iterrows():
            all_swings.append({
                'timestamp': idx,
                'type': 'HIGH',
                'price': row['swing_high_price']
            })
            
        for idx, row in swing_lows.iterrows():
            all_swings.append({
                'timestamp': idx,
                'type': 'LOW', 
                'price': row['swing_low_price']
            })
        
        # Sort by timestamp
        all_swings.sort(key=lambda x: x['timestamp'])
        
        if len(all_swings) >= 4:
            # Analyze pattern: need at least 4 points for structure analysis
            for i in range(3, len(all_swings)):
                current_swing = all_swings[i]
                prev_swing = all_swings[i-1]
                
                # Detect structure breaks
                if current_swing['type'] == 'HIGH' and prev_swing['type'] == 'LOW':
                    # Potential bullish structure
                    if current_swing['price'] > all_swings[i-2]['price']:  # Higher High
                        if prev_swing['price'] > all_swings[i-3]['price']:  # Higher Low
                            structure_type = 'BOS' if prev_trend == 'BULLISH' else 'CHoCH'
                            prev_trend = 'BULLISH'
                        else:
                            structure_type = 'BOS' if prev_trend == 'BEARISH' else 'CHoCH'
                            prev_trend = 'BEARISH'
                    else:
                        structure_type = 'BOS' if prev_trend == 'BEARISH' else 'CHoCH'
                        prev_trend = 'BEARISH'
                        
                elif current_swing['type'] == 'LOW' and prev_swing['type'] == 'HIGH':
                    # Potential bearish structure
                    if current_swing['price'] < all_swings[i-2]['price']:  # Lower Low
                        if prev_swing['price'] < all_swings[i-3]['price']:  # Lower High
                            structure_type = 'BOS' if prev_trend == 'BEARISH' else 'CHoCH'
                            prev_trend = 'BEARISH'
                        else:
                            structure_type = 'BOS' if prev_trend == 'BULLISH' else 'CHoCH'
                            prev_trend = 'BULLISH'
                    else:
                        structure_type = 'BOS' if prev_trend == 'BULLISH' else 'CHoCH'
                        prev_trend = 'BULLISH'
                else:
                    continue
                
                # Create MarketStructure object
                structures.append(MarketStructure(
                    timestamp=current_swing['timestamp'],
                    structure_type=structure_type,
                    price=current_swing['price'],
                    timeframe='current',  # Will be set by caller
                    strength=0.8  # Default strength, can be improved
                ))
        
        return structures
    
    def get_current_bias(self, structures: List[MarketStructure]) -> str:
        """
        Determine current market bias based on recent structures
        
        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if not structures:
            return 'NEUTRAL'
            
        recent_structures = structures[-3:]  # Look at last 3 structures
        
        bullish_count = sum(1 for s in recent_structures if 'BOS' in s.structure_type and s.price > 0)
        bearish_count = sum(1 for s in recent_structures if 'BOS' in s.structure_type and s.price < 0)
        
        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

# MarketStructureDetector class ready for use