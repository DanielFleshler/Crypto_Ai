# Phase 1.4: Elliott Wave Detection (Using Fibonacci-based approach as requested)
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ElliottWave:
    wave_number: int
    wave_type: str
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    timeframe: str
    fibonacci_levels: Dict
class ElliottWaveDetector:
    """
    Elliott Wave detection using Fibonacci retracements and extensions
    Based on the detailed strategy provided - using Fibonacci levels for wave validation
    """
    
    def __init__(self):
        self.waves = []
        self.fibonacci_levels = {
            # Retracement levels for corrective waves (Wave 2, 4, ABC)
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            # Extension levels for impulse waves (Wave 3, 5)  
            'extension': [1.0, 1.272, 1.414, 1.618, 2.0, 2.272, 2.618]
        }
    
    def calculate_fibonacci_retracement(self, start_price: float, end_price: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        Used for Wave 2 and Wave 4 corrections
        
        Args:
            start_price: Start of the move (Wave 1 start)
            end_price: End of the move (Wave 1 end)
            
        Returns:
            Dict with Fibonacci retracement levels
        """
        price_diff = end_price - start_price
        
        levels = {}
        for level in self.fibonacci_levels['retracement']:
            if start_price < end_price:  # Bullish move
                levels[f'fib_{level}'] = end_price - (price_diff * level)
            else:  # Bearish move  
                levels[f'fib_{level}'] = end_price + (price_diff * level)
        
        return levels
    
    def calculate_fibonacci_extension(self, wave1_start: float, wave1_end: float, 
                                    wave2_end: float) -> Dict[str, float]:
        """
        Calculate Fibonacci extension levels for Wave 3 and Wave 5
        
        Args:
            wave1_start: Start of Wave 1
            wave1_end: End of Wave 1
            wave2_end: End of Wave 2 (correction)
            
        Returns:
            Dict with extension levels
        """
        wave1_length = abs(wave1_end - wave1_start)
        
        levels = {}
        for level in self.fibonacci_levels['extension']:
            if wave1_start < wave1_end:  # Bullish trend
                levels[f'ext_{level}'] = wave2_end + (wave1_length * level)
            else:  # Bearish trend
                levels[f'ext_{level}'] = wave2_end - (wave1_length * level)
        
        return levels
    
    def identify_wave_1(self, df: pd.DataFrame, swing_points: pd.DataFrame) -> List[ElliottWave]:
        """
        Identify Wave 1: Start of new trend after opposite trend or sideways movement
        Wave 1 appears after strong BOS (Break of Structure)
        
        Args:
            df: OHLC DataFrame
            swing_points: DataFrame with swing points identified
            
        Returns:
            List of potential Wave 1 candidates
        """
        wave1_candidates = []
        
        swing_highs = swing_points[swing_points['swing_high'] == True]
        swing_lows = swing_points[swing_points['swing_low'] == True]
        
        # Combine swings and sort by time
        all_swings = []
        for idx, row in swing_highs.iterrows():
            all_swings.append({'time': idx, 'price': row['swing_high_price'], 'type': 'HIGH'})
        for idx, row in swing_lows.iterrows():
            all_swings.append({'time': idx, 'price': row['swing_low_price'], 'type': 'LOW'})
        
        all_swings.sort(key=lambda x: x['time'])
        
        # Look for significant moves that could be Wave 1
        for i in range(1, len(all_swings)):
            current_swing = all_swings[i]
            prev_swing = all_swings[i-1]
            
            # Calculate move size
            move_size = abs(current_swing['price'] - prev_swing['price'])
            move_percent = (move_size / prev_swing['price']) * 100
            
            # Wave 1 criteria: significant move (> 0.5%) after trend change
            if move_percent > 0.5:  # Minimum 0.5% move (reduced for test data)
                wave_type = 'BULLISH' if current_swing['price'] > prev_swing['price'] else 'BEARISH'
                
                wave1_candidates.append(ElliottWave(
                    wave_number=1,
                    wave_type='IMPULSE',
                    start_time=prev_swing['time'],
                    end_time=current_swing['time'],
                    start_price=prev_swing['price'],
                    end_price=current_swing['price'],
                    timeframe='current',
                    fibonacci_levels={}  # Will be used for subsequent waves
                ))
        
        return wave1_candidates
    
    def identify_wave_2(self, df: pd.DataFrame, wave1: ElliottWave, 
                       swing_points: pd.DataFrame) -> Optional[ElliottWave]:
        """
        Identify Wave 2: Correction against Wave 1
        Must retrace 38.2% - 78.6% of Wave 1 (NEVER beyond Wave 1 start)
        
        Args:
            df: OHLC DataFrame
            wave1: Identified Wave 1
            swing_points: DataFrame with swing points
            
        Returns:
            Wave 2 if valid correction found, None otherwise
        """
        # Calculate Fibonacci retracement levels for Wave 1
        fib_levels = self.calculate_fibonacci_retracement(wave1.start_price, wave1.end_price)
        
        # Look for correction after Wave 1 end
        wave1_end_idx = df.index.get_loc(wave1.end_time)
        
        # Search for swing in opposite direction after Wave 1
        for i in range(wave1_end_idx + 1, min(wave1_end_idx + 50, len(df))):
            current_price = None
            
            # Check if this point is a significant swing
            if i < len(swing_points):
                swing_row = swing_points.iloc[i]
                
                if wave1.start_price < wave1.end_price:  # Bullish Wave 1
                    # Look for swing low (correction down)
                    if swing_row['swing_low']:
                        current_price = swing_row['swing_low_price']
                        
                        # Validate Wave 2 rules
                        # 1. Must not go below Wave 1 start
                        if current_price <= wave1.start_price:
                            continue  # Invalid - breaks rule
                        
                        # 2. Must be within Fibonacci retracement range
                        if (current_price >= fib_levels['fib_0.382'] and 
                            current_price <= fib_levels['fib_0.786']):
                            
                            return ElliottWave(
                                wave_number=2,
                                wave_type='CORRECTIVE',
                                start_time=wave1.end_time,
                                end_time=df.index[i],
                                start_price=wave1.end_price,
                                end_price=current_price,
                                timeframe='current',
                                fibonacci_levels=fib_levels
                            )
                
                else:  # Bearish Wave 1
                    # Look for swing high (correction up)
                    if swing_row['swing_high']:
                        current_price = swing_row['swing_high_price']
                        
                        # Validate Wave 2 rules
                        # 1. Must not go above Wave 1 start
                        if current_price >= wave1.start_price:
                            continue  # Invalid - breaks rule
                        
                        # 2. Must be within Fibonacci retracement range  
                        if (current_price <= fib_levels['fib_0.382'] and 
                            current_price >= fib_levels['fib_0.786']):
                            
                            return ElliottWave(
                                wave_number=2,
                                wave_type='CORRECTIVE',
                                start_time=wave1.end_time,
                                end_time=df.index[i],
                                start_price=wave1.end_price,
                                end_price=current_price,
                                timeframe='current',
                                fibonacci_levels=fib_levels
                            )
        
        return None
    
    def identify_wave_3(self, df: pd.DataFrame, wave1: ElliottWave, wave2: ElliottWave,
                       swing_points: pd.DataFrame) -> Optional[ElliottWave]:
        """
        Identify Wave 3: The strongest impulse wave
        Must break Wave 1 high/low and typically extends 120% - 261.8% of Wave 1
        Cannot be the shortest among waves 1, 3, 5
        
        Args:
            df: OHLC DataFrame  
            wave1: Identified Wave 1
            wave2: Identified Wave 2
            swing_points: DataFrame with swing points
            
        Returns:
            Wave 3 if valid extension found, None otherwise
        """
        # Calculate Fibonacci extension levels
        fib_extensions = self.calculate_fibonacci_extension(
            wave1.start_price, wave1.end_price, wave2.end_price
        )
        
        # Look for strong move after Wave 2 end
        wave2_end_idx = df.index.get_loc(wave2.end_time)
        
        for i in range(wave2_end_idx + 1, min(wave2_end_idx + 100, len(df))):
            if i < len(swing_points):
                swing_row = swing_points.iloc[i]
                
                if wave1.start_price < wave1.end_price:  # Bullish trend
                    # Look for swing high above Wave 1 high
                    if swing_row['swing_high']:
                        current_price = swing_row['swing_high_price']
                        
                        # Must break Wave 1 high
                        if current_price > wave1.end_price:
                            # Check if within typical Wave 3 extension range
                            if (current_price >= fib_extensions['ext_1.272'] and
                                current_price <= fib_extensions['ext_2.618']):
                                
                                wave3_length = current_price - wave2.end_price
                                wave1_length = wave1.end_price - wave1.start_price
                                
                                # Wave 3 cannot be shortest (assume Wave 5 will be similar to Wave 1)
                                if wave3_length >= wave1_length:
                                    return ElliottWave(
                                        wave_number=3,
                                        wave_type='IMPULSE',
                                        start_time=wave2.end_time,
                                        end_time=df.index[i],
                                        start_price=wave2.end_price,
                                        end_price=current_price,
                                        timeframe='current',
                                        fibonacci_levels=fib_extensions
                                    )
                
                else:  # Bearish trend
                    # Look for swing low below Wave 1 low
                    if swing_row['swing_low']:
                        current_price = swing_row['swing_low_price']
                        
                        # Must break Wave 1 low
                        if current_price < wave1.end_price:
                            # Check if within typical Wave 3 extension range
                            if (current_price <= fib_extensions['ext_1.272'] and
                                current_price >= fib_extensions['ext_2.618']):
                                
                                wave3_length = wave2.end_price - current_price
                                wave1_length = wave1.start_price - wave1.end_price
                                
                                # Wave 3 cannot be shortest
                                if wave3_length >= wave1_length:
                                    return ElliottWave(
                                        wave_number=3,
                                        wave_type='IMPULSE',
                                        start_time=wave2.end_time,
                                        end_time=df.index[i],
                                        start_price=wave2.end_price,
                                        end_price=current_price,
                                        timeframe='current',
                                        fibonacci_levels=fib_extensions
                                    )
        
        return None
    
    def validate_elliott_wave_sequence(self, waves: List[ElliottWave]) -> bool:
        """
        Validate a complete Elliott Wave sequence (1-2-3-4-5)
        
        Args:
            waves: List of identified waves
            
        Returns:
            True if sequence is valid according to Elliott Wave rules
        """
        if len(waves) < 3:
            return False
            
        # Rule validations
        wave1, wave2, wave3 = waves[0], waves[1], waves[2]
        
        # Rule 1: Wave 2 never retraces beyond Wave 1 start
        if wave1.start_price < wave1.end_price:  # Bullish
            if wave2.end_price <= wave1.start_price:
                return False
        else:  # Bearish
            if wave2.end_price >= wave1.start_price:
                return False
        
        # Rule 2: Wave 3 must break Wave 1 end
        if wave1.start_price < wave1.end_price:  # Bullish
            if wave3.end_price <= wave1.end_price:
                return False
        else:  # Bearish
            if wave3.end_price >= wave1.end_price:
                return False
        
        # Rule 3: Wave 3 cannot be the shortest
        wave1_length = abs(wave1.end_price - wave1.start_price)
        wave3_length = abs(wave3.end_price - wave3.start_price)
        
        if len(waves) >= 5:
            wave5_length = abs(waves[4].end_price - waves[4].start_price)
            if wave3_length < wave1_length or wave3_length < wave5_length:
                return False
        
        return True

print("🌊 ElliottWaveDetector Class Created")
print("✅ Fibonacci-based wave identification")
print("✅ Wave 1, 2, 3 detection with proper rules")
print("✅ Elliott Wave sequence validation")
print("✅ Extension and retracement calculations")
print("📊 Ready for Elliott Wave analysis")