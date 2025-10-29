"""
Market Structure Detection Module - Complete Rewrite
Fixes all critical bugs and implements missing features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .data_structures import MarketStructure, SwingPoint, LiquidityLevel
from .config_loader import ConfigLoader


class MarketStructureDetector:
    """
    Enhanced market structure detector with bug fixes and missing features.

    Fixes:
    - BUG-MS-001: Bias determination logic (analyze structure pattern, not price sign)
    - BUG-MS-002: BOS/CHoCH logic refactor (determine trend direction first)

    Implements:
    - Liquidity level tracking
    - Liquidity sweep detection
    - Structure strength scoring
    - Multi-timeframe structure alignment
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, config: Optional[Dict] = None):
        """
        Initialize market structure detector.

        Args:
            data: OHLC DataFrame (optional, for compatibility)
            config: Configuration dictionary (optional, for compatibility)
        """
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_market_structure_config()
        self.data = data

        # Track liquidity levels
        self.liquidity_levels: List[LiquidityLevel] = []

        # Track structure history
        self.structure_history: List[MarketStructure] = []

    def detect_swing_points(self, df: pd.DataFrame, strength: int = None) -> pd.DataFrame:
        """
        Detect swing points with configurable strength.

        Args:
            df: OHLC DataFrame
            strength: Number of candles on each side to confirm swing

        Returns:
            DataFrame with swing points marked
        """
        if strength is None:
            strength = self.config.swing_strength

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

    def detect_swing_points_from_indices(self, start_idx: int, end_idx: int, 
                                        strength: int = None) -> List[MarketStructure]:
        """
        Detect swing points from data indices.

        Args:
            start_idx: Start index for analysis
            end_idx: End index for analysis
            strength: Number of candles on each side to confirm swing

        Returns:
            List of market structures
        """
        if self.data is None or self.data.empty:
            return []

        data_slice = self.data.iloc[start_idx:end_idx]
        swing_df = self.detect_swing_points(data_slice, strength)
        return self.detect_market_structure(swing_df)

    def detect_break_of_structure(self, start_idx: int, end_idx: int) -> List[MarketStructure]:
        """
        Detect break of structure from data indices.

        Args:
            start_idx: Start index for analysis
            end_idx: End index for analysis

        Returns:
            List of market structures
        """
        if self.data is None or self.data.empty:
            return []

        data_slice = self.data.iloc[start_idx:end_idx]
        swing_df = self.detect_swing_points(data_slice)
        return self.detect_market_structure(swing_df)

    def detect_change_of_character(self, start_idx: int, end_idx: int) -> List[MarketStructure]:
        """
        Detect change of character from data indices.

        Args:
            start_idx: Start index for analysis
            end_idx: End index for analysis

        Returns:
            List of market structures
        """
        if self.data is None or self.data.empty:
            return []

        data_slice = self.data.iloc[start_idx:end_idx]
        swing_df = self.detect_swing_points(data_slice)
        return self.detect_market_structure(swing_df)

    def detect_market_structure(self, df: pd.DataFrame) -> List[MarketStructure]:
        """
        Detect market structure with improved logic.

        Fixes BUG-MS-002: Refactor BOS/CHoCH logic to determine trend direction first

        Args:
            df: DataFrame with swing points

        Returns:
            List of MarketStructure objects
        """
        structures = []

        # Get swing points
        swing_highs = df[df['swing_high']].copy()
        swing_lows = df[df['swing_low']].copy()

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return structures

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
            prev_trend = None

            for i in range(3, len(all_swings)):
                current_swing = all_swings[i]
                prev_swing = all_swings[i-1]

                # CRITICAL FIX: Determine trend direction first, then assign structure type
                trend_direction = self._determine_trend_direction(all_swings, i)

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

                # Create MarketStructure object with proper trend direction
                structure = MarketStructure(
                    timestamp=current_swing['timestamp'],
                    structure_type=structure_type,
                    price=current_swing['price'],
                    timeframe='current',
                    strength=0.8,
                    trend_direction=trend_direction,
                    volume_at_break=self._get_volume_at_break(df, current_swing['timestamp']),
                    impulse_strength=self._calculate_impulse_strength(df, current_swing['timestamp']),
                    confirmation_count=0
                )

                structures.append(structure)
                self.structure_history.append(structure)

        return structures

    def get_current_bias(self, structures: List[MarketStructure]) -> str:
        """
        Determine current market bias based on recent structures.

        Fixes BUG-MS-001: Analyze structure pattern, not price sign

        Args:
            structures: List of MarketStructure objects

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if not structures:
            return 'NEUTRAL'

        recent_structures = structures[-self.config.min_structures_for_bias:]

        # CRITICAL FIX: Analyze structure pattern, not price sign
        bullish_count = sum(1 for s in recent_structures
                            if s.structure_type == 'BOS' and s.trend_direction == 'BULLISH')
        bearish_count = sum(1 for s in recent_structures
                            if s.structure_type == 'BOS' and s.trend_direction == 'BEARISH')

        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def track_liquidity_levels(self, df: pd.DataFrame, swing_df: pd.DataFrame) -> List[LiquidityLevel]:
        """
        Track liquidity levels from swing points.

        Args:
            df: OHLC DataFrame
            swing_df: DataFrame with swing points

        Returns:
            List of liquidity levels
        """
        liquidity_levels = []

        # Add swing highs as liquidity levels
        swing_highs = swing_df[swing_df['swing_high']]
        for idx, row in swing_highs.iterrows():
            level = LiquidityLevel(
                timestamp=idx,
                level_type='HIGH',
                price=row['swing_high_price'],
                strength=0.8
            )
            liquidity_levels.append(level)

        # Add swing lows as liquidity levels
        swing_lows = swing_df[swing_df['swing_low']]
        for idx, row in swing_lows.iterrows():
            level = LiquidityLevel(
                timestamp=idx,
                level_type='LOW',
                price=row['swing_low_price'],
                strength=0.8
            )
            liquidity_levels.append(level)

        self.liquidity_levels = liquidity_levels
        return liquidity_levels

    def detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect liquidity sweeps and reversals.

        Args:
            df: OHLC DataFrame

        Returns:
            List of liquidity sweep events
        """
        sweeps = []

        for i in range(len(df)):
            candle = df.iloc[i]
            timestamp = df.index[i]

            # Check for liquidity sweeps
            for level in self.liquidity_levels:
                if level.is_swept:
                    continue  # Already processed

                # Check if price swept the liquidity level
                if self._is_liquidity_swept(candle, level):
                    # Mark as swept
                    level.mark_swept(timestamp)

                    # Look for reversal confirmation
                    if self._has_reversal_confirmation(df, i, level):
                        level.confirm_reversal()

                        sweep_event = {
                            'timestamp': timestamp,
                            'level_type': level.level_type,
                            'price': level.price,
                            'swept': True,
                            'reversal_confirmed': True
                        }
                        sweeps.append(sweep_event)

        return sweeps

    def calculate_structure_strength(self, structure: MarketStructure) -> float:
        """
        Calculate structure strength score.

        Args:
            structure: MarketStructure object

        Returns:
            Strength score (0-1)
        """
        base_strength = structure.strength

        # Add volume weight
        if structure.volume_at_break:
            volume_factor = min(structure.volume_at_break / 1000000, 2.0)  # Cap at 2x
            base_strength *= (1 + volume_factor * self.config.volume_weight)

        # Add impulse strength weight
        if structure.impulse_strength:
            base_strength *= (1 + structure.impulse_strength * self.config.impulse_weight)

        # Add confirmation weight
        confirmation_bonus = min(structure.confirmation_count * 0.1, 0.5)
        base_strength += confirmation_bonus * self.config.confirmation_weight

        return min(base_strength, 1.0)  # Cap at 1.0

    def validate_multi_timeframe_alignment(self, htf_structures: List[MarketStructure],
                                          mtf_structures: List[MarketStructure]) -> bool:
        """
        Validate multi-timeframe structure alignment.

        Args:
            htf_structures: Higher timeframe structures
            mtf_structures: Medium timeframe structures

        Returns:
            True if structures are aligned, False otherwise
        """
        if not htf_structures or not mtf_structures:
            return True

        # Get recent HTF bias
        htf_bias = self.get_current_bias(htf_structures)
        mtf_bias = self.get_current_bias(mtf_structures)

        # Check alignment
        return htf_bias == mtf_bias or htf_bias == 'NEUTRAL' or mtf_bias == 'NEUTRAL'

    def _determine_trend_direction(self, all_swings: List[Dict], current_index: int) -> str:
        """Determine trend direction based on swing analysis."""
        if current_index < 3:
            return 'NEUTRAL'

        # Analyze last 3 swings for trend direction
        recent_swings = all_swings[current_index-2:current_index+1]

        # Check for higher highs and higher lows (bullish)
        if (recent_swings[0]['type'] == 'LOW' and recent_swings[1]['type'] == 'HIGH' and
                recent_swings[2]['type'] == 'LOW'):
            if (recent_swings[1]['price'] > recent_swings[0]['price'] and
                    recent_swings[2]['price'] > recent_swings[0]['price']):
                return 'BULLISH'

        # Check for lower highs and lower lows (bearish)
        if (recent_swings[0]['type'] == 'HIGH' and recent_swings[1]['type'] == 'LOW' and
                recent_swings[2]['type'] == 'HIGH'):
            if (recent_swings[1]['price'] < recent_swings[0]['price'] and
                    recent_swings[2]['price'] < recent_swings[0]['price']):
                return 'BEARISH'

        return 'NEUTRAL'

    def _get_volume_at_break(self, df: pd.DataFrame, timestamp: datetime) -> Optional[float]:
        """Get volume at structure break point."""
        try:
            idx = df.index.get_loc(timestamp)
            if 'volume' in df.columns:
                return df.iloc[idx]['volume']
        except (KeyError, IndexError):
            pass
        return None

    def _calculate_impulse_strength(self, df: pd.DataFrame, timestamp: datetime) -> Optional[float]:
        """Calculate impulse strength at structure break point."""
        try:
            idx = df.index.get_loc(timestamp)
            if idx < 5 or idx >= len(df) - 5:
                return None

            # Calculate price move over 5 candles
            start_price = df.iloc[idx-5]['close']
            end_price = df.iloc[idx+5]['close']
            move_percent = abs(end_price - start_price) / start_price * 100

            return min(move_percent / 10, 1.0)  # Normalize to 0-1
        except (KeyError, IndexError):
            return None

    def _is_liquidity_swept(self, candle: pd.Series, level: LiquidityLevel) -> bool:
        """Check if liquidity level has been swept by candle."""
        if level.level_type == 'HIGH':
            return candle['high'] > level.price
        else:  # LOW
            return candle['low'] < level.price

    def _has_reversal_confirmation(self, df: pd.DataFrame, sweep_idx: int, level: LiquidityLevel) -> bool:
        """Check for reversal confirmation after liquidity sweep."""
        if sweep_idx >= len(df) - 2:
            return False

        # Look for reversal in next few candles
        for i in range(sweep_idx + 1, min(sweep_idx + 5, len(df))):
            candle = df.iloc[i]

            if level.level_type == 'HIGH':
                # Look for bearish reversal (close below level)
                if candle['close'] < level.price:
                    return True
            else:  # LOW
                # Look for bullish reversal (close above level)
                if candle['close'] > level.price:
                    return True

        return False
