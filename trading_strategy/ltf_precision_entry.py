"""
LTF Precision Entry Refinement Module
Implements micro-level confirmation for MTF signals using LTF data.

Features:
- Micro OB/FVG/OTE detection on LTF (1m/5m)
- CHoCH confirmation on LTF
- Gates MTF signals by LTF confirmation
- Improves average RR with tighter SL
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from .data_structures import Signal, ICTConcept, Confirmation
from .ict_concepts import ICTConceptsDetector
from .market_structure import MarketStructureDetector
from .config_loader import ConfigLoader


class LTFPrecisionEntry:
    """
    LTF Precision Entry Refinement for better entry timing and tighter stops.

    This module:
    1. Loads LTF data (1m/5m) for micro-level analysis
    2. Detects micro OB/FVG/OTE and CHoCH on LTF
    3. Gates MTF signals by LTF confirmation
    4. Provides tighter SL placement based on LTF structure
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize LTF precision entry detector.

        Args:
            config_loader: Configuration loader instance
        """
        self.config_loader = config_loader or ConfigLoader()
        self.ltf_config = self.config_loader.get_ltf_config()

        # Initialize detectors for LTF analysis
        self.ict_detector = ICTConceptsDetector(self.config_loader)
        self.market_structure = MarketStructureDetector(self.config_loader)

        # LTF analysis cache
        self.ltf_analysis_cache: Dict[str, Dict] = {}
        self.cache_ttl_minutes = 5  # Cache LTF analysis for 5 minutes

    def refine_mtf_signals_with_ltf(self, mtf_signals: List[Signal],
                                   ltf_data: pd.DataFrame,
                                   mtf_analysis: Dict) -> List[Signal]:
        """
        Refine MTF signals using LTF confirmation.

        Args:
            mtf_signals: List of MTF signals to refine
            ltf_data: LTF DataFrame (1m/5m)
            mtf_analysis: MTF analysis results for context

        Returns:
            List of refined signals with LTF confirmation
        """
        if not mtf_signals or ltf_data.empty:
            return []

        refined_signals = []

        for signal in mtf_signals:
            # Get LTF analysis around signal timestamp
            ltf_analysis = self._get_ltf_analysis_around_signal(ltf_data, signal)

            if not ltf_analysis:
                continue

            # Check LTF confirmation
            ltf_confirmation = self._check_ltf_confirmation(signal, ltf_analysis)

            if ltf_confirmation['confirmed']:
                # Refine signal with LTF data
                refined_signal = self._refine_signal_with_ltf(signal, ltf_analysis, ltf_confirmation)
                if refined_signal:
                    refined_signals.append(refined_signal)

        return refined_signals

    def detect_micro_concepts(self, ltf_data: pd.DataFrame) -> Dict:
        """
        Detect micro-level ICT concepts on LTF.

        Args:
            ltf_data: LTF DataFrame

        Returns:
            Dictionary with micro concepts detected
        """
        # Detect swing points on LTF
        swing_df = self.market_structure.detect_swing_points(ltf_data)

        # Detect micro FVGs
        micro_fvgs = self.ict_detector.detect_fvg(ltf_data,
                                                 min_gap_percent=self.ltf_config.micro_fvg_min_gap)

        # Detect micro Order Blocks
        micro_order_blocks = self.ict_detector.detect_order_blocks(ltf_data, swing_df)

        # Detect micro OTE zones
        micro_ote_zones = self.ict_detector.detect_ote_zones(ltf_data, swing_df,
                                                           lookback_days=self.ltf_config.micro_ote_lookback_days)

        # Detect micro liquidity grabs
        micro_liquidity_grabs = self.ict_detector.detect_liquidity_grabs(ltf_data, swing_df)

        # Detect micro CHoCH
        micro_choch = self._detect_micro_choch(ltf_data, swing_df)

        return {
            'swing_points': swing_df,
            'micro_fvgs': micro_fvgs,
            'micro_order_blocks': micro_order_blocks,
            'micro_ote_zones': micro_ote_zones,
            'micro_liquidity_grabs': micro_liquidity_grabs,
            'micro_choch': micro_choch,
            'dataframe': ltf_data
        }

    def calculate_tighter_stop_loss(self, signal: Signal, ltf_analysis: Dict) -> float:
        """
        Calculate tighter stop loss based on LTF structure.

        Args:
            signal: Original signal
            ltf_analysis: LTF analysis results

        Returns:
            Refined stop loss price
        """
        # Get micro concepts near signal price
        micro_concepts = self._get_micro_concepts_near_price(signal.price, ltf_analysis)

        if not micro_concepts:
            return signal.stop_loss  # No refinement possible

        # Calculate ATR on LTF for dynamic stop sizing
        atr = self._calculate_ltf_atr(ltf_analysis['dataframe'])

        if signal.signal_type == 'BUY':
            # For long positions, find nearest support below entry
            support_levels = self._find_support_levels(signal.price, micro_concepts)

            if support_levels:
                # Use strongest support with ATR buffer
                strongest_support = max(support_levels, key=lambda x: x['strength'])
                tighter_sl = strongest_support['price'] - (atr * self.ltf_config.atr_sl_buffer)
                return max(tighter_sl, signal.stop_loss)  # Don't make SL worse
            else:
                # Use ATR-based stop
                tighter_sl = signal.price - (atr * self.ltf_config.atr_sl_multiplier)
                return max(tighter_sl, signal.stop_loss)

        else:  # SELL
            # For short positions, find nearest resistance above entry
            resistance_levels = self._find_resistance_levels(signal.price, micro_concepts)

            if resistance_levels:
                # Use strongest resistance with ATR buffer
                strongest_resistance = max(resistance_levels, key=lambda x: x['strength'])
                tighter_sl = strongest_resistance['price'] + (atr * self.ltf_config.atr_sl_buffer)
                return min(tighter_sl, signal.stop_loss)  # Don't make SL worse
            else:
                # Use ATR-based stop
                tighter_sl = signal.price + (atr * self.ltf_config.atr_sl_multiplier)
                return min(tighter_sl, signal.stop_loss)

    def _get_ltf_analysis_around_signal(self, ltf_data: pd.DataFrame, signal: Signal) -> Optional[Dict]:
        """
        Get LTF analysis around signal timestamp.

        Args:
            ltf_data: LTF DataFrame
            signal: Signal to analyze around

        Returns:
            LTF analysis results or None
        """
        # Define time window around signal
        window_minutes = self.ltf_config.analysis_window_minutes
        start_time = signal.timestamp - timedelta(minutes=window_minutes)
        end_time = signal.timestamp + timedelta(minutes=window_minutes)

        # Filter LTF data to window
        window_data = ltf_data[(ltf_data.index >= start_time) & (ltf_data.index <= end_time)]

        if window_data.empty:
            return None

        # Check cache first
        cache_key = f"{signal.timestamp}_{window_minutes}"
        if cache_key in self.ltf_analysis_cache:
            cached_analysis = self.ltf_analysis_cache[cache_key]
            if datetime.now() - cached_analysis['timestamp'] < timedelta(minutes=self.cache_ttl_minutes):
                return cached_analysis['analysis']

        # Perform LTF analysis
        analysis = self.detect_micro_concepts(window_data)

        # Cache the analysis
        self.ltf_analysis_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': datetime.now()
        }

        return analysis

    def _check_ltf_confirmation(self, signal: Signal, ltf_analysis: Dict) -> Dict:
        """
        Check if LTF confirms the MTF signal.

        Args:
            signal: MTF signal to confirm
            ltf_analysis: LTF analysis results

        Returns:
            Confirmation result dictionary
        """
        confirmations = []
        confirmation_score = 0.0

        # Check micro FVG confirmation
        micro_fvgs = ltf_analysis.get('micro_fvgs', [])
        for fvg in micro_fvgs:
            if fvg.is_price_in_zone(signal.price):
                if ((signal.signal_type == 'BUY' and fvg.is_bullish()) or
                    (signal.signal_type == 'SELL' and fvg.is_bearish())):
                    confirmations.append('MICRO_FVG')
                    confirmation_score += self.ltf_config.confirmation_weights['micro_fvg']

        # Check micro Order Block confirmation
        micro_order_blocks = ltf_analysis.get('micro_order_blocks', [])
        for ob in micro_order_blocks:
            if ob.is_price_in_zone(signal.price):
                confirmations.append('MICRO_OB')
                confirmation_score += self.ltf_config.confirmation_weights['micro_ob']

        # Check micro OTE confirmation
        micro_ote_zones = ltf_analysis.get('micro_ote_zones', [])
        for ote in micro_ote_zones:
            if ote.is_price_in_zone(signal.price):
                confirmations.append('MICRO_OTE')
                confirmation_score += self.ltf_config.confirmation_weights['micro_ote']

        # Check micro CHoCH confirmation
        micro_choch = ltf_analysis.get('micro_choch', [])
        for choch in micro_choch:
            if abs((choch['timestamp'] - signal.timestamp).total_seconds()) < 300:  # Within 5 minutes
                if ((signal.signal_type == 'BUY' and choch['direction'] == 'BULLISH') or
                    (signal.signal_type == 'SELL' and choch['direction'] == 'BEARISH')):
                    confirmations.append('MICRO_CHOCH')
                    confirmation_score += self.ltf_config.confirmation_weights['micro_choch']

        # Check micro liquidity grab confirmation
        micro_liquidity_grabs = ltf_analysis.get('micro_liquidity_grabs', [])
        for grab in micro_liquidity_grabs:
            if abs((grab.timestamp - signal.timestamp).total_seconds()) < 300:  # Within 5 minutes
                confirmations.append('MICRO_LIQUIDITY_GRAB')
                confirmation_score += self.ltf_config.confirmation_weights['micro_liquidity_grab']

        # Determine if confirmed
        min_confirmation_score = self.ltf_config.min_confirmation_score
        confirmed = confirmation_score >= min_confirmation_score

        return {
            'confirmed': confirmed,
            'confirmations': confirmations,
            'confirmation_score': confirmation_score,
            'min_score_required': min_confirmation_score
        }

    def _refine_signal_with_ltf(self, signal: Signal, ltf_analysis: Dict,
                               ltf_confirmation: Dict) -> Optional[Signal]:
        """
        Refine signal with LTF data.

        Args:
            signal: Original signal
            ltf_analysis: LTF analysis results
            ltf_confirmation: LTF confirmation results

        Returns:
            Refined signal or None
        """
        try:
            # Calculate tighter stop loss
            tighter_sl = self.calculate_tighter_stop_loss(signal, ltf_analysis)

            # Calculate refined risk-reward ratio
            risk = abs(signal.price - tighter_sl)
            reward = abs(signal.take_profits[0] - signal.price)
            refined_rr = reward / risk if risk > 0 else 0

            # Only proceed if RR improved
            if refined_rr <= signal.risk_reward:
                return None

            # Create refined signal
            refined_signal = Signal(
                timestamp=signal.timestamp,
                signal_type=signal.signal_type,
                entry_type=f"{signal.entry_type}_LTF_REFINED",
                price=signal.price,
                stop_loss=tighter_sl,
                take_profits=signal.take_profits,
                risk_reward=refined_rr,
                confidence=min(signal.confidence + 0.1, 1.0),  # Boost confidence slightly
                metadata={
                    **signal.metadata,
                    'ltf_confirmation': ltf_confirmation,
                    'original_sl': signal.stop_loss,
                    'refined_sl': tighter_sl,
                    'rr_improvement': refined_rr - signal.risk_reward
                }
            )

            return refined_signal

        except Exception as e:
            print(f"Error refining signal with LTF: {e}")
            return None

    def _detect_micro_choch(self, ltf_data: pd.DataFrame, swing_df: pd.DataFrame) -> List[Dict]:
        """
        Detect micro CHoCH on LTF.

        Args:
            ltf_data: LTF DataFrame
            swing_df: DataFrame with swing points

        Returns:
            List of micro CHoCH events
        """
        micro_choch = []

        # Get swing highs and lows
        swing_highs = swing_df[swing_df['swing_high']].sort_index()
        swing_lows = swing_df[swing_df['swing_low']].sort_index()

        # Look for CHoCH patterns
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs.iloc[i-1]
            curr_high = swing_highs.iloc[i]

            # Check for bearish CHoCH (lower high)
            if curr_high['swing_high_price'] < prev_high['swing_high_price']:
                micro_choch.append({
                    'timestamp': curr_high.name,
                    'direction': 'BEARISH',
                    'type': 'CHOCH',
                    'price': curr_high['swing_high_price'],
                    'strength': abs(curr_high['swing_high_price'] - prev_high['swing_high_price']) / prev_high['swing_high_price']
                })

        for i in range(1, len(swing_lows)):
            prev_low = swing_lows.iloc[i-1]
            curr_low = swing_lows.iloc[i]

            # Check for bullish CHoCH (higher low)
            if curr_low['swing_low_price'] > prev_low['swing_low_price']:
                micro_choch.append({
                    'timestamp': curr_low.name,
                    'direction': 'BULLISH',
                    'type': 'CHOCH',
                    'price': curr_low['swing_low_price'],
                    'strength': abs(curr_low['swing_low_price'] - prev_low['swing_low_price']) / prev_low['swing_low_price']
                })

        return micro_choch

    def _get_micro_concepts_near_price(self, price: float, ltf_analysis: Dict) -> List[Dict]:
        """
        Get micro concepts near the given price.

        Args:
            price: Price to search around
            ltf_analysis: LTF analysis results

        Returns:
            List of micro concepts near price
        """
        concepts = []
        price_tolerance = price * self.ltf_config.price_tolerance_percent

        # Add micro FVGs
        for fvg in ltf_analysis.get('micro_fvgs', []):
            if abs(fvg.start_price - price) <= price_tolerance or abs(fvg.end_price - price) <= price_tolerance:
                concepts.append({
                    'type': 'FVG',
                    'price': (fvg.start_price + fvg.end_price) / 2,
                    'strength': fvg.strength,
                    'direction': 'BULLISH' if fvg.is_bullish() else 'BEARISH'
                })

        # Add micro Order Blocks
        for ob in ltf_analysis.get('micro_order_blocks', []):
            if abs(ob.start_price - price) <= price_tolerance or abs(ob.end_price - price) <= price_tolerance:
                concepts.append({
                    'type': 'OB',
                    'price': (ob.start_price + ob.end_price) / 2,
                    'strength': ob.strength,
                    'direction': 'BULLISH' if ob.is_bullish() else 'BEARISH'
                })

        # Add micro OTE zones
        for ote in ltf_analysis.get('micro_ote_zones', []):
            if abs(ote.start_price - price) <= price_tolerance or abs(ote.end_price - price) <= price_tolerance:
                concepts.append({
                    'type': 'OTE',
                    'price': (ote.start_price + ote.end_price) / 2,
                    'strength': ote.strength,
                    'direction': 'BULLISH' if ote.is_bullish() else 'BEARISH'
                })

        return concepts

    def _find_support_levels(self, price: float, micro_concepts: List[Dict]) -> List[Dict]:
        """
        Find support levels below the given price.

        Args:
            price: Reference price
            micro_concepts: List of micro concepts

        Returns:
            List of support levels
        """
        support_levels = []

        for concept in micro_concepts:
            concept_price = concept['price']

            # Only consider concepts below the price
            if concept_price < price:
                # Only consider bullish concepts as support
                if concept['direction'] == 'BULLISH':
                    support_levels.append({
                        'price': concept_price,
                        'strength': concept['strength'],
                        'type': concept['type']
                    })

        return support_levels

    def _find_resistance_levels(self, price: float, micro_concepts: List[Dict]) -> List[Dict]:
        """
        Find resistance levels above the given price.

        Args:
            price: Reference price
            micro_concepts: List of micro concepts

        Returns:
            List of resistance levels
        """
        resistance_levels = []

        for concept in micro_concepts:
            concept_price = concept['price']

            # Only consider concepts above the price
            if concept_price > price:
                # Only consider bearish concepts as resistance
                if concept['direction'] == 'BEARISH':
                    resistance_levels.append({
                        'price': concept_price,
                        'strength': concept['strength'],
                        'type': concept['type']
                    })

        return resistance_levels

    def _calculate_ltf_atr(self, ltf_data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate ATR on LTF data.

        Args:
            ltf_data: LTF DataFrame
            period: ATR period

        Returns:
            ATR value
        """
        if len(ltf_data) < period + 1:
            return 0.0

        # Calculate True Range
        high_low = ltf_data['high'] - ltf_data['low']
        high_close_prev = abs(ltf_data['high'] - ltf_data['close'].shift(1))
        low_close_prev = abs(ltf_data['low'] - ltf_data['close'].shift(1))

        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate ATR as simple moving average of True Range
        atr = true_range.rolling(window=period).mean().iloc[-1]

        return atr if not pd.isna(atr) else 0.0
