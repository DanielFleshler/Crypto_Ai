"""
ICT Entry Types Module - Complete Implementation
Implements all 5 ICT entry types with proper validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .data_structures import Signal, ICTConcept, Confirmation, LiquidityLevel
from .config_loader import ConfigLoader


class ICTEntries:
    """
    ICT entry types implementation.

    Implements all 5 ICT entry types:
    1. Liquidity Grab + CHoCH
    2. FVG Entry
    3. Order Block Entry
    4. OTE Entry
    5. Breaker Block Entry
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize ICT entries detector.

        Args:
            config_loader: Configuration loader instance
        """
        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.get_ict_concepts_config()
        self.entry_config = self.config_loader.get_entry_confirmation_config()

    def detect_liquidity_grab_choch_entries(self, df: pd.DataFrame, mtf_data: Dict) -> List[Signal]:
        """
        Entry Type 1: Liquidity Grab + CHoCH

        Detect liquidity sweep beyond swing high/low followed by reversal entry.

        Args:
            df: OHLC DataFrame
            mtf_data: Multi-timeframe analysis data

        Returns:
            List of liquidity grab + CHoCH entry signals
        """
        signals = []

        # Get liquidity grabs and structures
        liquidity_grabs = mtf_data.get('liquidity_grabs', [])
        structures = mtf_data.get('structures', [])
        swing_points = mtf_data.get('swing_points', pd.DataFrame())
        fvgs = mtf_data.get('fvgs', [])
        order_blocks = mtf_data.get('order_blocks', [])

        for grab in liquidity_grabs:
            if not grab.is_liquidity_grab():
                continue

            # Check for CHoCH confirmation
            if not self._has_choch_confirmation(grab.timestamp, structures):
                continue

            # Check for reversal structure
            if not self._has_reversal_structure(grab.timestamp, structures):
                continue

            # Determine entry direction and calculate structure-based stops/TPs
            if grab.concept_type == 'LIQUIDITY_GRAB_HIGH':
                signal_type = 'SELL'
                entry_price = grab.start_price

                # Structure-based stop loss: above the liquidity grab high with small buffer
                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

                # Structure-based take profits
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            else:  # LIQUIDITY_GRAB_LOW
                signal_type = 'BUY'
                entry_price = grab.end_price

                # Structure-based stop loss: below the liquidity grab low with small buffer
                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

                # Structure-based take profits
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0

            # FIXED BUG-RR-001: Use minimum_rr_ratio from config (was hardcoded to 3.0)
            min_rr = self.config_loader.get_risk_management_config().minimum_rr_ratio
            
            # Only create signal if RR meets minimum requirement
            if risk_reward >= min_rr:
                signal = Signal(
                    timestamp=grab.timestamp,
                    signal_type=signal_type,
                    entry_type='LIQUIDITY_GRAB_CHOCH',
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    confidence=0.8,
                    metadata={
                        'liquidity_grab': grab,
                        'confirmation_type': 'CHoCH',
                        'stop_type': 'structure_based',
                        'tp_type': 'structure_based'
                    }
                )

                signals.append(signal)

        return signals

    def detect_fvg_entries(self, df: pd.DataFrame, mtf_data: Dict) -> List[Signal]:
        """
        Entry Type 2: FVG Entry

        Detect unfilled FVG in HTF direction with price retracement.

        Args:
            df: OHLC DataFrame
            mtf_data: Multi-timeframe analysis data

        Returns:
            List of FVG entry signals
        """
        signals = []

        # Get FVGs and HTF bias
        fvgs = mtf_data.get('fvgs', [])
        htf_bias = mtf_data.get('htf_bias', 'NEUTRAL')
        swing_points = mtf_data.get('swing_points', pd.DataFrame())
        order_blocks = mtf_data.get('order_blocks', [])
        structures = mtf_data.get('structures', [])

        for fvg in fvgs:
            if not fvg.is_fvg() or fvg.status == 'filled':
                continue

            # Check HTF bias alignment
            if not self._is_htf_bias_aligned(fvg, htf_bias):
                continue

            # Check for price retracement into FVG
            if not self._is_price_in_fvg_zone(df, fvg):
                continue

            # Check for BOS or structure shift confirmation
            if not self._has_structure_confirmation(fvg.timestamp, mtf_data):
                continue

            # FIXED BUG-ICT-COUNTER-001: Counter-trend FVG logic
            # In bearish market, ALL FVGs are potential SELL setups (rejection or continuation)
            # In bullish market, ALL FVGs are potential BUY setups (rejection or continuation)

            if htf_bias == 'BEARISH':
                # BEARISH MARKET: Trade downtrend
                signal_type = 'SELL'
                if fvg.is_bullish():
                    # Bullish FVG in bearish market = Retracement/resistance zone
                    # Enter SELL at top of FVG (expecting rejection)
                    entry_price = fvg.end_price
                else:
                    # Bearish FVG in bearish market = Continuation zone
                    entry_price = fvg.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            elif htf_bias == 'BULLISH':
                # BULLISH MARKET: Trade uptrend
                signal_type = 'BUY'
                if fvg.is_bearish():
                    # Bearish FVG in bullish market = Retracement/support zone
                    # Enter BUY at bottom of FVG (expecting rejection)
                    entry_price = fvg.start_price
                else:
                    # Bullish FVG in bullish market = Continuation zone
                    entry_price = fvg.start_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            else:  # NEUTRAL - use original logic
                if fvg.is_bullish():
                    signal_type = 'BUY'
                    entry_price = fvg.start_price
                else:
                    signal_type = 'SELL'
                    entry_price = fvg.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0

            # FIXED BUG-RR-002: Use minimum_rr_ratio from config (was hardcoded to 3.0)
            min_rr = self.config_loader.get_risk_management_config().minimum_rr_ratio
            
            # Only create signal if RR meets minimum requirement
            if risk_reward >= min_rr:
                signal = Signal(
                    timestamp=fvg.timestamp,
                    signal_type=signal_type,
                    entry_type='FVG_ENTRY',
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    confidence=0.7,
                    metadata={
                        'fvg': fvg,
                        'confirmation_type': 'BOS',
                        'stop_type': 'structure_based',
                        'tp_type': 'structure_based'
                    }
                )

                signals.append(signal)

        return signals

    def detect_order_block_entries(self, df: pd.DataFrame, mtf_data: Dict) -> List[Signal]:
        """
        Entry Type 3: Order Block Entry

        Fresh OB aligned with HTF bias with price retracement.

        Args:
            df: OHLC DataFrame
            mtf_data: Multi-timeframe analysis data

        Returns:
            List of Order Block entry signals
        """
        signals = []

        # Get Order Blocks and HTF bias
        order_blocks = mtf_data.get('order_blocks', [])
        htf_bias = mtf_data.get('htf_bias', 'NEUTRAL')
        swing_points = mtf_data.get('swing_points', pd.DataFrame())
        fvgs = mtf_data.get('fvgs', [])
        structures = mtf_data.get('structures', [])

        for ob in order_blocks:
            if not ob.is_order_block() or not ob.is_fresh:
                continue

            # Check HTF bias alignment
            if not self._is_htf_bias_aligned(ob, htf_bias):
                continue

            # Check for price retracement to OB zone
            if not self._is_price_in_ob_zone(df, ob):
                continue

            # Check for additional pattern confirmation (FVG/OTE)
            if not self._has_additional_pattern_confirmation(ob, mtf_data):
                continue

            # FIXED BUG-ICT-COUNTER-003: Counter-trend Order Block logic
            if htf_bias == 'BEARISH':
                signal_type = 'SELL'
                if ob.is_bullish():
                    # Bullish OB in bearish market = Resistance zone
                    entry_price = ob.end_price
                else:
                    entry_price = ob.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            elif htf_bias == 'BULLISH':
                signal_type = 'BUY'
                if ob.is_bearish():
                    # Bearish OB in bullish market = Support zone
                    entry_price = ob.start_price
                else:
                    entry_price = ob.start_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            else:  # NEUTRAL
                if ob.is_bullish():
                    signal_type = 'BUY'
                    entry_price = ob.start_price
                else:
                    signal_type = 'SELL'
                    entry_price = ob.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0

            # FIXED BUG-RR-003: Use minimum_rr_ratio from config (was hardcoded to 3.0)
            min_rr = self.config_loader.get_risk_management_config().minimum_rr_ratio
            
            # Only create signal if RR meets minimum requirement
            if risk_reward >= min_rr:
                signal = Signal(
                    timestamp=ob.timestamp,
                    signal_type=signal_type,
                    entry_type='ORDER_BLOCK_ENTRY',
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    confidence=0.8,
                    metadata={
                        'order_block': ob,
                        'confirmation_type': 'FVG_OTE',
                        'stop_type': 'structure_based',
                        'tp_type': 'structure_based'
                    }
                )

                signals.append(signal)

        return signals

    def detect_ote_entries(self, df: pd.DataFrame, mtf_data: Dict) -> List[Signal]:
        """
        Entry Type 4: OTE Entry

        62%-79% retracement zone with HTF bias alignment.

        Args:
            df: OHLC DataFrame
            mtf_data: Multi-timeframe analysis data

        Returns:
            List of OTE entry signals
        """
        signals = []

        # Get OTE zones and HTF bias
        ote_zones = mtf_data.get('ote_zones', [])
        htf_bias = mtf_data.get('htf_bias', 'NEUTRAL')
        swing_points = mtf_data.get('swing_points', pd.DataFrame())
        fvgs = mtf_data.get('fvgs', [])
        order_blocks = mtf_data.get('order_blocks', [])
        structures = mtf_data.get('structures', [])

        for ote in ote_zones:
            if not ote.is_ote():
                continue

            # Check HTF bias alignment
            if not self._is_htf_bias_aligned(ote, htf_bias):
                continue

            # Check for price in OTE zone
            if not self._is_price_in_ote_zone(df, ote):
                continue

            # Check for OB or FVG confluence
            if not self._has_ote_confluence(ote, mtf_data):
                continue

            # FIXED BUG-ICT-COUNTER-004: Counter-trend OTE logic
            if htf_bias == 'BEARISH':
                signal_type = 'SELL'
                if ote.is_bullish():
                    # Bullish OTE in bearish market = Retracement resistance
                    entry_price = ote.end_price
                else:
                    entry_price = ote.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            elif htf_bias == 'BULLISH':
                signal_type = 'BUY'
                if ote.is_bearish():
                    # Bearish OTE in bullish market = Retracement support
                    entry_price = ote.start_price
                else:
                    entry_price = ote.start_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            else:  # NEUTRAL
                if ote.is_bullish():
                    signal_type = 'BUY'
                    entry_price = ote.start_price
                else:
                    signal_type = 'SELL'
                    entry_price = ote.end_price

                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0

            # FIXED BUG-RR-004: Use minimum_rr_ratio from config (was hardcoded to 3.0)
            min_rr = self.config_loader.get_risk_management_config().minimum_rr_ratio
            
            # Only create signal if RR meets minimum requirement
            if risk_reward >= min_rr:
                signal = Signal(
                    timestamp=ote.timestamp,
                    signal_type=signal_type,
                    entry_type='OTE_ENTRY',
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    confidence=0.9,
                    metadata={
                        'ote_zone': ote,
                        'confirmation_type': 'OB_FVG',
                        'stop_type': 'structure_based',
                        'tp_type': 'structure_based'
                    }
                )

                signals.append(signal)

        return signals

    def detect_breaker_block_entries(self, df: pd.DataFrame, mtf_data: Dict) -> List[Signal]:
        """
        Entry Type 5: Breaker Block Entry

        Previously broken OB becomes BB with price return to BB zone.

        Args:
            df: OHLC DataFrame
            mtf_data: Multi-timeframe analysis data

        Returns:
            List of Breaker Block entry signals
        """
        signals = []

        # Get Breaker Blocks
        breaker_blocks = mtf_data.get('breaker_blocks', [])
        swing_points = mtf_data.get('swing_points', pd.DataFrame())
        fvgs = mtf_data.get('fvgs', [])
        order_blocks = mtf_data.get('order_blocks', [])
        structures = mtf_data.get('structures', [])

        for bb in breaker_blocks:
            if not bb.is_breaker_block():
                continue

            # Check for price return to BB zone
            if not self._is_price_in_bb_zone(df, bb):
                continue

            # Check for structure shift confirmation
            if not self._has_structure_confirmation(bb.timestamp, mtf_data):
                continue

            # Determine entry direction and calculate structure-based stops/TPs
            if bb.is_bullish():
                signal_type = 'BUY'
                entry_price = bb.start_price

                # Structure-based stop loss: below BB with small buffer
                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

                # Structure-based take profits
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            else:  # Bearish BB
                signal_type = 'SELL'
                entry_price = bb.end_price

                # Structure-based stop loss: above BB with small buffer
                stop_loss = self._calculate_structure_based_stop_loss(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

                # Structure-based take profits
                take_profits = self._calculate_structure_based_tps(
                    entry_price, signal_type, swing_points, fvgs, order_blocks, structures
                )

            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price) if take_profits else 0
            risk_reward = reward / risk if risk > 0 else 0

            # FIXED BUG-RR-005: Use minimum_rr_ratio from config (was hardcoded to 3.0)
            min_rr = self.config_loader.get_risk_management_config().minimum_rr_ratio
            
            # Only create signal if RR meets minimum requirement
            if risk_reward >= min_rr:
                signal = Signal(
                    timestamp=bb.timestamp,
                    signal_type=signal_type,
                    entry_type='BREAKER_BLOCK_ENTRY',
                    price=entry_price,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    confidence=0.7,
                    metadata={
                        'breaker_block': bb,
                        'confirmation_type': 'STRUCTURE_SHIFT',
                        'stop_type': 'structure_based',
                        'tp_type': 'structure_based'
                    }
                )

                signals.append(signal)

        return signals

    def _has_choch_confirmation(self, timestamp: datetime, structures: List) -> bool:
        """Check for CHoCH confirmation near timestamp."""
        for structure in structures:
            if abs((structure.timestamp - timestamp).total_seconds()) < 3600:  # Within 1 hour
                if structure.structure_type == 'CHoCH':
                    return True
        return False

    def _has_reversal_structure(self, timestamp: datetime, structures: List) -> bool:
        """Check for reversal structure near timestamp."""
        for structure in structures:
            if abs((structure.timestamp - timestamp).total_seconds()) < 3600:  # Within 1 hour
                if structure.structure_type in ['BOS', 'CHoCH']:
                    return True
        return False

    def _is_htf_bias_aligned(self, concept: ICTConcept, htf_bias: str) -> bool:
        """
        Check if concept is aligned with HTF bias.

        FIXED BUG-ICT-COUNTER-002: Don't filter concepts by type
        All FVGs are potential trading opportunities - we use HTF bias
        to determine signal direction (BUY/SELL) in the entry logic instead.
        """
        # Allow all concepts through - direction is determined in entry logic
        return True

    def _is_price_in_fvg_zone(self, df: pd.DataFrame, fvg: ICTConcept) -> bool:
        """Check if current price is in FVG zone."""
        current_price = df['close'].iloc[-1]
        return fvg.is_price_in_zone(current_price)

    def _is_price_in_ob_zone(self, df: pd.DataFrame, ob: ICTConcept) -> bool:
        """Check if current price is in Order Block zone."""
        current_price = df['close'].iloc[-1]
        return ob.is_price_in_zone(current_price)

    def _is_price_in_ote_zone(self, df: pd.DataFrame, ote: ICTConcept) -> bool:
        """Check if current price is in OTE zone."""
        current_price = df['close'].iloc[-1]
        return ote.is_price_in_zone(current_price)

    def _is_price_in_bb_zone(self, df: pd.DataFrame, bb: ICTConcept) -> bool:
        """Check if current price is in Breaker Block zone."""
        current_price = df['close'].iloc[-1]
        return bb.is_price_in_zone(current_price)

    def _has_structure_confirmation(self, timestamp: datetime, mtf_data: Dict) -> bool:
        """Check for structure confirmation near timestamp."""
        structures = mtf_data.get('structures', [])
        for structure in structures:
            if abs((structure.timestamp - timestamp).total_seconds()) < 3600:  # Within 1 hour
                if structure.structure_type in ['BOS', 'CHoCH']:
                    return True
        return False

    def _has_additional_pattern_confirmation(self, ob: ICTConcept, mtf_data: Dict) -> bool:
        """Check for additional pattern confirmation (FVG/OTE)."""
        fvgs = mtf_data.get('fvgs', [])
        ote_zones = mtf_data.get('ote_zones', [])

        # Check for FVG confluence
        for fvg in fvgs:
            if fvg.is_fvg() and self._is_confluence_with_ob(ob, fvg):
                return True

        # Check for OTE confluence
        for ote in ote_zones:
            if ote.is_ote() and self._is_confluence_with_ob(ob, ote):
                return True

        return False

    def _has_ote_confluence(self, ote: ICTConcept, mtf_data: Dict) -> bool:
        """Check for OTE confluence with OB or FVG."""
        order_blocks = mtf_data.get('order_blocks', [])
        fvgs = mtf_data.get('fvgs', [])

        # Check for OB confluence
        for ob in order_blocks:
            if ob.is_order_block() and self._is_confluence_with_ote(ote, ob):
                return True

        # Check for FVG confluence
        for fvg in fvgs:
            if fvg.is_fvg() and self._is_confluence_with_ote(ote, fvg):
                return True

        return False

    def _is_confluence_with_ob(self, ob: ICTConcept, other: ICTConcept) -> bool:
        """Check if OB has confluence with another concept."""
        # Check if zones overlap
        ob_start, ob_end = ob.start_price, ob.end_price
        other_start, other_end = other.start_price, other.end_price

        return not (ob_end < other_start or ob_start > other_end)

    def _is_confluence_with_ote(self, ote: ICTConcept, other: ICTConcept) -> bool:
        """Check if OTE has confluence with another concept."""
        # Check if zones overlap
        ote_start, ote_end = ote.start_price, ote.end_price
        other_start, other_end = other.start_price, other.end_price

        return not (ote_end < other_start or ote_start > other_end)

    def _calculate_structure_based_tps(self, entry_price: float, signal_type: str,
                                       swing_points: pd.DataFrame, fvgs: List,
                                       order_blocks: List, structures: List) -> List[float]:
        """
        Calculate structure-based take profit levels.

        TP Hierarchy:
        1. TP1: Nearest opposite liquidity level (swing high/low)
        2. TP2: Next FVG or OB in trend direction
        3. TP3: RR milestone (≥1:3)

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            swing_points: DataFrame with swing points
            fvgs: List of FVG concepts
            order_blocks: List of Order Block concepts
            structures: List of MarketStructure objects

        Returns:
            List of take profit levels
        """
        take_profits = []

        # TP1: Find nearest opposite liquidity level
        tp1 = self._find_nearest_opposite_liquidity(entry_price, signal_type, swing_points, structures)
        if tp1:
            take_profits.append(tp1)

        # TP2: Find next FVG or OB in trend direction
        tp2 = self._find_next_fvg_or_ob_in_direction(entry_price, signal_type, fvgs, order_blocks)
        if tp2:
            take_profits.append(tp2)

        # TP3: RR milestone (≥1:3)
        if take_profits:
            # Calculate risk from first TP
            risk = abs(take_profits[0] - entry_price)
            if signal_type == 'BUY':
                tp3 = entry_price + (risk * 3)  # 1:3 RR
            else:  # SELL
                tp3 = entry_price - (risk * 3)  # 1:3 RR
            take_profits.append(tp3)

        # Ensure we have at least one TP
        if not take_profits:
            # Fallback to conservative RR target
            if signal_type == 'BUY':
                conservative_tp = entry_price * 1.02
            else:
                conservative_tp = entry_price * 0.98
            take_profits.append(conservative_tp)

        return take_profits

    def _find_nearest_opposite_liquidity(self, entry_price: float, signal_type: str,
                                        swing_points: pd.DataFrame, structures: List) -> Optional[float]:
        """
        Find the nearest opposite liquidity level (swing high for sells, swing low for buys).
        FIXED: Added minimum TP distance requirement to ensure adequate reward.

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            swing_points: DataFrame with swing points
            structures: List of MarketStructure objects

        Returns:
            Nearest opposite liquidity price or None
        """
        # FIXED BUG-TP-001: Enforce minimum take profit distance
        MIN_TP_DISTANCE_PERCENT = 0.025  # Minimum 2.5% profit per trade
        
        if signal_type == 'BUY':
            # Look for swing highs above entry
            swing_highs = swing_points[swing_points['swing_high']]
            if not swing_highs.empty:
                # Find swing highs above entry price
                highs_above = swing_highs[swing_highs['swing_high_price'] > entry_price]
                if not highs_above.empty:
                    nearest_high = highs_above['swing_high_price'].min()
                    
                    # FIXED: Check if TP distance is adequate
                    tp_distance_pct = (nearest_high - entry_price) / entry_price
                    
                    if tp_distance_pct < MIN_TP_DISTANCE_PERCENT:
                        # TP is too close - look for next swing high or use minimum distance
                        # Find next swing high beyond minimum distance
                        adequate_highs = highs_above[
                            (highs_above['swing_high_price'] - entry_price) / entry_price >= MIN_TP_DISTANCE_PERCENT
                        ]
                        
                        if not adequate_highs.empty:
                            return adequate_highs['swing_high_price'].min()
                        else:
                            # No adequate swing high found - use minimum distance
                            return entry_price * (1.0 + MIN_TP_DISTANCE_PERCENT)
                    else:
                        return nearest_high

            # Fallback: look in structures for HH/LH
            for structure in structures:
                if structure.structure_type in ['HH', 'LH'] and structure.price > entry_price:
                    tp_distance_pct = (structure.price - entry_price) / entry_price
                    
                    if tp_distance_pct >= MIN_TP_DISTANCE_PERCENT:
                        return structure.price
            
            # If no adequate structure found, use minimum distance
            return entry_price * (1.0 + MIN_TP_DISTANCE_PERCENT)

        else:  # SELL
            # Look for swing lows below entry
            swing_lows = swing_points[swing_points['swing_low']]
            if not swing_lows.empty:
                # Find swing lows below entry price
                lows_below = swing_lows[swing_lows['swing_low_price'] < entry_price]
                if not lows_below.empty:
                    nearest_low = lows_below['swing_low_price'].max()
                    
                    # FIXED: Check if TP distance is adequate
                    tp_distance_pct = (entry_price - nearest_low) / entry_price
                    
                    if tp_distance_pct < MIN_TP_DISTANCE_PERCENT:
                        # TP is too close - look for next swing low or use minimum distance
                        # Find next swing low beyond minimum distance
                        adequate_lows = lows_below[
                            (entry_price - lows_below['swing_low_price']) / entry_price >= MIN_TP_DISTANCE_PERCENT
                        ]
                        
                        if not adequate_lows.empty:
                            return adequate_lows['swing_low_price'].max()
                        else:
                            # No adequate swing low found - use minimum distance
                            return entry_price * (1.0 - MIN_TP_DISTANCE_PERCENT)
                    else:
                        return nearest_low

            # Fallback: look in structures for LL/HL
            for structure in structures:
                if structure.structure_type in ['LL', 'HL'] and structure.price < entry_price:
                    tp_distance_pct = (entry_price - structure.price) / entry_price
                    
                    if tp_distance_pct >= MIN_TP_DISTANCE_PERCENT:
                        return structure.price
            
            # If no adequate structure found, use minimum distance
            return entry_price * (1.0 - MIN_TP_DISTANCE_PERCENT)

        return None

    def _find_next_fvg_or_ob_in_direction(self, entry_price: float, signal_type: str,
                                         fvgs: List, order_blocks: List) -> Optional[float]:
        """
        Find the next FVG or OB in trend direction.

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            fvgs: List of FVG concepts
            order_blocks: List of Order Block concepts

        Returns:
            Next FVG/OB price in direction or None
        """
        if signal_type == 'BUY':
            # Look for bullish FVGs above entry
            bullish_fvgs = [fvg for fvg in fvgs if fvg.is_bullish() and fvg.end_price > entry_price]
            if bullish_fvgs:
                return min(fvg.end_price for fvg in bullish_fvgs)

            # Look for bullish OBs above entry
            bullish_obs = [ob for ob in order_blocks if ob.is_bullish() and ob.end_price > entry_price]
            if bullish_obs:
                return min(ob.end_price for ob in bullish_obs)

        else:  # SELL
            # Look for bearish FVGs below entry
            bearish_fvgs = [fvg for fvg in fvgs if fvg.is_bearish() and fvg.start_price < entry_price]
            if bearish_fvgs:
                return max(fvg.start_price for fvg in bearish_fvgs)

            # Look for bearish OBs below entry
            bearish_obs = [ob for ob in order_blocks if ob.is_bearish() and ob.start_price < entry_price]
            if bearish_obs:
                return max(ob.start_price for ob in bearish_obs)

        return None

    def _calculate_structure_based_stop_loss(self, entry_price: float, signal_type: str,
                                           swing_points: pd.DataFrame, fvgs: List,
                                           order_blocks: List, structures: List) -> float:
        """
        Calculate structure-based stop loss levels.

        Stop Loss Hierarchy:
        1. SL1: Nearest opposite structure level (swing high/low)
        2. SL2: Next FVG or OB in opposite direction
        3. SL3: ATR-based stop with small buffer

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            swing_points: DataFrame with swing points
            fvgs: List of FVG concepts
            order_blocks: List of Order Block concepts
            structures: List of MarketStructure objects

        Returns:
            Structure-based stop loss level
        """
        # SL1: Find nearest opposite structure level
        sl1 = self._find_nearest_opposite_structure_level(entry_price, signal_type, swing_points, structures)
        if sl1:
            return sl1

        # SL2: Find next FVG or OB in opposite direction
        sl2 = self._find_next_fvg_or_ob_opposite_direction(entry_price, signal_type, fvgs, order_blocks)
        if sl2:
            return sl2

        # SL3: ATR-based stop with small buffer (fallback)
        atr_buffer = 0.005  # 0.5% buffer
        if signal_type == 'BUY':
            return entry_price * (1.0 - atr_buffer)
        else:  # SELL
            return entry_price * (1.0 + atr_buffer)

    def _find_nearest_opposite_structure_level(self, entry_price: float, signal_type: str,
                                             swing_points: pd.DataFrame, structures: List) -> Optional[float]:
        """
        Find the nearest opposite structure level for stop loss.
        FIXED: Added buffer beyond swing point and maximum distance limit.

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            swing_points: DataFrame with swing points
            structures: List of MarketStructure objects

        Returns:
            Nearest opposite structure level or None
        """
        # FIXED BUG-SL-001: Add buffer beyond swing point to avoid wick stops
        SL_BUFFER_PERCENT = 0.003  # 0.3% buffer beyond swing point
        
        # FIXED BUG-SL-002: Enforce maximum stop loss distance
        MAX_SL_DISTANCE_PERCENT = 0.015  # Maximum 1.5% risk per trade
        
        if signal_type == 'BUY':
            # Look for swing lows below entry
            swing_lows = swing_points[swing_points['swing_low']]
            if not swing_lows.empty:
                # Find swing lows below entry price
                lows_below = swing_lows[swing_lows['swing_low_price'] < entry_price]
                if not lows_below.empty:
                    nearest_low = lows_below['swing_low_price'].max()
                    
                    # FIXED: Check if stop distance is reasonable
                    stop_distance_pct = (entry_price - nearest_low) / entry_price
                    
                    if stop_distance_pct > MAX_SL_DISTANCE_PERCENT:
                        # Stop is too far - use maximum distance instead
                        return entry_price * (1.0 - MAX_SL_DISTANCE_PERCENT)
                    else:
                        # FIXED: Add buffer beyond swing low to avoid wick stops
                        return nearest_low * (1.0 - SL_BUFFER_PERCENT)

            # Fallback: look in structures for LL/HL
            for structure in structures:
                if structure.structure_type in ['LL', 'HL'] and structure.price < entry_price:
                    stop_distance_pct = (entry_price - structure.price) / entry_price
                    
                    if stop_distance_pct > MAX_SL_DISTANCE_PERCENT:
                        return entry_price * (1.0 - MAX_SL_DISTANCE_PERCENT)
                    else:
                        return structure.price * (1.0 - SL_BUFFER_PERCENT)

        else:  # SELL
            # Look for swing highs above entry
            swing_highs = swing_points[swing_points['swing_high']]
            if not swing_highs.empty:
                # Find swing highs above entry price
                highs_above = swing_highs[swing_highs['swing_high_price'] > entry_price]
                if not highs_above.empty:
                    nearest_high = highs_above['swing_high_price'].min()
                    
                    # FIXED: Check if stop distance is reasonable
                    stop_distance_pct = (nearest_high - entry_price) / entry_price
                    
                    if stop_distance_pct > MAX_SL_DISTANCE_PERCENT:
                        # Stop is too far - use maximum distance instead
                        return entry_price * (1.0 + MAX_SL_DISTANCE_PERCENT)
                    else:
                        # FIXED: Add buffer beyond swing high to avoid wick stops
                        return nearest_high * (1.0 + SL_BUFFER_PERCENT)

            # Fallback: look in structures for HH/LH
            for structure in structures:
                if structure.structure_type in ['HH', 'LH'] and structure.price > entry_price:
                    stop_distance_pct = (structure.price - entry_price) / entry_price
                    
                    if stop_distance_pct > MAX_SL_DISTANCE_PERCENT:
                        return entry_price * (1.0 + MAX_SL_DISTANCE_PERCENT)
                    else:
                        return structure.price * (1.0 + SL_BUFFER_PERCENT)

        return None

    def _find_next_fvg_or_ob_opposite_direction(self, entry_price: float, signal_type: str,
                                              fvgs: List, order_blocks: List) -> Optional[float]:
        """
        Find the next FVG or OB in opposite direction for stop loss.

        Args:
            entry_price: Entry price
            signal_type: 'BUY' or 'SELL'
            fvgs: List of FVG concepts
            order_blocks: List of Order Block concepts

        Returns:
            Next FVG/OB price in opposite direction or None
        """
        if signal_type == 'BUY':
            # Look for bearish FVGs below entry
            bearish_fvgs = [fvg for fvg in fvgs if fvg.is_bearish() and fvg.start_price < entry_price]
            if bearish_fvgs:
                return max(fvg.start_price for fvg in bearish_fvgs)

            # Look for bearish OBs below entry
            bearish_obs = [ob for ob in order_blocks if ob.is_bearish() and ob.start_price < entry_price]
            if bearish_obs:
                return max(ob.start_price for ob in bearish_obs)

        else:  # SELL
            # Look for bullish FVGs above entry
            bullish_fvgs = [fvg for fvg in fvgs if fvg.is_bullish() and fvg.end_price > entry_price]
            if bullish_fvgs:
                return min(fvg.end_price for fvg in bullish_fvgs)

            # Look for bullish OBs above entry
            bullish_obs = [ob for ob in order_blocks if ob.is_bullish() and ob.end_price > entry_price]
            if bullish_obs:
                return min(ob.end_price for ob in bullish_obs)

        return None
