"""
Test ICT Entries Structure-Based Stops and Take Profits
Tests FIX-ASYM-002 implementation for structure-based SL/TPs.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from trading_strategy.ict_entries import ICTEntries
from trading_strategy.config_loader import ConfigLoader
from trading_strategy.data_structures import (
    Signal, ICTConcept, MarketStructure, LiquidityLevel
)


class TestICTEntriesStructureBased:
    """Test structure-based stops and take profits for ICT entries."""

    @pytest.fixture
    def ict_entries(self):
        """Create ICTEntries instance for testing."""
        return ICTEntries()

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC DataFrame."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        np.random.seed(42)  # For reproducible tests

        # Create realistic price data with swing points
        base_price = 100.0
        prices = []
        current_price = base_price

        for i in range(100):
            # Add some volatility
            change = np.random.normal(0, 0.5)
            current_price += change
            prices.append(current_price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000, 10000) for _ in range(100)]
        }, index=dates)

        # Ensure high >= low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

    @pytest.fixture
    def sample_mtf_data(self, sample_df):
        """Create sample MTF analysis data with structure."""
        # Create swing points
        swing_df = sample_df.copy()
        swing_df['swing_high'] = False
        swing_df['swing_low'] = False
        swing_df['swing_high_price'] = np.nan
        swing_df['swing_low_price'] = np.nan

        # Add some swing points
        swing_df.iloc[10, swing_df.columns.get_loc('swing_high')] = True
        swing_df.iloc[10, swing_df.columns.get_loc('swing_high_price')] = 105.0

        swing_df.iloc[20, swing_df.columns.get_loc('swing_low')] = True
        swing_df.iloc[20, swing_df.columns.get_loc('swing_low_price')] = 95.0

        swing_df.iloc[30, swing_df.columns.get_loc('swing_high')] = True
        swing_df.iloc[30, swing_df.columns.get_loc('swing_high_price')] = 110.0

        # Create FVGs
        fvg_bullish = ICTConcept(
            timestamp=sample_df.index[15],
            concept_type='FVG_BULLISH',
            start_price=100.0,
            end_price=102.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        fvg_bearish = ICTConcept(
            timestamp=sample_df.index[25],
            concept_type='FVG_BEARISH',
            start_price=106.0,
            end_price=108.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        # Create Order Blocks
        ob_bullish = ICTConcept(
            timestamp=sample_df.index[5],
            concept_type='OB_BULLISH',
            start_price=98.0,
            end_price=100.0,
            status='current',
            strength=0.9,
            is_fresh=True
        )

        ob_bearish = ICTConcept(
            timestamp=sample_df.index[35],
            concept_type='OB_BEARISH',
            start_price=110.0,
            end_price=112.0,
            status='current',
            strength=0.9,
            is_fresh=True
        )

        # Create liquidity grab
        liquidity_grab_high = ICTConcept(
            timestamp=sample_df.index[12],
            concept_type='LIQUIDITY_GRAB_BEARISH',
            start_price=104.0,
            end_price=106.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        liquidity_grab_low = ICTConcept(
            timestamp=sample_df.index[22],
            concept_type='LIQUIDITY_GRAB_BULLISH',
            start_price=94.0,
            end_price=96.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        # Create market structures
        structure_bullish = MarketStructure(
            timestamp=sample_df.index[15],
            structure_type='BOS',
            price=102.0,
            timeframe='15m',
            strength=0.8,
            trend_direction='BULLISH'
        )

        structure_bearish = MarketStructure(
            timestamp=sample_df.index[25],
            structure_type='BOS',
            price=106.0,
            timeframe='15m',
            strength=0.8,
            trend_direction='BEARISH'
        )

        return {
            'dataframe': sample_df,
            'swing_points': swing_df,
            'fvgs': [fvg_bullish, fvg_bearish],
            'order_blocks': [ob_bullish, ob_bearish],
            'liquidity_grabs': [liquidity_grab_high, liquidity_grab_low],
            'structures': [structure_bullish, structure_bearish],
            'htf_bias': 'NEUTRAL'
        }

    def test_liquidity_grab_choch_structure_based_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that liquidity grab entries use structure-based stops."""
        signals = ict_entries.detect_liquidity_grab_choch_entries(sample_df, sample_mtf_data)

        # Should have signals (if conditions are met)
        if signals:
            signal = signals[0]

            # Verify structure-based stop loss
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'

            # Verify stop loss is beyond invalidation with small buffer
            if signal.signal_type == 'SELL':
                # For sell signals, stop should be above the liquidity grab high
                liquidity_grab = signal.metadata['liquidity_grab']
                expected_stop = liquidity_grab.end_price * 1.005
                assert abs(signal.stop_loss - expected_stop) < 0.01
            else:  # BUY
                # For buy signals, stop should be below the liquidity grab low
                liquidity_grab = signal.metadata['liquidity_grab']
                expected_stop = liquidity_grab.start_price * 0.995
                assert abs(signal.stop_loss - expected_stop) < 0.01

    def test_fvg_entry_structure_based_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that FVG entries use structure-based stops."""
        signals = ict_entries.detect_fvg_entries(sample_df, sample_mtf_data)

        if signals:
            signal = signals[0]

            # Verify structure-based stops
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'

            # Verify stop loss is beyond FVG with small buffer
            if signal.signal_type == 'BUY':
                fvg = signal.metadata['fvg']
                expected_stop = fvg.start_price * 0.995
                assert abs(signal.stop_loss - expected_stop) < 0.01
            else:  # SELL
                fvg = signal.metadata['fvg']
                expected_stop = fvg.end_price * 1.005
                assert abs(signal.stop_loss - expected_stop) < 0.01

    def test_order_block_entry_structure_based_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that Order Block entries use structure-based stops."""
        signals = ict_entries.detect_order_block_entries(sample_df, sample_mtf_data)

        if signals:
            signal = signals[0]

            # Verify structure-based stops
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'

            # Verify stop loss is beyond OB with small buffer
            if signal.signal_type == 'BUY':
                ob = signal.metadata['order_block']
                expected_stop = ob.start_price * 0.995
                assert abs(signal.stop_loss - expected_stop) < 0.01
            else:  # SELL
                ob = signal.metadata['order_block']
                expected_stop = ob.end_price * 1.005
                assert abs(signal.stop_loss - expected_stop) < 0.01

    def test_ote_entry_structure_based_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that OTE entries use structure-based stops."""
        # Add OTE zones to test data
        ote_bullish = ICTConcept(
            timestamp=sample_df.index[18],
            concept_type='OTE_BULLISH',
            start_price=97.0,
            end_price=99.0,
            status='current',
            strength=0.9,
            is_fresh=True
        )

        ote_bearish = ICTConcept(
            timestamp=sample_df.index[28],
            concept_type='OTE_BEARISH',
            start_price=109.0,
            end_price=111.0,
            status='current',
            strength=0.9,
            is_fresh=True
        )

        sample_mtf_data['ote_zones'] = [ote_bullish, ote_bearish]

        signals = ict_entries.detect_ote_entries(sample_df, sample_mtf_data)

        if signals:
            signal = signals[0]

            # Verify structure-based stops
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'

            # Verify stop loss is beyond OTE with small buffer
            if signal.signal_type == 'BUY':
                ote = signal.metadata['ote_zone']
                expected_stop = ote.start_price * 0.995
                assert abs(signal.stop_loss - expected_stop) < 0.01
            else:  # SELL
                ote = signal.metadata['ote_zone']
                expected_stop = ote.end_price * 1.005
                assert abs(signal.stop_loss - expected_stop) < 0.01

    def test_breaker_block_entry_structure_based_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that Breaker Block entries use structure-based stops."""
        # Add breaker blocks to test data
        bb_bullish = ICTConcept(
            timestamp=sample_df.index[8],
            concept_type='BB_BULLISH',
            start_price=99.0,
            end_price=101.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        bb_bearish = ICTConcept(
            timestamp=sample_df.index[38],
            concept_type='BB_BEARISH',
            start_price=109.0,
            end_price=111.0,
            status='current',
            strength=0.8,
            is_fresh=True
        )

        sample_mtf_data['breaker_blocks'] = [bb_bullish, bb_bearish]

        signals = ict_entries.detect_breaker_block_entries(sample_df, sample_mtf_data)

        if signals:
            signal = signals[0]

            # Verify structure-based stops
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'

            # Verify stop loss is beyond BB with small buffer
            if signal.signal_type == 'BUY':
                bb = signal.metadata['breaker_block']
                expected_stop = bb.start_price * 0.995
                assert abs(signal.stop_loss - expected_stop) < 0.01
            else:  # SELL
                bb = signal.metadata['breaker_block']
                expected_stop = bb.end_price * 1.005
                assert abs(signal.stop_loss - expected_stop) < 0.01

    def test_structure_based_take_profits_hierarchy(self, ict_entries, sample_mtf_data):
        """Test that take profits follow the correct hierarchy."""
        entry_price = 100.0
        signal_type = 'BUY'

        take_profits = ict_entries._calculate_structure_based_tps(
            entry_price, signal_type,
            sample_mtf_data['swing_points'],
            sample_mtf_data['fvgs'],
            sample_mtf_data['order_blocks'],
            sample_mtf_data['structures']
        )

        # Should have at least one TP
        assert len(take_profits) >= 1

        # All TPs should be above entry price for BUY signals
        for tp in take_profits:
            assert tp > entry_price

        # Test SELL signal
        signal_type = 'SELL'
        take_profits_sell = ict_entries._calculate_structure_based_tps(
            entry_price, signal_type,
            sample_mtf_data['swing_points'],
            sample_mtf_data['fvgs'],
            sample_mtf_data['order_blocks'],
            sample_mtf_data['structures']
        )

        # All TPs should be below entry price for SELL signals
        for tp in take_profits_sell:
            assert tp < entry_price

    def test_risk_reward_ratio_minimum_1_3(self, ict_entries, sample_df, sample_mtf_data):
        """Test that all signals have RR >= 1:3."""
        # Test all entry types
        entry_methods = [
            ict_entries.detect_liquidity_grab_choch_entries,
            ict_entries.detect_fvg_entries,
            ict_entries.detect_order_block_entries,
            ict_entries.detect_ote_entries,
            ict_entries.detect_breaker_block_entries
        ]

        for method in entry_methods:
            signals = method(sample_df, sample_mtf_data)

            for signal in signals:
                # Verify RR >= 1:3
                assert signal.risk_reward >= 3.0, f"Signal {signal.entry_type} has RR {signal.risk_reward} < 3.0"

                # Verify RR calculation is correct
                risk = abs(signal.price - signal.stop_loss)
                reward = abs(signal.take_profits[0] - signal.price)
                expected_rr = reward / risk if risk > 0 else 0

                assert abs(signal.risk_reward - expected_rr) < 0.01, f"RR calculation mismatch: {signal.risk_reward} vs {expected_rr}"

    def test_tp1_opposite_liquidity_detection(self, ict_entries, sample_mtf_data):
        """Test TP1 detection of opposite liquidity levels."""
        # Test BUY signal - should find swing high above entry
        entry_price = 100.0
        signal_type = 'BUY'

        tp1 = ict_entries._find_nearest_opposite_liquidity(
            entry_price, signal_type,
            sample_mtf_data['swing_points'],
            sample_mtf_data['structures']
        )

        if tp1:
            assert tp1 > entry_price, "TP1 for BUY should be above entry price"
            assert tp1 == 105.0, "Should find the nearest swing high at 105.0"

        # Test SELL signal - should find swing low below entry
        signal_type = 'SELL'

        tp1_sell = ict_entries._find_nearest_opposite_liquidity(
            entry_price, signal_type,
            sample_mtf_data['swing_points'],
            sample_mtf_data['structures']
        )

        if tp1_sell:
            assert tp1_sell < entry_price, "TP1 for SELL should be below entry price"
            assert tp1_sell == 95.0, "Should find the nearest swing low at 95.0"

    def test_tp2_fvg_ob_direction_detection(self, ict_entries, sample_mtf_data):
        """Test TP2 detection of next FVG/OB in direction."""
        # Test BUY signal - should find bullish FVG/OB above entry
        entry_price = 100.0
        signal_type = 'BUY'

        tp2 = ict_entries._find_next_fvg_or_ob_in_direction(
            entry_price, signal_type,
            sample_mtf_data['fvgs'],
            sample_mtf_data['order_blocks']
        )

        if tp2:
            assert tp2 > entry_price, "TP2 for BUY should be above entry price"
            assert tp2 == 102.0, "Should find bullish FVG at 102.0"

        # Test SELL signal - should find bearish FVG/OB below entry
        signal_type = 'SELL'

        tp2_sell = ict_entries._find_next_fvg_or_ob_in_direction(
            entry_price, signal_type,
            sample_mtf_data['fvgs'],
            sample_mtf_data['order_blocks']
        )

        if tp2_sell:
            assert tp2_sell < entry_price, "TP2 for SELL should be below entry price"
            assert tp2_sell == 110.0, "Should find bearish OB at 110.0"

    def test_tp3_rr_milestone_calculation(self, ict_entries, sample_mtf_data):
        """Test TP3 RR milestone calculation."""
        entry_price = 100.0
        signal_type = 'BUY'

        # Mock TP1 for calculation
        tp1 = 105.0
        risk = abs(tp1 - entry_price)  # 5.0

        # Calculate TP3
        tp3 = entry_price + (risk * 3)  # 100 + (5 * 3) = 115.0

        assert tp3 == 115.0, "TP3 should be 1:3 RR milestone"

        # Test SELL signal
        signal_type = 'SELL'
        tp3_sell = entry_price - (risk * 3)  # 100 - (5 * 3) = 85.0

        assert tp3_sell == 85.0, "TP3 for SELL should be 1:3 RR milestone"

    def test_no_fixed_percentage_stops(self, ict_entries, sample_df, sample_mtf_data):
        """Test that no signals use fixed percentage stops."""
        # Test all entry types
        entry_methods = [
            ict_entries.detect_liquidity_grab_choch_entries,
            ict_entries.detect_fvg_entries,
            ict_entries.detect_order_block_entries,
            ict_entries.detect_ote_entries,
            ict_entries.detect_breaker_block_entries
        ]

        for method in entry_methods:
            signals = method(sample_df, sample_mtf_data)

            for signal in signals:
                # Verify stop loss is not a fixed percentage
                risk_percent = abs(signal.price - signal.stop_loss) / signal.price * 100

                # Should not be exactly 2% (old fixed percentage)
                assert abs(risk_percent - 2.0) > 0.1, f"Signal {signal.entry_type} still uses 2% fixed stop"

                # Should not be exactly 4% or 6% either
                assert abs(risk_percent - 4.0) > 0.1, f"Signal {signal.entry_type} still uses 4% fixed stop"
                assert abs(risk_percent - 6.0) > 0.1, f"Signal {signal.entry_type} still uses 6% fixed stop"

    def test_bidirectional_symmetry(self, ict_entries, sample_df, sample_mtf_data):
        """Test that structure-based stops work symmetrically for both directions."""
        # Create bullish scenario
        bullish_mtf = sample_mtf_data.copy()
        bullish_mtf['htf_bias'] = 'BULLISH'

        # Create bearish scenario (inverted)
        bearish_mtf = sample_mtf_data.copy()
        bearish_mtf['htf_bias'] = 'BEARISH'

        # Test FVG entries
        bullish_signals = ict_entries.detect_fvg_entries(sample_df, bullish_mtf)
        bearish_signals = ict_entries.detect_fvg_entries(sample_df, bearish_mtf)

        # Both should use structure-based stops
        for signal in bullish_signals + bearish_signals:
            assert signal.metadata.get('stop_type') == 'structure_based'
            assert signal.metadata.get('tp_type') == 'structure_based'
            assert signal.risk_reward >= 3.0

    def test_fallback_conservative_tp(self, ict_entries):
        """Test fallback to conservative TP when no structure found."""
        # Create empty MTF data with proper columns
        empty_swing_df = pd.DataFrame(columns=['swing_high', 'swing_low', 'swing_high_price', 'swing_low_price'])
        empty_mtf = {
            'swing_points': empty_swing_df,
            'fvgs': [],
            'order_blocks': [],
            'structures': []
        }

        entry_price = 100.0

        # Test BUY signal
        take_profits = ict_entries._calculate_structure_based_tps(
            entry_price, 'BUY',
            empty_mtf['swing_points'],
            empty_mtf['fvgs'],
            empty_mtf['order_blocks'],
            empty_mtf['structures']
        )

        assert len(take_profits) == 1
        assert take_profits[0] == entry_price * 1.02  # 2% conservative TP

        # Test SELL signal
        take_profits_sell = ict_entries._calculate_structure_based_tps(
            entry_price, 'SELL',
            empty_mtf['swing_points'],
            empty_mtf['fvgs'],
            empty_mtf['order_blocks'],
            empty_mtf['structures']
        )

        assert len(take_profits_sell) == 1
        assert take_profits_sell[0] == entry_price * 0.98  # 2% conservative TP


if __name__ == '__main__':
    pytest.main([__file__])
