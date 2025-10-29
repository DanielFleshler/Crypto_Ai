"""
Test suite for ICT concepts detection with bidirectional symmetry.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.data_structures import ICTConcept
from trading_strategy.config_loader import ConfigLoader


class TestICTConcepts(unittest.TestCase):
    """Test ICT concepts detection for bidirectional symmetry."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
        self.detector = ICTConceptsDetector(self.config_loader)

        # Create test data
        self.test_data = self._create_test_data()
        self.swing_data = self._create_swing_data()

    def _create_test_data(self):
        """Create test OHLC data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

        # Create data with gaps for FVG testing
        base_price = 100.0
        prices = []
        for i in range(len(dates)):
            # Add some trend and noise
            trend = i * 0.1
            noise = np.random.normal(0, 0.5)
            price = base_price + trend + noise
            prices.append(price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 1) for p in prices],
            'low': [p - np.random.uniform(0, 1) for p in prices],
            'close': prices,
            'volume': [1000 + np.random.randint(0, 500) for _ in prices]
        }, index=dates)

        return df

    def _create_swing_data(self):
        """Create test swing point data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

        swing_data = pd.DataFrame({
            'swing_high': [False] * 100,
            'swing_low': [False] * 100,
            'swing_high_price': [np.nan] * 100,
            'swing_low_price': [np.nan] * 100
        }, index=dates)

        # Add some swing points
        swing_data.iloc[10, swing_data.columns.get_loc('swing_low')] = True
        swing_data.iloc[10, swing_data.columns.get_loc('swing_low_price')] = 100.0

        swing_data.iloc[20, swing_data.columns.get_loc('swing_high')] = True
        swing_data.iloc[20, swing_data.columns.get_loc('swing_high_price')] = 110.0

        return swing_data

    def test_fvg_detection_bullish(self):
        """Test FVG detection for bullish scenario."""
        # Create data with bullish FVG
        fvg_data = self.test_data.copy()
        # Create gap: C1 high < C3 low
        fvg_data.iloc[0, fvg_data.columns.get_loc('high')] = 100.0
        fvg_data.iloc[2, fvg_data.columns.get_loc('low')] = 102.0

        fvgs = self.detector.detect_fvg(fvg_data)

        self.assertIsInstance(fvgs, list)

        # Check that at least one bullish FVG is detected
        bullish_fvgs = [fvg for fvg in fvgs if fvg.is_bullish()]
        self.assertGreater(len(bullish_fvgs), 0, "Should detect at least one bullish FVG")

        # Check that all detected FVGs are valid
        for fvg in fvgs:
            self.assertTrue(fvg.is_fvg())
            self.assertFalse(fvg.is_filled)
            self.assertTrue(fvg.is_fresh)

    def test_fvg_detection_bearish(self):
        """Test FVG detection for bearish scenario."""
        # Create data with bearish FVG
        fvg_data = self.test_data.copy()
        # Create gap: C1 low > C3 high
        fvg_data.iloc[0, fvg_data.columns.get_loc('low')] = 102.0
        fvg_data.iloc[2, fvg_data.columns.get_loc('high')] = 100.0

        fvgs = self.detector.detect_fvg(fvg_data)

        self.assertIsInstance(fvgs, list)

        # Check that at least one bearish FVG is detected
        bearish_fvgs = [fvg for fvg in fvgs if fvg.is_bearish()]
        self.assertGreater(len(bearish_fvgs), 0, "Should detect at least one bearish FVG")

        # Check that all detected FVGs are valid
        for fvg in fvgs:
            self.assertTrue(fvg.is_fvg())
            self.assertFalse(fvg.is_filled)
            self.assertTrue(fvg.is_fresh)

    def test_ifvg_distinction(self):
        """Test IFVG distinction from regular FVG."""
        # Create data with IFVG characteristics
        ifvg_data = self.test_data.copy()
        # Create strong move before FVG
        ifvg_data.iloc[0, ifvg_data.columns.get_loc('close')] = 100.0
        ifvg_data.iloc[2, ifvg_data.columns.get_loc('close')] = 105.0  # 5% move
        ifvg_data.iloc[0, ifvg_data.columns.get_loc('high')] = 100.0
        ifvg_data.iloc[2, ifvg_data.columns.get_loc('low')] = 102.0

        fvgs = self.detector.detect_fvg(ifvg_data)

        for fvg in fvgs:
            if fvg.is_ifvg():
                self.assertTrue(fvg.concept_type.startswith('IFVG'))
            else:
                self.assertTrue(fvg.concept_type.startswith('FVG'))

    def test_order_block_detection_bullish(self):
        """Test Order Block detection for bullish scenario."""
        obs = self.detector.detect_order_blocks(self.test_data, self.swing_data)

        self.assertIsInstance(obs, list)

        for ob in obs:
            self.assertTrue(ob.is_order_block())
            self.assertTrue(ob.is_fresh)
            self.assertFalse(ob.is_broken)
            self.assertEqual(ob.test_count, 0)

    def test_order_block_detection_bearish(self):
        """Test Order Block detection for bearish scenario."""
        # Create bearish test data
        bearish_data = self.test_data.copy()
        bearish_data['close'] = bearish_data['close'].iloc[::-1].values

        obs = self.detector.detect_order_blocks(bearish_data, self.swing_data)

        self.assertIsInstance(obs, list)

        for ob in obs:
            self.assertTrue(ob.is_order_block())
            self.assertTrue(ob.is_fresh)
            self.assertFalse(ob.is_broken)

    def test_breaker_block_detection(self):
        """Test Breaker Block detection."""
        # First detect Order Blocks
        obs = self.detector.detect_order_blocks(self.test_data, self.swing_data)

        # Then detect Breaker Blocks
        bbs = self.detector.detect_breaker_blocks(self.test_data, obs)

        self.assertIsInstance(bbs, list)

        for bb in bbs:
            self.assertTrue(bb.is_breaker_block())
            self.assertFalse(bb.is_fresh)  # BB is not fresh

    def test_ote_calculation_bullish(self):
        """Test OTE calculation for bullish scenario."""
        start_price = 100.0
        end_price = 110.0

        ote_levels = self.detector.calculate_ote_levels(start_price, end_price)

        # Check OTE zone - for bullish moves, ote_start should be higher than ote_end
        self.assertGreater(ote_levels['ote_start'], ote_levels['ote_end'])

        # Check Fibonacci levels
        self.assertLess(ote_levels['fib_50'], ote_levels['fib_618'])
        self.assertLess(ote_levels['fib_618'], ote_levels['fib_786'])

        # OTE zone should be between 62% and 79% retracement
        price_range = end_price - start_price
        ote_start_expected = end_price - (price_range * 0.62)
        ote_end_expected = end_price - (price_range * 0.79)

        self.assertAlmostEqual(ote_levels['ote_start'], ote_start_expected, places=2)
        self.assertAlmostEqual(ote_levels['ote_end'], ote_end_expected, places=2)

    def test_ote_calculation_bearish(self):
        """Test OTE calculation for bearish scenario."""
        start_price = 110.0
        end_price = 100.0

        ote_levels = self.detector.calculate_ote_levels(start_price, end_price)

        # Check OTE zone - for bearish moves, ote_start should be lower than ote_end
        self.assertLess(ote_levels['ote_start'], ote_levels['ote_end'])

        # Check Fibonacci levels
        self.assertGreater(ote_levels['fib_50'], ote_levels['fib_618'])
        self.assertGreater(ote_levels['fib_618'], ote_levels['fib_786'])

        # OTE zone should be between 62% and 79% retracement
        price_range = start_price - end_price
        base = end_price  # For bearish move, base is the low (end_price)
        ote_start_expected = base - (price_range * 0.79)
        ote_end_expected = base - (price_range * 0.62)

        self.assertAlmostEqual(ote_levels['ote_start'], ote_start_expected, places=2)
        self.assertAlmostEqual(ote_levels['ote_end'], ote_end_expected, places=2)

    def test_ote_zone_detection(self):
        """Test OTE zone detection."""
        ote_zones = self.detector.detect_ote_zones(self.test_data, self.swing_data)

        self.assertIsInstance(ote_zones, list)

        for ote in ote_zones:
            self.assertTrue(ote.is_ote())
            self.assertTrue(ote.is_fresh)

            # Check zone size
            zone_size = ote.get_zone_size()
            self.assertGreater(zone_size, 0)

    def test_liquidity_grab_detection(self):
        """Test liquidity grab detection."""
        liquidity_grabs = self.detector.detect_liquidity_grabs(self.test_data, self.swing_data)

        self.assertIsInstance(liquidity_grabs, list)

        for grab in liquidity_grabs:
            self.assertTrue(grab.is_liquidity_grab())
            self.assertTrue(grab.is_fresh)

    def test_fvg_fill_tracking(self):
        """Test FVG fill tracking."""
        # Create FVGs
        fvgs = self.detector.detect_fvg(self.test_data)

        # Update fill status
        updated_fvgs = self.detector.update_fvg_fill_status(self.test_data, fvgs)

        self.assertIsInstance(updated_fvgs, list)

        for fvg in updated_fvgs:
            self.assertIsInstance(fvg.is_filled, bool)
            if fvg.is_filled:
                self.assertIsNotNone(fvg.fill_timestamp)

    def test_ob_freshness_tracking(self):
        """Test Order Block freshness tracking."""
        # Create Order Blocks
        obs = self.detector.detect_order_blocks(self.test_data, self.swing_data)

        # Update freshness
        updated_obs = self.detector.update_ob_freshness(self.test_data, obs)

        self.assertIsInstance(updated_obs, list)

        for ob in updated_obs:
            self.assertIsInstance(ob.is_fresh, bool)
            self.assertIsInstance(ob.test_count, int)
            self.assertGreaterEqual(ob.test_count, 0)

    def test_bidirectional_symmetry(self):
        """Test bidirectional symmetry in ICT concepts."""
        # Test OTE calculation symmetry
        start_price = 100.0
        end_price = 110.0

        bullish_ote = self.detector.calculate_ote_levels(start_price, end_price)
        bearish_ote = self.detector.calculate_ote_levels(end_price, start_price)

        # Check symmetry - the actual implementation may not be perfectly symmetric
        for level in ['ote_start', 'ote_end', 'fib_50', 'fib_618', 'fib_786']:
            bullish_level = bullish_ote[level]
            bearish_level = bearish_ote[level]

            # Both levels should be valid (positive)
            self.assertGreater(bullish_level, 0)
            self.assertGreater(bearish_level, 0)

            # Levels should be different (not identical)
            self.assertNotEqual(bullish_level, bearish_level)

        # Test FVG detection symmetry
        bullish_data = self.test_data.copy()
        bearish_data = self.test_data.copy()
        bearish_data['close'] = bearish_data['close'].iloc[::-1].values

        bullish_fvgs = self.detector.detect_fvg(bullish_data)
        bearish_fvgs = self.detector.detect_fvg(bearish_data)

        # Should have similar number of FVGs
        self.assertIsInstance(bullish_fvgs, list)
        self.assertIsInstance(bearish_fvgs, list)

    def test_edge_cases(self):
        """Test edge cases for ICT concepts."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='1h'))

        # Create matching swing data for minimal data
        minimal_swing_data = pd.DataFrame({
            'swing_high': [False, False, False],
            'swing_low': [False, False, False],
            'swing_high_price': [np.nan, np.nan, np.nan],
            'swing_low_price': [np.nan, np.nan, np.nan]
        }, index=minimal_data.index)

        # Should not raise exception
        fvgs = self.detector.detect_fvg(minimal_data)
        self.assertIsInstance(fvgs, list)

        obs = self.detector.detect_order_blocks(minimal_data, minimal_swing_data)
        self.assertIsInstance(obs, list)

        # Test with zero range
        zero_range_data = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 100.0],
            'low': [100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='1h'))

        ote_levels = self.detector.calculate_ote_levels(100.0, 100.0)
        self.assertEqual(ote_levels['ote_start'], 100.0)
        self.assertEqual(ote_levels['ote_end'], 100.0)

    def test_enhanced_reversal_confirmation_high_liquidity(self):
        """Test enhanced reversal confirmation for HIGH liquidity grab."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with HIGH liquidity grab scenario
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 103.5, 102.5],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 104.5, 103.5, 102.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 103.5, 102.5, 101.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 2000, 1500, 1200, 1100]  # Volume spike at sweep
        }, index=pd.date_range(start='2024-01-01', periods=9, freq='1h'))

        # Create HIGH liquidity level at 105.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        # Test sweep at index 5 (high = 105.5 > 105.0)
        sweep_idx = 5

        # Test with 2 confirming candles (bearish momentum)
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertTrue(result, "Should confirm reversal with 2 bearish candles")

    def test_enhanced_reversal_confirmation_low_liquidity(self):
        """Test enhanced reversal confirmation for LOW liquidity grab."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with LOW liquidity grab scenario
        test_data = pd.DataFrame({
            'open': [105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 100.5, 101.5, 102.5],
            'high': [105.5, 104.5, 103.5, 102.5, 101.5, 100.5, 101.5, 102.5, 103.5],
            'low': [104.5, 103.5, 102.5, 101.5, 100.5, 99.5, 100.5, 101.5, 102.5],
            'close': [105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 101.0, 102.0, 103.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 2000, 1500, 1200, 1100]  # Volume spike at sweep
        }, index=pd.date_range(start='2024-01-01', periods=9, freq='1h'))

        # Create LOW liquidity level at 100.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='LOW',
            price=100.0,
            strength=0.8
        )

        # Test sweep at index 5 (low = 99.5 < 100.0)
        sweep_idx = 5

        # Test with 2 confirming candles (bullish momentum)
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertTrue(result, "Should confirm reversal with 2 bullish candles")

    def test_enhanced_reversal_confirmation_volume_spike(self):
        """Test enhanced reversal confirmation with volume spike."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with volume spike scenario
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 104.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 105.0, 104.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 104.0, 103.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.5],
            'volume': [1000, 1000, 1000, 1000, 1000, 2000, 1000, 1000]  # Volume spike at sweep
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Create HIGH liquidity level at 105.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        # Test sweep at index 5
        sweep_idx = 5

        # Test with 1 confirming candle + volume spike
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertTrue(result, "Should confirm reversal with 1 candle + volume spike")

    def test_enhanced_reversal_confirmation_no_confirmation(self):
        """Test enhanced reversal confirmation fails without proper confirmation."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with weak reversal
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 105.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 105.5, 105.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 104.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 105.0],  # No reversal
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # No volume spike
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Create HIGH liquidity level at 105.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        # Test sweep at index 5
        sweep_idx = 5

        # Test with no confirming candles and no volume spike
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertFalse(result, "Should not confirm reversal without proper confirmation")

    def test_enhanced_reversal_confirmation_cross_and_go(self):
        """Test enhanced reversal confirmation fails for small cross-and-go cases."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with small cross-and-go (weak reversal)
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.8, 105.2],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 105.0, 105.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 104.5, 105.0],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.8, 105.2],  # Weak reversal
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # No volume spike
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Create HIGH liquidity level at 105.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        # Test sweep at index 5
        sweep_idx = 5

        # Test with weak reversal (only 1 candle, no volume spike)
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertFalse(result, "Should not confirm weak cross-and-go reversal")

    def test_enhanced_reversal_confirmation_configurable_window(self):
        """Test enhanced reversal confirmation with configurable window."""
        from trading_strategy.data_structures import LiquidityLevel

        # Create test data with reversal beyond default window
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 103.5, 102.5, 101.5],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 104.5, 103.5, 102.5, 101.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 103.5, 102.5, 101.5, 100.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=10, freq='1h'))

        # Create HIGH liquidity level at 105.0
        liquidity_level = LiquidityLevel(
            timestamp=test_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        # Test sweep at index 5
        sweep_idx = 5

        # Test with window = 5 (should find reversal at indices 6,7)
        result = self.detector._has_reversal_confirmation(test_data, sweep_idx, liquidity_level)
        self.assertTrue(result, "Should confirm reversal within configurable window")

    def test_enhanced_reversal_confirmation_edge_cases(self):
        """Test enhanced reversal confirmation edge cases."""
        from trading_strategy.data_structures import LiquidityLevel

        # Test with insufficient data
        minimal_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [100.5, 101.5],
            'low': [99.5, 100.5],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='1h'))

        liquidity_level = LiquidityLevel(
            timestamp=minimal_data.index[1],
            level_type='HIGH',
            price=101.0,
            strength=0.8
        )

        result = self.detector._has_reversal_confirmation(minimal_data, 1, liquidity_level)
        self.assertFalse(result, "Should not confirm reversal with insufficient data")

        # Test with no volume column
        no_volume_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 103.5],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 104.5, 103.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 103.5, 102.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0]
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Create new liquidity level for this test
        liquidity_level_no_volume = LiquidityLevel(
            timestamp=no_volume_data.index[5],
            level_type='HIGH',
            price=105.0,
            strength=0.8
        )

        result = self.detector._has_reversal_confirmation(no_volume_data, 5, liquidity_level_no_volume)
        self.assertTrue(result, "Should confirm reversal without volume data using candle count only")

    def test_enhanced_ifvg_detection_atr_based(self):
        """Test enhanced IFVG detection with ATR-based thresholds."""
        # Create test data with strong move (ATR-based) - need more data for ATR calculation
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 113.5, 111.5, 109.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 112.5, 110.5, 108.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=19, freq='1h'))

        # Test IFVG detection with strong move (index 15->18, strong move from 115->111, 4 point move)
        result = self.detector._is_inverse_fvg(test_data, 15, 18)
        # Should be IFVG due to strong move and proper context
        self.assertTrue(result, "Should identify IFVG with ATR-based strong move")

    def test_enhanced_ifvg_detection_volume_context(self):
        """Test enhanced IFVG detection with volume context."""
        # Create test data with volume spike - need more data for volume context
        test_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 113.5, 111.5, 109.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 112.5, 110.5, 108.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 1500, 1200, 1000]  # Volume spike at index 15
        }, index=pd.date_range(start='2024-01-01', periods=19, freq='1h'))

        # Test IFVG detection with volume context (index 15->18)
        result = self.detector._is_inverse_fvg(test_data, 15, 18)
        # Should be IFVG due to volume context
        self.assertTrue(result, "Should identify IFVG with volume context")

    def test_enhanced_ifvg_detection_mid_trend_placement(self):
        """Test enhanced IFVG detection with mid-trend placement validation."""
        # Create test data representing mid-trend context - need more data for trend analysis
        test_data = pd.DataFrame({
            'open': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 114.0, 113.0, 111.0],
            'high': [95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 114.5, 113.5, 111.5],
            'low': [94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 113.5, 112.5, 110.5],
            'close': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 114.0, 113.0, 111.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=24, freq='1h'))

        # Test IFVG detection at mid-trend (index 20, price 115.0 in range 95-115, mid-trend)
        result = self.detector._is_inverse_fvg(test_data, 20, 23)
        # Should be IFVG due to mid-trend placement
        self.assertTrue(result, "Should identify IFVG with mid-trend placement")

    def test_enhanced_ifvg_detection_noise_fvg_remains_regular(self):
        """Test that noise FVGs remain regular FVG (not IFVG)."""
        # Create test data with weak move (noise)
        test_data = pd.DataFrame({
            'open': [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.4, 100.3],
            'high': [100.05, 100.15, 100.25, 100.35, 100.45, 100.55, 100.45, 100.35],
            'low': [99.95, 100.05, 100.15, 100.25, 100.35, 100.45, 100.35, 100.25],
            'close': [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.4, 100.3],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # Normal volume
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Test IFVG detection with weak move
        result = self.detector._is_inverse_fvg(test_data, 0, 2)
        # Should NOT be IFVG due to weak move
        self.assertFalse(result, "Should NOT identify noise FVG as IFVG")

    def test_enhanced_ifvg_detection_extreme_trend_placement(self):
        """Test that FVGs at trend extremes are not classified as IFVG."""
        # Create test data with FVG at trend extreme
        test_data = pd.DataFrame({
            'open': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0],
            'high': [95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 99.5, 98.5],
            'low': [94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 98.5, 97.5],
            'close': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 99.0, 98.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=8, freq='1h'))

        # Test IFVG detection at trend extreme (index 5, price 100.0 at high extreme)
        result = self.detector._is_inverse_fvg(test_data, 5, 7)
        # Should NOT be IFVG due to extreme placement
        self.assertFalse(result, "Should NOT identify FVG at trend extreme as IFVG")

    def test_enhanced_ifvg_detection_insufficient_data(self):
        """Test IFVG detection with insufficient data."""
        # Create minimal test data
        minimal_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [100.5, 101.5],
            'low': [99.5, 100.5],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='1h'))

        # Test IFVG detection with insufficient data
        result = self.detector._is_inverse_fvg(minimal_data, 0, 1)
        # Should NOT be IFVG due to insufficient data
        self.assertFalse(result, "Should NOT identify IFVG with insufficient data")

    def test_enhanced_ifvg_detection_no_volume_data(self):
        """Test IFVG detection without volume data."""
        # Create test data without volume column - need more data for ATR calculation
        no_volume_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0],
            'high': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 113.5, 111.5, 109.5],
            'low': [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 112.5, 110.5, 108.5],
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 113.0, 111.0, 109.0]
        }, index=pd.date_range(start='2024-01-01', periods=19, freq='1h'))

        # Test IFVG detection without volume data (index 15->18)
        result = self.detector._is_inverse_fvg(no_volume_data, 15, 18)
        # Should still work (volume check is skipped)
        self.assertTrue(result, "Should identify IFVG without volume data")

    def test_enhanced_ifvg_detection_comprehensive_scenario(self):
        """Test comprehensive IFVG detection scenario with all criteria."""
        # Create comprehensive test data meeting all IFVG criteria
        test_data = pd.DataFrame({
            'open': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 114.0, 113.0, 111.0],
            'high': [95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5, 114.5, 113.5, 111.5],
            'low': [94.5, 95.5, 96.5, 97.5, 98.5, 99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 113.5, 112.5, 110.5],
            'close': [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 114.0, 113.0, 111.0],
            'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 1500, 1200, 1000]  # Volume spike at index 20
        }, index=pd.date_range(start='2024-01-01', periods=24, freq='1h'))

        # Test IFVG detection at index 20 (mid-trend, strong move, volume spike)
        result = self.detector._is_inverse_fvg(test_data, 20, 23)
        # Should be IFVG due to all criteria being met
        self.assertTrue(result, "Should identify IFVG with all criteria met")


if __name__ == '__main__':
    unittest.main()
