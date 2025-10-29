"""
Test suite for market structure detection with bidirectional symmetry.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.data_structures import MarketStructure
from trading_strategy.config_loader import ConfigLoader


class TestMarketStructure(unittest.TestCase):
    """Test market structure detection for bidirectional symmetry."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
        self.detector = MarketStructureDetector(self.config_loader)

        # Create test data
        self.test_data = self._create_test_data()

    def _create_test_data(self):
        """Create test OHLC data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')

        # Create trending data
        base_price = 100.0
        prices = []
        for i in range(len(dates)):
            # Add trend
            trend = i * 0.1
            price = base_price + trend
            prices.append(price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p + np.random.uniform(0, 1) for p in prices],
            'low': [p - np.random.uniform(0, 1) for p in prices],
            'close': prices,
            'volume': [1000 + np.random.randint(0, 500) for _ in prices]
        }, index=dates)

        return df

    def test_swing_point_detection(self):
        """Test swing point detection."""
        swing_df = self.detector.detect_swing_points(self.test_data)

        self.assertIsInstance(swing_df, pd.DataFrame)
        self.assertIn('swing_high', swing_df.columns)
        self.assertIn('swing_low', swing_df.columns)
        self.assertIn('swing_high_price', swing_df.columns)
        self.assertIn('swing_low_price', swing_df.columns)

        # Check that swing points are detected
        swing_highs = swing_df[swing_df['swing_high']]
        swing_lows = swing_df[swing_df['swing_low']]

        self.assertGreater(len(swing_highs), 0)
        self.assertGreater(len(swing_lows), 0)

    def test_market_structure_detection(self):
        """Test market structure detection."""
        swing_df = self.detector.detect_swing_points(self.test_data)
        structures = self.detector.detect_market_structure(swing_df)

        self.assertIsInstance(structures, list)

        for structure in structures:
            self.assertIsInstance(structure, MarketStructure)
            self.assertIn(structure.structure_type, ['BOS', 'CHoCH', 'HH', 'HL', 'LH', 'LL'])
            self.assertIn(structure.trend_direction, ['BULLISH', 'BEARISH', 'NEUTRAL'])
            self.assertGreater(structure.strength, 0)
            self.assertLessEqual(structure.strength, 1)

    def test_bias_determination_bullish(self):
        """Test bias determination for bullish scenario."""
        # Create bullish structures
        structures = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BULLISH'
            ),
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 20),
                structure_type='BOS',
                price=115.0,
                timeframe='1h',
                strength=0.9,
                trend_direction='BULLISH'
            )
        ]

        bias = self.detector.get_current_bias(structures)
        self.assertEqual(bias, 'BULLISH')

    def test_bias_determination_bearish(self):
        """Test bias determination for bearish scenario."""
        # Create bearish structures
        structures = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BEARISH'
            ),
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 20),
                structure_type='BOS',
                price=105.0,
                timeframe='1h',
                strength=0.9,
                trend_direction='BEARISH'
            )
        ]

        bias = self.detector.get_current_bias(structures)
        self.assertEqual(bias, 'BEARISH')

    def test_bias_determination_neutral(self):
        """Test bias determination for neutral scenario."""
        # Create mixed structures
        structures = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BULLISH'
            ),
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 20),
                structure_type='BOS',
                price=105.0,
                timeframe='1h',
                strength=0.9,
                trend_direction='BEARISH'
            )
        ]

        bias = self.detector.get_current_bias(structures)
        self.assertEqual(bias, 'NEUTRAL')

    def test_liquidity_level_tracking(self):
        """Test liquidity level tracking."""
        swing_df = self.detector.detect_swing_points(self.test_data)
        liquidity_levels = self.detector.track_liquidity_levels(self.test_data, swing_df)

        self.assertIsInstance(liquidity_levels, list)

        for level in liquidity_levels:
            self.assertIn(level.level_type, ['HIGH', 'LOW'])
            self.assertGreater(level.price, 0)
            self.assertGreaterEqual(level.strength, 0)
            self.assertLessEqual(level.strength, 1)
            self.assertFalse(level.is_swept)
            self.assertFalse(level.reversal_confirmed)

    def test_liquidity_sweep_detection(self):
        """Test liquidity sweep detection."""
        swing_df = self.detector.detect_swing_points(self.test_data)
        liquidity_levels = self.detector.track_liquidity_levels(self.test_data, swing_df)

        sweeps = self.detector.detect_liquidity_sweeps(self.test_data)

        self.assertIsInstance(sweeps, list)

        for sweep in sweeps:
            self.assertIn('timestamp', sweep)
            self.assertIn('level_type', sweep)
            self.assertIn('price', sweep)
            self.assertIn('swept', sweep)
            self.assertIn('reversal_confirmed', sweep)

    def test_structure_strength_calculation(self):
        """Test structure strength calculation."""
        structure = MarketStructure(
            timestamp=datetime(2024, 1, 1, 10),
            structure_type='BOS',
            price=110.0,
            timeframe='1h',
            strength=0.8,
            trend_direction='BULLISH',
            volume_at_break=1000000,
            impulse_strength=0.7,
            confirmation_count=2
        )

        strength_score = self.detector.calculate_structure_strength(structure)

        self.assertGreaterEqual(strength_score, 0)
        self.assertLessEqual(strength_score, 1)
        self.assertGreater(strength_score, structure.strength)  # Should be enhanced

    def test_multi_timeframe_alignment(self):
        """Test multi-timeframe structure alignment."""
        # Create HTF structures
        htf_structures = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='4h',
                strength=0.8,
                trend_direction='BULLISH'
            )
        ]

        # Create MTF structures
        mtf_structures = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BULLISH'
            )
        ]

        is_aligned = self.detector.validate_multi_timeframe_alignment(htf_structures, mtf_structures)
        self.assertTrue(is_aligned)

        # Test misaligned structures
        misaligned_mtf = [
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='BOS',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BEARISH'
            )
        ]

        is_misaligned = self.detector.validate_multi_timeframe_alignment(htf_structures, misaligned_mtf)
        self.assertFalse(is_misaligned)

    def test_bidirectional_symmetry(self):
        """Test bidirectional symmetry in market structure."""
        # Create bullish and bearish test data
        bullish_data = self.test_data.copy()
        bearish_data = self.test_data.copy()
        bearish_data['close'] = bearish_data['close'].iloc[::-1].values

        # Test swing point detection symmetry
        bullish_swings = self.detector.detect_swing_points(bullish_data)
        bearish_swings = self.detector.detect_swing_points(bearish_data)

        self.assertIsInstance(bullish_swings, pd.DataFrame)
        self.assertIsInstance(bearish_swings, pd.DataFrame)

        # Test structure detection symmetry
        bullish_structures = self.detector.detect_market_structure(bullish_swings)
        bearish_structures = self.detector.detect_market_structure(bearish_swings)

        self.assertIsInstance(bullish_structures, list)
        self.assertIsInstance(bearish_structures, list)

        # Test bias determination symmetry
        bullish_bias = self.detector.get_current_bias(bullish_structures)
        bearish_bias = self.detector.get_current_bias(bearish_structures)

        # Should have opposite biases
        if bullish_bias == 'BULLISH':
            self.assertIn(bearish_bias, ['BEARISH', 'NEUTRAL'])
        elif bullish_bias == 'BEARISH':
            self.assertIn(bearish_bias, ['BULLISH', 'NEUTRAL'])

    def test_edge_cases(self):
        """Test edge cases for market structure detection."""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [100.5, 101.5, 102.5],
            'low': [99.5, 100.5, 101.5],
            'close': [100.0, 101.0, 102.0],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range(start='2024-01-01', periods=3, freq='1h'))

        # Should not raise exception
        swing_df = self.detector.detect_swing_points(minimal_data)
        self.assertIsInstance(swing_df, pd.DataFrame)

        structures = self.detector.detect_market_structure(swing_df)
        self.assertIsInstance(structures, list)

        # Test with empty data
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_swing_df = self.detector.detect_swing_points(empty_data)
        self.assertIsInstance(empty_swing_df, pd.DataFrame)

        empty_structures = self.detector.detect_market_structure(empty_swing_df)
        self.assertIsInstance(empty_structures, list)
        self.assertEqual(len(empty_structures), 0)

    def test_structure_validation(self):
        """Test structure validation rules."""
        # Test valid structure
        valid_structure = MarketStructure(
            timestamp=datetime(2024, 1, 1, 10),
            structure_type='BOS',
            price=110.0,
            timeframe='1h',
            strength=0.8,
            trend_direction='BULLISH'
        )

        # Should not raise exception
        self.assertIsInstance(valid_structure, MarketStructure)

        # Test invalid structure (should raise exception)
        with self.assertRaises(ValueError):
            MarketStructure(
                timestamp=datetime(2024, 1, 1, 10),
                structure_type='INVALID',
                price=110.0,
                timeframe='1h',
                strength=0.8,
                trend_direction='BULLISH'
            )

    def test_structure_methods(self):
        """Test structure methods."""
        structure = MarketStructure(
            timestamp=datetime(2024, 1, 1, 10),
            structure_type='BOS',
            price=110.0,
            timeframe='1h',
            strength=0.8,
            trend_direction='BULLISH'
        )

        # Test direction methods
        self.assertTrue(structure.is_bullish_structure())
        self.assertFalse(structure.is_bearish_structure())

        # Test type methods
        self.assertTrue(structure.is_break_of_structure())
        self.assertFalse(structure.is_change_of_character())

        # Test confirmation
        initial_count = structure.confirmation_count
        structure.add_confirmation()
        self.assertEqual(structure.confirmation_count, initial_count + 1)

        # Test strength score
        strength_score = structure.get_strength_score()
        self.assertGreaterEqual(strength_score, 0)
        self.assertLessEqual(strength_score, 1)


if __name__ == '__main__':
    unittest.main()
