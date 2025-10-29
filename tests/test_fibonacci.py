"""
Test suite for Fibonacci calculations with bidirectional symmetry.
"""

import unittest
import numpy as np
from datetime import datetime
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.config_loader import ConfigLoader


class TestFibonacci(unittest.TestCase):
    """Test Fibonacci calculations for bidirectional symmetry."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
        self.detector = ElliottWaveDetector(self.config_loader)

    def test_bullish_retracement_calculation(self):
        """Test bullish retracement calculations."""
        start_price = 100.0
        end_price = 110.0  # 10% bullish move

        fibs = self.detector.calculate_fibonacci_retracement(start_price, end_price)

        # Check that retracement levels are below end price
        self.assertLess(fibs['fib_0.236'], end_price)
        self.assertLess(fibs['fib_0.382'], end_price)
        self.assertLess(fibs['fib_0.5'], end_price)
        self.assertLess(fibs['fib_0.618'], end_price)
        self.assertLess(fibs['fib_0.786'], end_price)

        # Check that levels are in correct order (descending for bullish)
        self.assertGreater(fibs['fib_0.236'], fibs['fib_0.382'])
        self.assertGreater(fibs['fib_0.382'], fibs['fib_0.5'])
        self.assertGreater(fibs['fib_0.5'], fibs['fib_0.618'])
        self.assertGreater(fibs['fib_0.618'], fibs['fib_0.786'])

    def test_bearish_retracement_calculation(self):
        """Test bearish retracement calculations."""
        start_price = 110.0
        end_price = 100.0  # 10% bearish move

        fibs = self.detector.calculate_fibonacci_retracement(start_price, end_price)

        # Check that retracement levels are above end price
        self.assertGreater(fibs['fib_0.236'], end_price)
        self.assertGreater(fibs['fib_0.382'], end_price)
        self.assertGreater(fibs['fib_0.5'], end_price)
        self.assertGreater(fibs['fib_0.618'], end_price)
        self.assertGreater(fibs['fib_0.786'], end_price)

        # Check that levels are in correct order (ascending for bearish)
        self.assertLess(fibs['fib_0.236'], fibs['fib_0.382'])
        self.assertLess(fibs['fib_0.382'], fibs['fib_0.5'])
        self.assertLess(fibs['fib_0.5'], fibs['fib_0.618'])
        self.assertLess(fibs['fib_0.618'], fibs['fib_0.786'])

    def test_bullish_extension_calculation(self):
        """Test bullish extension calculations."""
        start_price = 100.0
        end_price = 110.0
        extension_start = 105.0

        exts = self.detector.calculate_fibonacci_extension(start_price, end_price, extension_start)

        # Check that extension levels are above extension start
        self.assertGreater(exts['ext_1.0'], extension_start)
        self.assertGreater(exts['ext_1.272'], extension_start)
        self.assertGreater(exts['ext_1.414'], extension_start)
        self.assertGreater(exts['ext_1.618'], extension_start)

        # Check that levels are in correct order
        self.assertLess(exts['ext_1.0'], exts['ext_1.272'])
        self.assertLess(exts['ext_1.272'], exts['ext_1.414'])
        self.assertLess(exts['ext_1.414'], exts['ext_1.618'])

    def test_bearish_extension_calculation(self):
        """Test bearish extension calculations."""
        start_price = 110.0
        end_price = 100.0
        extension_start = 105.0

        exts = self.detector.calculate_fibonacci_extension(start_price, end_price, extension_start)

        # Check that extension levels are below extension start
        self.assertLess(exts['ext_1.0'], extension_start)
        self.assertLess(exts['ext_1.272'], extension_start)
        self.assertLess(exts['ext_1.414'], extension_start)
        self.assertLess(exts['ext_1.618'], extension_start)

        # Check that levels are in correct order (reversed for bearish)
        self.assertGreater(exts['ext_1.0'], exts['ext_1.272'])
        self.assertGreater(exts['ext_1.272'], exts['ext_1.414'])
        self.assertGreater(exts['ext_1.414'], exts['ext_1.618'])

    def test_bidirectional_symmetry(self):
        """Test bidirectional symmetry between bullish and bearish calculations."""
        start_price = 100.0
        end_price = 110.0
        extension_start = 105.0

        # Bullish calculations
        bullish_fibs = self.detector.calculate_fibonacci_retracement(start_price, end_price)
        bullish_exts = self.detector.calculate_fibonacci_extension(start_price, end_price, extension_start)

        # Bearish calculations (inverted)
        bearish_fibs = self.detector.calculate_fibonacci_retracement(end_price, start_price)
        bearish_exts = self.detector.calculate_fibonacci_extension(end_price, start_price, extension_start)

        # Check symmetry for retracements
        for level in ['fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786']:
            bullish_level = bullish_fibs[level]
            bearish_level = bearish_fibs[level]

            # Symmetry: bullish level + bearish level should equal start_price + end_price
            expected_sum = start_price + end_price
            actual_sum = bullish_level + bearish_level
            self.assertAlmostEqual(actual_sum, expected_sum, places=2)

        # Check symmetry for extensions
        for level in ['ext_1.0', 'ext_1.272', 'ext_1.414', 'ext_1.618']:
            bullish_level = bullish_exts[level]
            bearish_level = bearish_exts[level]

            # Symmetry: bullish level + bearish level should equal 2 * extension_start
            expected_sum = 2 * extension_start
            actual_sum = bullish_level + bearish_level
            self.assertAlmostEqual(actual_sum, expected_sum, places=2)

    def test_edge_cases(self):
        """Test edge cases for Fibonacci calculations."""
        # Test zero range
        start_price = 100.0
        end_price = 100.0

        fibs = self.detector.calculate_fibonacci_retracement(start_price, end_price)
        exts = self.detector.calculate_fibonacci_extension(start_price, end_price, start_price)

        # All levels should equal start price
        for level in fibs.values():
            self.assertEqual(level, start_price)

        for level in exts.values():
            self.assertEqual(level, start_price)

        # Test negative values (should handle gracefully)
        start_price = -100.0
        end_price = -110.0

        fibs = self.detector.calculate_fibonacci_retracement(start_price, end_price)
        exts = self.detector.calculate_fibonacci_extension(start_price, end_price, -105.0)

        # Should not raise exception
        self.assertIsInstance(fibs, dict)
        self.assertIsInstance(exts, dict)


if __name__ == '__main__':
    unittest.main()
