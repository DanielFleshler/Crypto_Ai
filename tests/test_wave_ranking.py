"""
Unit tests for wave ranking system.

Tests objective ranking logic with:
- Fibonacci proximity scoring
- Volume profile/OBV-based scoring
- Score normalization to [0,1]
- Deterministic ranking with tie-breaking
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.data_structures import ElliottWave
from trading_strategy.config_loader import ConfigLoader


class TestWaveRanking(unittest.TestCase):
    """Test wave ranking system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
        self.strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        self.test_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        }, index=dates)

        # Create test waves with different characteristics
        self.wave_perfect_fib = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),
            start_price=100.0, end_price=161.8,  # Perfect 61.8% extension
            status='current'
        )

        self.wave_high_volume = ElliottWave(
            wave_number=2, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 12), end_time=datetime(2024, 1, 1, 14),
            start_price=100.0, end_price=110.0,  # 10% move
            status='current'
        )

        self.wave_recent = ElliottWave(
            wave_number=3, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 14), end_time=datetime(2024, 1, 1, 16),
            start_price=100.0, end_price=105.0,  # 5% move
            status='current'
        )

        self.wave_old = ElliottWave(
            wave_number=4, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 8), end_time=datetime(2024, 1, 1, 10),
            start_price=100.0, end_price=120.0,  # 20% move
            status='current'
        )

        # Create high volume data for wave_high_volume
        self.high_volume_data = self.test_data.copy()
        self.high_volume_data.loc[self.high_volume_data.index[12:14], 'volume'] = 5000  # High volume spike

        # Create swing data
        self.swing_data = pd.DataFrame({
            'swing_high': [False] * 100,
            'swing_low': [False] * 100,
            'swing_high_price': [np.nan] * 100,
            'swing_low_price': [np.nan] * 100
        }, index=dates)

    def test_fibonacci_proximity_scoring(self):
        """Test Fibonacci proximity scoring."""
        # Test perfect Fibonacci alignment
        fib_score = self.strategy._calculate_fibonacci_score(self.wave_perfect_fib, self.test_data)
        self.assertGreater(fib_score, 0.8, "Perfect Fibonacci alignment should score high")

        # Test poor Fibonacci alignment
        wave_poor_fib = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),
            start_price=100.0, end_price=103.7,  # Random price, not Fibonacci level
            status='current'
        )
        poor_fib_score = self.strategy._calculate_fibonacci_score(wave_poor_fib, self.test_data)
        self.assertLess(poor_fib_score, fib_score, "Poor Fibonacci alignment should score lower")

        # Test score normalization
        self.assertGreaterEqual(fib_score, 0.0, "Fibonacci score should be >= 0")
        self.assertLessEqual(fib_score, 1.0, "Fibonacci score should be <= 1")

    def test_volume_profile_scoring(self):
        """Test volume profile scoring."""
        # Test high volume wave
        high_vol_score = self.strategy._calculate_volume_score(self.wave_high_volume, self.high_volume_data)

        # Test normal volume wave
        normal_vol_score = self.strategy._calculate_volume_score(self.wave_high_volume, self.test_data)

        self.assertGreater(high_vol_score, normal_vol_score, "High volume wave should score higher")

        # Test score normalization
        self.assertGreaterEqual(high_vol_score, 0.0, "Volume score should be >= 0")
        self.assertLessEqual(high_vol_score, 1.0, "Volume score should be <= 1")

    def test_structure_confirmation_scoring(self):
        """Test structure confirmation scoring."""
        # Mock structures
        mock_structures = [
            type('MockStructure', (), {'timestamp': datetime(2024, 1, 1, 11)})(),  # Nearby
            type('MockStructure', (), {'timestamp': datetime(2024, 1, 1, 11, 30)})(),  # Nearby
            type('MockStructure', (), {'timestamp': datetime(2024, 1, 1, 20)})(),  # Far
        ]

        # Test with nearby structures
        structure_score = self.strategy._calculate_structure_score(self.wave_perfect_fib, mock_structures)
        self.assertGreater(structure_score, 0.0, "Nearby structures should increase score")

        # Test with no structures
        no_structure_score = self.strategy._calculate_structure_score(self.wave_perfect_fib, [])
        self.assertEqual(no_structure_score, 0.0, "No structures should result in 0 score")

        # Test score normalization
        self.assertGreaterEqual(structure_score, 0.0, "Structure score should be >= 0")
        self.assertLessEqual(structure_score, 1.0, "Structure score should be <= 1")

    def test_session_timing_scoring(self):
        """Test session timing scoring."""
        # Test London session (preferred)
        london_wave = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),  # 10-12 UTC (London)
            start_price=100.0, end_price=110.0,
            status='current'
        )
        london_score = self.strategy._calculate_session_score(london_wave, self.test_data)

        # Test Asia session (less preferred)
        asia_wave = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 2), end_time=datetime(2024, 1, 1, 4),  # 2-4 UTC (Asia)
            start_price=100.0, end_price=110.0,
            status='current'
        )
        asia_score = self.strategy._calculate_session_score(asia_wave, self.test_data)

        self.assertGreater(london_score, asia_score, "London session should score higher than Asia")

        # Test score normalization
        self.assertGreaterEqual(london_score, 0.0, "Session score should be >= 0")
        self.assertLessEqual(london_score, 1.0, "Session score should be <= 1")

    def test_deterministic_ranking(self):
        """Test deterministic ranking with fixed data."""
        # Create waves with known characteristics
        waves = [self.wave_perfect_fib, self.wave_high_volume, self.wave_recent, self.wave_old]

        # Rank waves
        ranked_waves = self.strategy._rank_wave_candidates(waves, self.test_data, self.swing_data)

        # Should return same order for same input
        ranked_waves_2 = self.strategy._rank_wave_candidates(waves, self.test_data, self.swing_data)
        self.assertEqual(ranked_waves, ranked_waves_2, "Ranking should be deterministic")

        # Should return all waves
        self.assertEqual(len(ranked_waves), len(waves), "Should return all waves")

        # Should be sorted (highest score first)
        for i in range(len(ranked_waves) - 1):
            self.assertGreaterEqual(
                self._get_wave_score(ranked_waves[i], self.test_data),
                self._get_wave_score(ranked_waves[i + 1], self.test_data),
                "Waves should be sorted by score"
            )

    def test_tie_breaking_recency(self):
        """Test tie-breaking by recency."""
        # Create waves with same characteristics but different times
        recent_wave = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 14), end_time=datetime(2024, 1, 1, 16),
            start_price=100.0, end_price=110.0,
            status='current'
        )

        older_wave = ElliottWave(
            wave_number=2, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),
            start_price=100.0, end_price=110.0,
            status='current'
        )

        waves = [older_wave, recent_wave]
        ranked_waves = self.strategy._rank_wave_candidates(waves, self.test_data, self.swing_data)

        # Recent wave should rank higher
        self.assertEqual(ranked_waves[0], recent_wave, "Recent wave should rank higher")

    def test_tie_breaking_strength(self):
        """Test tie-breaking by strength."""
        # Create waves with same time but different strengths
        strong_wave = ElliottWave(
            wave_number=1, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),
            start_price=100.0, end_price=120.0,  # 20% move
            status='current'
        )

        weak_wave = ElliottWave(
            wave_number=2, wave_type='IMPULSE',
            start_time=datetime(2024, 1, 1, 10), end_time=datetime(2024, 1, 1, 12),
            start_price=100.0, end_price=105.0,  # 5% move
            status='current'
        )

        waves = [weak_wave, strong_wave]
        ranked_waves = self.strategy._rank_wave_candidates(waves, self.test_data, self.swing_data)

        # Strong wave should rank higher
        self.assertEqual(ranked_waves[0], strong_wave, "Strong wave should rank higher")

    def test_score_normalization(self):
        """Test that all scores are normalized to [0,1] range."""
        waves = [self.wave_perfect_fib, self.wave_high_volume, self.wave_recent, self.wave_old]

        for wave in waves:
            fib_score = self.strategy._calculate_fibonacci_score(wave, self.test_data)
            volume_score = self.strategy._calculate_volume_score(wave, self.test_data)
            structure_score = self.strategy._calculate_structure_score(wave, [])
            session_score = self.strategy._calculate_session_score(wave, self.test_data)

            # All scores should be in [0,1] range
            self.assertGreaterEqual(fib_score, 0.0, f"Fibonacci score should be >= 0 for wave {wave.wave_number}")
            self.assertLessEqual(fib_score, 1.0, f"Fibonacci score should be <= 1 for wave {wave.wave_number}")

            self.assertGreaterEqual(volume_score, 0.0, f"Volume score should be >= 0 for wave {wave.wave_number}")
            self.assertLessEqual(volume_score, 1.0, f"Volume score should be <= 1 for wave {wave.wave_number}")

            self.assertGreaterEqual(structure_score, 0.0, f"Structure score should be >= 0 for wave {wave.wave_number}")
            self.assertLessEqual(structure_score, 1.0, f"Structure score should be <= 1 for wave {wave.wave_number}")

            self.assertGreaterEqual(session_score, 0.0, f"Session score should be >= 0 for wave {wave.wave_number}")
            self.assertLessEqual(session_score, 1.0, f"Session score should be <= 1 for wave {wave.wave_number}")

    def test_weighted_composite_scoring(self):
        """Test weighted composite scoring."""
        waves = [self.wave_perfect_fib, self.wave_high_volume]
        ranked_waves = self.strategy._rank_wave_candidates(waves, self.test_data, self.swing_data)

        # Should consider all scoring factors
        self.assertIsInstance(ranked_waves, list, "Should return list of ranked waves")
        self.assertEqual(len(ranked_waves), len(waves), "Should return all waves")

        # Should be sorted by composite score
        for i in range(len(ranked_waves) - 1):
            current_score = self._get_wave_score(ranked_waves[i], self.test_data)
            next_score = self._get_wave_score(ranked_waves[i + 1], self.test_data)
            self.assertGreaterEqual(current_score, next_score, "Waves should be sorted by composite score")

    def test_empty_candidates_list(self):
        """Test ranking with empty candidates list."""
        ranked_waves = self.strategy._rank_wave_candidates([], self.test_data, self.swing_data)
        self.assertEqual(ranked_waves, [], "Empty candidates should return empty list")

    def _get_wave_score(self, wave, df):
        """Helper method to get composite score for a wave."""
        fib_score = self.strategy._calculate_fibonacci_score(wave, df)
        volume_score = self.strategy._calculate_volume_score(wave, df)
        structure_score = self.strategy._calculate_structure_score(wave, [])
        session_score = self.strategy._calculate_session_score(wave, df)

        return (
            fib_score * self.strategy.ranking_config.fibonacci_proximity_weight +
            volume_score * self.strategy.ranking_config.volume_profile_weight +
            structure_score * self.strategy.ranking_config.structure_confirmation_weight +
            session_score * self.strategy.ranking_config.session_timing_weight
        )


if __name__ == '__main__':
    unittest.main()
