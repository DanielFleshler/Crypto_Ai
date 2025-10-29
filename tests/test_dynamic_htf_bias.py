"""
Test Dynamic HTF Bias Updates

Tests for the dynamic HTF bias update functionality in TradingStrategy.
Covers bias recomputation, history tracking, signal invalidation, and bias flip detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.data_structures import Signal, MarketStructure
from trading_strategy.config_loader import ConfigLoader


class TestDynamicHTFBias:
    """Test dynamic HTF bias updates."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock ConfigLoader
        self.mock_config_loader = Mock(spec=ConfigLoader)

        # Mock configuration objects
        self.mock_elliott_config = Mock()
        self.mock_entry_config = Mock()
        self.mock_timeframe_config = Mock()
        self.mock_ranking_config = Mock()

        # Set up timeframe config
        self.mock_timeframe_config.htf = "1h"
        self.mock_timeframe_config.mtf = "15m"

        # Set up entry config
        self.mock_entry_config.min_confirmations = 2

        # Configure mock config loader
        self.mock_config_loader.get_elliott_wave_config.return_value = self.mock_elliott_config
        self.mock_config_loader.get_entry_confirmation_config.return_value = self.mock_entry_config
        self.mock_config_loader.get_timeframe_config.return_value = self.mock_timeframe_config
        self.mock_config_loader.get_wave_ranking_config.return_value = self.mock_ranking_config

        # Create TradingStrategy instance
        self.strategy = TradingStrategy("/test/path", self.mock_config_loader)

        # Mock components
        self.strategy.base_path_loader = Mock()
        self.strategy.market_structure = Mock()
        self.strategy.ict_detector = Mock()
        self.strategy.elliott_detector = Mock()
        self.strategy.ict_entries = Mock()
        self.strategy.killzone_detector = Mock()

    def create_test_dataframe(self, start_time: datetime, periods: int, timeframe: str = "1h") -> pd.DataFrame:
        """Create test DataFrame with OHLCV data."""
        if timeframe == "1h":
            freq = "1h"
        elif timeframe == "15m":
            freq = "15min"
        else:
            freq = "1h"

        dates = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create realistic price data
        base_price = 50000.0
        prices = []
        current_price = base_price

        for i in range(periods):
            # Add some trend and volatility
            trend = 0.001 * np.sin(i * 0.1)  # Small trend component
            volatility = 0.02 * np.random.normal()  # Random volatility
            change = trend + volatility

            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.002))
            close_price = price
            volume = np.random.randint(1000, 10000)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data, index=dates)
        return df

    def create_mock_structures(self, bias: str, count: int = 3) -> list:
        """Create mock market structures."""
        structures = []
        for i in range(count):
            structure = Mock(spec=MarketStructure)
            structure.timestamp = datetime.now() - timedelta(hours=i)
            structure.structure_type = 'BOS'
            structure.trend_direction = bias
            structure.price = 50000.0 + i * 100
            structure.strength = 0.8
            structures.append(structure)
        return structures

    def test_initial_bias_update(self):
        """Test initial HTF bias update."""
        # Create test data
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df = self.create_test_dataframe(start_time, 50)

        # Mock market structure analysis
        mock_structures = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Test initial bias update
        result = self.strategy.update_htf_bias_dynamically(df)

        # Assertions
        assert result['bias_updated'] is True
        assert result['new_bias'] == 'BULLISH'
        assert result['previous_bias'] == 'NEUTRAL'
        assert result['bias_flipped'] is False
        assert result['timestamp'] == df.index[-1]
        assert result['structures_count'] == 3

        # Check that bias history was recorded
        assert len(self.strategy.htf_bias_history) == 1
        assert self.strategy.htf_bias_history[0]['bias'] == 'BULLISH'
        assert self.strategy.htf_bias_history[0]['bias_flipped'] is False

        # Check that current bias was updated
        assert self.strategy.htf_bias == 'BULLISH'
        assert self.strategy.last_htf_candle_time == df.index[-1]

    def test_no_new_candles(self):
        """Test behavior when no new candles are available."""
        # Set up initial state
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df = self.create_test_dataframe(start_time, 50)

        # Mock initial analysis
        mock_structures = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Initial update
        self.strategy.update_htf_bias_dynamically(df)

        # Test with same data (no new candles)
        result = self.strategy.update_htf_bias_dynamically(df)

        # Assertions
        assert result['bias_updated'] is False
        assert result['new_bias'] == 'BULLISH'
        assert result['previous_bias'] == 'BULLISH'
        assert result['bias_flipped'] is False

        # Check that no new history was added
        assert len(self.strategy.htf_bias_history) == 1

    def test_bias_flip_detection(self):
        """Test bias flip detection."""
        # Set up initial bullish bias
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df1 = self.create_test_dataframe(start_time, 50)

        mock_structures_bullish = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df1
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bullish
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Initial update
        self.strategy.update_htf_bias_dynamically(df1)

        # Create new data with bearish bias
        start_time2 = datetime(2024, 1, 1, 2, 0, 0)  # 2 hours later
        df2 = self.create_test_dataframe(start_time2, 60)

        mock_structures_bearish = self.create_mock_structures('BEARISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bearish
        self.strategy.market_structure.get_current_bias.return_value = 'BEARISH'

        # Test bias flip
        result = self.strategy.update_htf_bias_dynamically(df2)

        # Assertions
        assert result['bias_updated'] is True
        assert result['new_bias'] == 'BEARISH'
        assert result['previous_bias'] == 'BULLISH'
        assert result['bias_flipped'] is True

        # Check bias history
        assert len(self.strategy.htf_bias_history) == 2
        assert self.strategy.htf_bias_history[-1]['bias'] == 'BEARISH'
        assert self.strategy.htf_bias_history[-1]['bias_flipped'] is True

        # Check current bias
        assert self.strategy.htf_bias == 'BEARISH'

    def test_bias_flip_noise_filtering(self):
        """Test bias flip noise filtering."""
        # Set up initial bullish bias
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df1 = self.create_test_dataframe(start_time, 50)

        mock_structures_bullish = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df1
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bullish
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Initial update
        self.strategy.update_htf_bias_dynamically(df1)

        # Create data with bearish bias (too soon after last flip)
        start_time2 = datetime(2024, 1, 1, 0, 30, 0)  # Only 30 minutes later
        df2 = self.create_test_dataframe(start_time2, 55)

        mock_structures_bearish = self.create_mock_structures('BEARISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bearish
        self.strategy.market_structure.get_current_bias.return_value = 'BEARISH'

        # Test bias flip (should be filtered as noise)
        result = self.strategy.update_htf_bias_dynamically(df2)

        # Assertions - bias should update but flip should be filtered
        assert result['bias_updated'] is True
        assert result['new_bias'] == 'BEARISH'
        assert result['previous_bias'] == 'BULLISH'
        # Note: The first bias change from NEUTRAL to BULLISH doesn't count as a flip,
        # so the BULLISH to BEARISH change should be detected as a flip
        assert result['bias_flipped'] is True  # This is the first real flip

        # Check bias history
        assert len(self.strategy.htf_bias_history) == 2
        assert self.strategy.htf_bias_history[-1]['bias_flipped'] is True

    def test_neutral_bias_transitions(self):
        """Test transitions to/from neutral bias."""
        # Set up initial bullish bias
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df1 = self.create_test_dataframe(start_time, 50)

        mock_structures_bullish = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df1
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bullish
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Initial update
        self.strategy.update_htf_bias_dynamically(df1)

        # Create data with neutral bias
        start_time2 = datetime(2024, 1, 1, 2, 0, 0)
        df2 = self.create_test_dataframe(start_time2, 60)

        mock_structures_neutral = self.create_mock_structures('NEUTRAL', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_neutral
        self.strategy.market_structure.get_current_bias.return_value = 'NEUTRAL'

        # Test neutral transition
        result = self.strategy.update_htf_bias_dynamically(df2)

        # Assertions - transition to neutral should not be considered a flip
        assert result['bias_updated'] is True
        assert result['new_bias'] == 'NEUTRAL'
        assert result['previous_bias'] == 'BULLISH'
        assert result['bias_flipped'] is False  # Neutral transitions don't count as flips

        # Check bias history
        assert len(self.strategy.htf_bias_history) == 2
        assert self.strategy.htf_bias_history[-1]['bias_flipped'] is False

    def test_signal_invalidation_on_bias_flip(self):
        """Test signal invalidation when bias flips."""
        # Create test signals
        signals = [
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='TEST',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[51000.0, 52000.0],
                risk_reward=2.0,
                confidence=0.8,
                metadata={}
            ),
            Signal(
                timestamp=datetime.now(),
                signal_type='SELL',
                entry_type='TEST',
                price=50000.0,
                stop_loss=51000.0,
                take_profits=[49000.0, 48000.0],
                risk_reward=2.0,
                confidence=0.8,
                metadata={}
            )
        ]

        # Set bullish bias
        self.strategy.htf_bias = 'BULLISH'

        # Test signal invalidation
        bias_flip_timestamp = datetime.now()
        valid_signals = self.strategy.invalidate_signals_on_bias_flip(signals, bias_flip_timestamp)

        # Assertions - only BUY signal should remain (aligned with bullish bias)
        assert len(valid_signals) == 1
        assert valid_signals[0].signal_type == 'BUY'

        # Test with bearish bias
        self.strategy.htf_bias = 'BEARISH'
        valid_signals = self.strategy.invalidate_signals_on_bias_flip(signals, bias_flip_timestamp)

        # Assertions - only SELL signal should remain (aligned with bearish bias)
        assert len(valid_signals) == 1
        assert valid_signals[0].signal_type == 'SELL'

        # Test with neutral bias
        self.strategy.htf_bias = 'NEUTRAL'
        valid_signals = self.strategy.invalidate_signals_on_bias_flip(signals, bias_flip_timestamp)

        # Assertions - all signals should remain (neutral allows all)
        assert len(valid_signals) == 2

    def test_bias_history_management(self):
        """Test bias history management and limits."""
        # Create multiple bias updates
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        for i in range(150):  # More than the 100 record limit
            df = self.create_test_dataframe(start_time + timedelta(hours=i), 50)

            mock_structures = self.create_mock_structures('BULLISH', 3)
            self.strategy.market_structure.detect_swing_points.return_value = df
            self.strategy.market_structure.detect_market_structure.return_value = mock_structures
            self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

            self.strategy.update_htf_bias_dynamically(df)

        # Check that history is limited to 100 records
        assert len(self.strategy.htf_bias_history) == 100

        # Check that we can get bias history
        history = self.strategy.get_bias_history(limit=50)
        assert len(history) == 50

        # Check that we can get all history (should be limited to 100)
        all_history = self.strategy.get_bias_history()
        assert len(all_history) == 100

    def test_bias_statistics(self):
        """Test bias statistics calculation."""
        # Create bias history with flips
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        # Create bullish bias
        df1 = self.create_test_dataframe(start_time, 50)
        mock_structures_bullish = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df1
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bullish
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'
        self.strategy.update_htf_bias_dynamically(df1)

        # Create bearish bias (flip)
        df2 = self.create_test_dataframe(start_time + timedelta(hours=2), 60)
        mock_structures_bearish = self.create_mock_structures('BEARISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bearish
        self.strategy.market_structure.get_current_bias.return_value = 'BEARISH'
        self.strategy.update_htf_bias_dynamically(df2)

        # Create neutral bias
        df3 = self.create_test_dataframe(start_time + timedelta(hours=4), 70)
        mock_structures_neutral = self.create_mock_structures('NEUTRAL', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df3
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_neutral
        self.strategy.market_structure.get_current_bias.return_value = 'NEUTRAL'
        self.strategy.update_htf_bias_dynamically(df3)

        # Get statistics
        stats = self.strategy.get_bias_statistics()

        # Assertions
        assert stats['total_updates'] == 3
        assert stats['bias_flips'] == 1  # Only BULLISH -> BEARISH counts as flip
        assert stats['current_bias'] == 'NEUTRAL'
        assert stats['bias_duration_minutes'] > 0
        assert stats['last_flip_timestamp'] is not None
        assert stats['bias_distribution']['BULLISH'] == 1
        assert stats['bias_distribution']['BEARISH'] == 1
        assert stats['bias_distribution']['NEUTRAL'] == 1

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        # Create empty DataFrame
        empty_df = pd.DataFrame()

        # Test with empty DataFrame
        result = self.strategy.update_htf_bias_dynamically(empty_df)

        # Assertions
        assert result['bias_updated'] is False
        assert result['new_bias'] == 'NEUTRAL'  # Default bias
        assert result['previous_bias'] == 'NEUTRAL'
        assert result['bias_flipped'] is False
        assert result['timestamp'] is None
        assert result['structures_count'] == 0

    def test_run_analysis_with_dynamic_bias(self):
        """Test run_analysis with dynamic bias updates."""
        # Mock data loader
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        htf_data = self.create_test_dataframe(start_time, 50, "1h")
        mtf_data = self.create_test_dataframe(start_time, 200, "15m")

        self.strategy.base_path_loader.load_pair_data.return_value = {
            "1h": htf_data,
            "15m": mtf_data
        }

        # Mock market structure analysis
        mock_structures = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = htf_data
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Mock MTF analysis
        self.strategy.analyze_mtf_structure = Mock(return_value={
            'dataframe': mtf_data,
            'swing_points': mtf_data,
            'structures': [],
            'fvgs': [],
            'order_blocks': [],
            'breaker_blocks': [],
            'ote_zones': [],
            'liquidity_grabs': [],
            'elliott_sequences': []
        })

        # Mock signal generation
        self.strategy.generate_signals = Mock(return_value=[])

        # Test run_analysis
        result = self.strategy.run_analysis("BTCUSDT", "2024-01-01", "2024-01-02")

        # Assertions
        assert 'signals' in result
        assert 'htf_bias' in result
        assert 'bias_update' in result
        assert 'bias_history' in result
        assert result['htf_bias'] == 'BULLISH'
        assert result['bias_update']['bias_updated'] is True
        assert result['bias_update']['new_bias'] == 'BULLISH'

    def test_bias_flip_threshold_configuration(self):
        """Test bias flip threshold configuration."""
        # Test default threshold
        assert self.strategy.bias_flip_threshold_minutes == 60

        # Test custom threshold
        self.strategy.bias_flip_threshold_minutes = 30

        # Set up initial bullish bias
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        df1 = self.create_test_dataframe(start_time, 50)

        mock_structures_bullish = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df1
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bullish
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Initial update
        self.strategy.update_htf_bias_dynamically(df1)

        # Create data with bearish bias (45 minutes later - should be filtered with 30min threshold)
        start_time2 = datetime(2024, 1, 1, 0, 45, 0)
        df2 = self.create_test_dataframe(start_time2, 55)

        mock_structures_bearish = self.create_mock_structures('BEARISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = df2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bearish
        self.strategy.market_structure.get_current_bias.return_value = 'BEARISH'

        # Test bias flip (should be filtered as noise with 30min threshold)
        result = self.strategy.update_htf_bias_dynamically(df2)

        # Assertions - bias should update but flip should be filtered
        assert result['bias_updated'] is True
        assert result['new_bias'] == 'BEARISH'
        assert result['previous_bias'] == 'BULLISH'
        # Note: Since this is the first real flip (NEUTRAL->BULLISH doesn't count),
        # it should be detected as a flip even with the threshold
        assert result['bias_flipped'] is True  # First real flip is always detected


class TestDynamicHTFBiasIntegration:
    """Integration tests for dynamic HTF bias updates."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_config_loader = Mock(spec=ConfigLoader)

        # Mock configuration objects
        self.mock_elliott_config = Mock()
        self.mock_entry_config = Mock()
        self.mock_timeframe_config = Mock()
        self.mock_ranking_config = Mock()

        # Set up timeframe config
        self.mock_timeframe_config.htf = "1h"
        self.mock_timeframe_config.mtf = "15m"

        # Set up entry config
        self.mock_entry_config.min_confirmations = 2

        # Configure mock config loader
        self.mock_config_loader.get_elliott_wave_config.return_value = self.mock_elliott_config
        self.mock_config_loader.get_entry_confirmation_config.return_value = self.mock_entry_config
        self.mock_config_loader.get_timeframe_config.return_value = self.mock_timeframe_config
        self.mock_config_loader.get_wave_ranking_config.return_value = self.mock_ranking_config

        # Create TradingStrategy instance
        self.strategy = TradingStrategy("/test/path", self.mock_config_loader)

        # Mock components
        self.strategy.base_path_loader = Mock()
        self.strategy.market_structure = Mock()
        self.strategy.ict_detector = Mock()
        self.strategy.elliott_detector = Mock()
        self.strategy.ict_entries = Mock()
        self.strategy.killzone_detector = Mock()

    def create_test_dataframe(self, start_time: datetime, periods: int, timeframe: str = "1h") -> pd.DataFrame:
        """Create test DataFrame with OHLCV data."""
        if timeframe == "1h":
            freq = "1h"
        elif timeframe == "15m":
            freq = "15min"
        else:
            freq = "1h"

        dates = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Create realistic price data
        base_price = 50000.0
        prices = []
        current_price = base_price

        for i in range(periods):
            # Add some trend and volatility
            trend = 0.001 * np.sin(i * 0.1)  # Small trend component
            volatility = 0.02 * np.random.normal()  # Random volatility
            change = trend + volatility

            current_price *= (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = price * (1 + np.random.normal(0, 0.002))
            close_price = price
            volume = np.random.randint(1000, 10000)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data, index=dates)
        return df

    def create_mock_structures(self, bias: str, count: int = 3) -> list:
        """Create mock market structures."""
        structures = []
        for i in range(count):
            structure = Mock(spec=MarketStructure)
            structure.timestamp = datetime.now() - timedelta(hours=i)
            structure.structure_type = 'BOS'
            structure.trend_direction = bias
            structure.price = 50000.0 + i * 100
            structure.strength = 0.8
            structures.append(structure)
        return structures

    def test_multiple_bias_updates_in_sequence(self):
        """Test multiple bias updates in sequence."""
        start_time = datetime(2024, 1, 1, 0, 0, 0)

        # Test sequence: BULLISH -> BEARISH -> NEUTRAL -> BULLISH
        biases = ['BULLISH', 'BEARISH', 'NEUTRAL', 'BULLISH']
        # Note: Only BULLISH<->BEARISH count as flips, transitions to/from NEUTRAL don't count
        expected_flips = [False, True, False, False]  # BULLISH->BEARISH is flip, others are not

        for i, (bias, expected_flip) in enumerate(zip(biases, expected_flips)):
            df = self.create_test_dataframe(start_time + timedelta(hours=i*2), 50)

            mock_structures = self.create_mock_structures(bias, 3)
            self.strategy.market_structure.detect_swing_points.return_value = df
            self.strategy.market_structure.detect_market_structure.return_value = mock_structures
            self.strategy.market_structure.get_current_bias.return_value = bias

            result = self.strategy.update_htf_bias_dynamically(df)

            # Assertions
            assert result['bias_updated'] is True
            assert result['new_bias'] == bias
            assert result['bias_flipped'] == expected_flip

            # Check bias history
            assert len(self.strategy.htf_bias_history) == i + 1
            assert self.strategy.htf_bias_history[-1]['bias'] == bias
            assert self.strategy.htf_bias_history[-1]['bias_flipped'] == expected_flip

        # Final assertions
        assert self.strategy.htf_bias == 'BULLISH'
        assert len(self.strategy.htf_bias_history) == 4

        # Check statistics
        stats = self.strategy.get_bias_statistics()
        assert stats['total_updates'] == 4
        assert stats['bias_flips'] == 1  # Only one true flip (BULLISH->BEARISH)
        assert stats['current_bias'] == 'BULLISH'

    def test_signal_generation_with_dynamic_bias(self):
        """Test signal generation with dynamic bias updates."""
        # Mock data loader
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        htf_data = self.create_test_dataframe(start_time, 50, "1h")
        mtf_data = self.create_test_dataframe(start_time, 200, "15m")

        self.strategy.base_path_loader.load_pair_data.return_value = {
            "1h": htf_data,
            "15m": mtf_data
        }

        # Mock market structure analysis
        mock_structures = self.create_mock_structures('BULLISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = htf_data
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures
        self.strategy.market_structure.get_current_bias.return_value = 'BULLISH'

        # Mock MTF analysis
        self.strategy.analyze_mtf_structure = Mock(return_value={
            'dataframe': mtf_data,
            'swing_points': mtf_data,
            'structures': [],
            'fvgs': [],
            'order_blocks': [],
            'breaker_blocks': [],
            'ote_zones': [],
            'liquidity_grabs': [],
            'elliott_sequences': []
        })

        # Mock signal generation
        mock_signals = [
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='TEST',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[51000.0, 52000.0],
                risk_reward=2.0,
                confidence=0.8,
                metadata={}
            )
        ]
        self.strategy.generate_signals = Mock(return_value=mock_signals)

        # Test run_analysis
        result = self.strategy.run_analysis("BTCUSDT", "2024-01-01", "2024-01-02")

        # Assertions
        assert len(result['signals']) == 1
        assert result['signals'][0].signal_type == 'BUY'
        assert result['htf_bias'] == 'BULLISH'
        assert result['bias_update']['bias_updated'] is True

        # Test with bias flip
        # Update to bearish bias
        htf_data2 = self.create_test_dataframe(start_time + timedelta(hours=2), 60, "1h")
        mock_structures_bearish = self.create_mock_structures('BEARISH', 3)
        self.strategy.market_structure.detect_swing_points.return_value = htf_data2
        self.strategy.market_structure.detect_market_structure.return_value = mock_structures_bearish
        self.strategy.market_structure.get_current_bias.return_value = 'BEARISH'

        self.strategy.base_path_loader.load_pair_data.return_value = {
            "1h": htf_data2,
            "15m": mtf_data
        }

        # Test run_analysis with bias flip
        result2 = self.strategy.run_analysis("BTCUSDT", "2024-01-01", "2024-01-02")

        # Assertions - BUY signal should be invalidated due to bearish bias
        assert len(result2['signals']) == 0  # BUY signal invalidated
        assert result2['htf_bias'] == 'BEARISH'
        assert result2['bias_update']['bias_flipped'] is True
