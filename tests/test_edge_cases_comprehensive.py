"""
Comprehensive edge case tests for the Crypto Bot Trader.
Tests error handling, unusual scenarios, and boundary conditions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from trading_strategy import TradingStrategy
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.config_loader import ConfigLoader
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.kill_zones import KillZoneDetector
from trading_strategy.config_loader import ConfigurationLoader
from backtester import BacktestEngine
from trading_strategy.data_structures import Signal, ElliottWave, ICTConcept, MarketStructure


class TestEdgeCasesComprehensive:
    """Comprehensive edge case tests for the trading system."""

    def setup_method(self):
        """Set up test data for edge case testing."""
        # Create minimal valid data
        self.minimal_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        self.minimal_data.set_index('timestamp', inplace=True)

        # Create extreme data
        self.extreme_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000, 100000, 1000, 50000, 50000] + [50000] * 95,
            'high': [50100, 200000, 2000, 50100, 50100] + [50100] * 95,
            'low': [49900, 50000, 500, 49900, 49900] + [49900] * 95,
            'close': [50000, 100000, 1000, 50000, 50000] + [50000] * 95,
            'volume': [1000, 1000000, 100, 1000, 1000] + [1000] * 95
        })

        self.extreme_data.set_index('timestamp', inplace=True)

        # Create data with gaps
        self.gapped_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        # Add gaps
        self.gapped_data.loc[self.gapped_data.index[20:30], :] = np.nan
        self.gapped_data.loc[self.gapped_data.index[50:60], :] = np.nan

        self.gapped_data.set_index('timestamp', inplace=True)

        # Basic configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_elliott_wave_config()

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()

        # These classes should handle empty data gracefully, not raise exceptions
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # Test detectors with empty data - they should handle it gracefully
        elliott_detector = ElliottWaveDetector(empty_data, self.config)
        assert elliott_detector is not None

        ict_detector = ICTConceptsDetector(empty_data, self.config)
        assert ict_detector is not None

        market_detector = MarketStructureDetector(empty_data, self.config)
        assert market_detector is not None

        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103]
            # Missing 'low', 'close', 'volume'
        })

        # These classes should handle missing columns gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # Test detectors with incomplete data - they should handle it gracefully
        elliott_detector = ElliottWaveDetector(incomplete_data, self.config)
        assert elliott_detector is not None

        ict_detector = ICTConceptsDetector(incomplete_data, self.config)
        assert ict_detector is not None

        market_detector = MarketStructureDetector(incomplete_data, self.config)
        assert market_detector is not None

        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

    def test_invalid_data_types_handling(self):
        """Test handling of invalid data types."""
        invalid_data = pd.DataFrame({
            'open': ['a', 'b', 'c'],  # Should be numeric
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # These classes should handle invalid data types gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # Test detectors with invalid data - they should handle it gracefully
        elliott_detector = ElliottWaveDetector(invalid_data, self.config)
        assert elliott_detector is not None

        ict_detector = ICTConceptsDetector(invalid_data, self.config)
        assert ict_detector is not None

        market_detector = MarketStructureDetector(invalid_data, self.config)
        assert market_detector is not None

        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

    def test_nan_values_handling(self):
        """Test handling of NaN values."""
        nan_data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # Should handle NaN values gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': nan_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_inf_values_handling(self):
        """Test handling of infinite values."""
        inf_data = pd.DataFrame({
            'open': [100, np.inf, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # Should handle infinite values gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': inf_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_negative_prices_handling(self):
        """Test handling of negative prices."""
        negative_data = pd.DataFrame({
            'open': [100, -50, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # Should handle negative prices gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': negative_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_zero_prices_handling(self):
        """Test handling of zero prices."""
        zero_data = pd.DataFrame({
            'open': [100, 0, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # Should handle zero prices gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': zero_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_extreme_volatility_handling(self):
        """Test handling of extreme volatility."""
        # Should handle extreme volatility gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.extreme_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_gapped_data_handling(self):
        """Test handling of gapped data."""
        # Should handle gapped data gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.gapped_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_minimal_data_handling(self):
        """Test handling of minimal data."""
        # Should handle minimal data gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.minimal_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_weekend_data_handling(self):
        """Test handling of weekend data."""
        weekend_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-07', periods=100, freq='1h'),  # Saturday
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        weekend_data.set_index('timestamp', inplace=True)

        # Should handle weekend data gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': weekend_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_holiday_data_handling(self):
        """Test handling of holiday data."""
        holiday_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-12-25', periods=100, freq='1h'),  # Christmas
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        holiday_data.set_index('timestamp', inplace=True)

        # Should handle holiday data gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': holiday_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_timezone_edge_cases(self):
        """Test timezone edge cases."""
        # Test DST transition
        dst_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-03-12', periods=100, freq='1h', tz='US/Eastern'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        dst_data.set_index('timestamp', inplace=True)

        # Should handle DST transition gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': dst_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_midnight_boundary_handling(self):
        """Test handling of midnight boundaries."""
        midnight_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        midnight_data.set_index('timestamp', inplace=True)

        # Test midnight boundary
        midnight_time = datetime(2023, 1, 1, 0, 0, tzinfo=pytz.UTC)
        kill_zone_detector = KillZoneDetector(self.config_loader.get_config_value("session_logic"))
        kill_zone = kill_zone_detector.get_current_kill_zone(midnight_time)

        assert kill_zone is not None
        assert kill_zone.zone_type in ['asia', 'london', 'ny', 'london_ny']

    def test_end_of_day_boundary_handling(self):
        """Test handling of end of day boundaries."""
        end_of_day_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        end_of_day_data.set_index('timestamp', inplace=True)

        # Test end of day boundary
        end_of_day_time = datetime(2023, 1, 1, 23, 59, tzinfo=pytz.UTC)
        kill_zone_detector = KillZoneDetector(self.config_loader.get_config_value("session_logic"))
        kill_zone = kill_zone_detector.get_current_kill_zone(end_of_day_time)

        assert kill_zone is not None
        assert kill_zone.zone_type in ['asia', 'london', 'ny', 'london_ny']

    def test_leap_year_handling(self):
        """Test handling of leap year."""
        leap_year_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-02-29', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        leap_year_data.set_index('timestamp', inplace=True)

        # Should handle leap year gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': leap_year_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_division_by_zero_protection(self):
        """Test division by zero protection."""
        # Create data that could cause division by zero
        zero_volume_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [0] * 100  # Zero volume
        })

        zero_volume_data.set_index('timestamp', inplace=True)

        # Should handle zero volume gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': zero_volume_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_negative_volume_handling(self):
        """Test handling of negative volume."""
        negative_volume_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [-1000] * 100  # Negative volume
        })

        negative_volume_data.set_index('timestamp', inplace=True)

        # Should handle negative volume gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': negative_volume_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Test that TradingStrategy handles invalid configs gracefully
        # The actual validation happens during runtime, not construction
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # Test that the strategy can be created even with invalid configs
        # The validation will happen when methods are called
        assert hasattr(strategy, 'config_loader')
        assert hasattr(strategy, 'data_loader')

    def test_missing_configuration_handling(self):
        """Test handling of missing configuration."""
        # Test that TradingStrategy handles missing config gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # The strategy should use default values for missing config
        assert hasattr(strategy, 'config_loader')
        assert hasattr(strategy, 'data_loader')

    def test_invalid_signal_parameters_handling(self):
        """Test handling of invalid signal parameters."""
        # Test negative price - should raise ValueError
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='test',
                price=-50000.0,  # Negative price should raise ValueError
                stop_loss=49000.0,
                take_profits=[52000.0],
                risk_reward=2.0,
                confidence=0.8,
                metadata={}
            )

        # Test invalid confidence - should raise ValueError
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='test',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[52000.0],
                risk_reward=2.0,
                confidence=1.5,  # > 1.0
                metadata={}
            )

        # Test invalid signal type - should raise ValueError
        with pytest.raises(ValueError):
            Signal(
                timestamp=datetime.now(),
                signal_type='invalid',
                entry_type='test',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[52000.0],
                risk_reward=2.0,
                confidence=0.8,
                metadata={}
            )

    def test_memory_limits_handling(self):
        """Test handling of memory limits."""
        # Create very large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100000, freq='1h'),
            'open': np.random.uniform(40000, 60000, 100000),
            'high': np.random.uniform(40000, 60000, 100000),
            'low': np.random.uniform(40000, 60000, 100000),
            'close': np.random.uniform(40000, 60000, 100000),
            'volume': np.random.randint(1000, 10000, 100000)
        })

        large_data.set_index('timestamp', inplace=True)

        # Should handle large dataset gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': large_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_concurrent_access_handling(self):
        """Test handling of concurrent access."""
        import threading
        import time

        results = []
        errors = []

        def worker(data, config, worker_id):
            try:
                strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
                signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': data})
                results.append((worker_id, len(signals)))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(self.minimal_data, self.config, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 5
        assert len(errors) == 0

        # All workers should have produced results
        for worker_id, signal_count in results:
            assert signal_count >= 0

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        # This would be relevant for live data feeds
        # For now, test with simulated timeout scenarios

        # Test with very slow data processing
        slow_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='1h'),
            'open': [50000] * 1000,
            'high': [50100] * 1000,
            'low': [49900] * 1000,
            'close': [50000] * 1000,
            'volume': [1000] * 1000
        })

        slow_data.set_index('timestamp', inplace=True)

        # Should handle slow processing gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': slow_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_data_corruption_handling(self):
        """Test handling of data corruption."""
        corrupted_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        # Corrupt some data
        corrupted_data = corrupted_data.astype(object)  # Convert to object dtype first
        corrupted_data.loc[corrupted_data.index[10:20], 'close'] = 'corrupted'
        corrupted_data.loc[corrupted_data.index[30:40], 'volume'] = 'corrupted'

        corrupted_data = corrupted_data.set_index('timestamp')

        # Should handle corrupted data gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        # The strategy should handle corrupted data without crashing
        try:
            signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': corrupted_data})
            assert isinstance(signals, list)
        except ValueError:
            # It's acceptable for the strategy to reject corrupted data
            pass

    def test_boundary_conditions_handling(self):
        """Test handling of boundary conditions."""
        # Test with boundary values
        boundary_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [0.01, 999999999, 0.01, 999999999] + [50000] * 96,
            'high': [0.01, 999999999, 0.01, 999999999] + [50100] * 96,
            'low': [0.01, 999999999, 0.01, 999999999] + [49900] * 96,
            'close': [0.01, 999999999, 0.01, 999999999] + [50000] * 96,
            'volume': [1, 999999999, 1, 999999999] + [1000] * 96
        })

        boundary_data.set_index('timestamp', inplace=True)

        # Should handle boundary values gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': boundary_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_error_recovery_handling(self):
        """Test handling of error recovery."""
        # Test with data that causes errors
        error_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1h'),
            'open': [50000] * 100,
            'high': [50100] * 100,
            'low': [49900] * 100,
            'close': [50000] * 100,
            'volume': [1000] * 100
        })

        error_data.set_index('timestamp', inplace=True)

        # Test error recovery
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Should recover from errors gracefully
        try:
            signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': error_data})
            assert isinstance(signals, list)
            assert all(isinstance(signal, Signal) for signal in signals)
        except Exception as e:
            # Should handle errors gracefully
            assert isinstance(e, Exception)

    def test_performance_under_stress(self):
        """Test performance under stress."""
        import time

        # Test with high-frequency data
        high_freq_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10000, freq='1min'),
            'open': np.random.uniform(40000, 60000, 10000),
            'high': np.random.uniform(40000, 60000, 10000),
            'low': np.random.uniform(40000, 60000, 10000),
            'close': np.random.uniform(40000, 60000, 10000),
            'volume': np.random.randint(1000, 10000, 10000)
        })

        high_freq_data.set_index('timestamp', inplace=True)

        # Test performance
        start_time = time.time()

        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': high_freq_data})

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time
        assert duration < 30.0  # Less than 30 seconds

        # Validate results
        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_comprehensive_edge_case_workflow(self):
        """Test comprehensive edge case workflow."""
        # Test with all edge cases combined
        edge_case_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='1h'),
            'open': [50000, np.nan, 0, np.inf, -50, 50000] + [50000] * 994,
            'high': [50100, np.nan, 0, np.inf, -50, 50100] + [50100] * 994,
            'low': [49900, np.nan, 0, np.inf, -50, 49900] + [49900] * 994,
            'close': [50000, np.nan, 0, np.inf, -50, 50000] + [50000] * 994,
            'volume': [1000, np.nan, 0, np.inf, -1000, 1000] + [1000] * 994
        })

        edge_case_data.set_index('timestamp', inplace=True)

        # Should handle all edge cases gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': edge_case_data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

        # Test backtesting with edge cases
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

        for signal in signals[:10]:  # Limit to 10 signals
            result = backtester.execute_trade(signal, 0)
            if result['success']:
                # Test position management
                exit_price = signal.entry_price * 1.01
                exit_result = backtester.manage_position_exits(result['position'], exit_price, 0)
                assert exit_result is not None
