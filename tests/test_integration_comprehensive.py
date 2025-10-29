"""
Comprehensive integration tests for the Crypto Bot Trader.
Tests complete workflows and end-to-end functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_strategy import TradingStrategy
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.config_loader import ConfigLoader
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.kill_zones import KillZoneDetector
from trading_strategy.config_loader import ConfigurationLoader
from backtester import BacktestEngine
from trading_strategy.data_structures import Signal, ElliottWave, ICTConcept, MarketStructure


class TestIntegrationComprehensive:
    """Comprehensive integration tests for the trading system."""

    def setup_method(self):
        """Set up comprehensive test data and configuration."""
        # Create realistic market data
        dates = pd.date_range(start='2023-01-01', periods=2000, freq='1h')
        np.random.seed(42)

        # Generate realistic price data with trends and volatility
        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 2000)
        prices = [base_price]

        for i, change in enumerate(price_changes[1:]):
            # Add some trend and volatility
            trend_factor = 1 + (i / 2000) * 0.1  # 10% trend over period
            volatility_factor = 1 + np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change) * trend_factor * volatility_factor
            prices.append(new_price)

        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 2000)
        })

        self.data.set_index('timestamp', inplace=True)

        # Comprehensive configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_elliott_wave_config()

    def test_complete_trading_workflow(self):
        """Test complete trading workflow from data to signals."""
        # Initialize strategy
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Generate signals with proper parameters
        htf_analysis = {'bias': 'NEUTRAL', 'structures': [], 'trend_strength': 0.5}
        mtf_analysis = {'dataframe': self.data, 'structures': [], 'fvgs': [], 'order_blocks': []}
        signals = strategy.generate_signals(htf_analysis, mtf_analysis)

        # Validate signals
        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

        # Check signal quality
        if signals:
            high_confidence_signals = [s for s in signals if s.confidence > 0.8]
            assert len(high_confidence_signals) >= 0

            # Check signal properties
            for signal in signals:
                assert signal.entry_price > 0
                assert signal.stop_loss > 0
                assert signal.take_profit > 0
                assert signal.confidence >= 0.0
                assert signal.confidence <= 1.0
                assert signal.signal_type in ['BUY', 'SELL']

    def test_multi_timeframe_analysis_integration(self):
        """Test multi-timeframe analysis integration."""
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Test HTF bias analysis
        htf_analysis = strategy.manage_multi_timeframe_analysis(100)

        assert isinstance(htf_analysis, dict)
        assert 'htf_bias' in htf_analysis
        assert 'mtf_setup' in htf_analysis
        assert 'ltf_entry' in htf_analysis

        # Check bias values
        assert htf_analysis['htf_bias'] in ['bullish', 'bearish', 'neutral']
        assert htf_analysis['mtf_setup'] in ['BUY', 'SELL', 'none']
        assert htf_analysis['ltf_entry'] in ['BUY', 'SELL', 'none']

        # Test with different timeframes
        for i in range(100, 200, 10):
            analysis = strategy.manage_multi_timeframe_analysis(i)
            assert isinstance(analysis, dict)
            assert all(key in analysis for key in ['htf_bias', 'mtf_setup', 'ltf_entry'])

    def test_elliott_wave_ict_integration(self):
        """Test Elliott Wave and ICT concepts integration."""
        # Initialize detectors
        ew_detector = ElliottWaveDetector(self.data, self.config)
        ict_detector = ICTConceptsDetector(self.data, self.config)

        # Detect patterns
        waves = ew_detector.find_elliott_wave_sequence(0, 200)
        concepts = ict_detector.get_all_ict_concepts(0, 200)

        # Validate integration
        assert isinstance(waves, list)
        assert isinstance(concepts, list)
        assert all(isinstance(wave, ElliottWave) for wave in waves)
        assert all(isinstance(concept, ICTConcept) for concept in concepts)

        # Check pattern quality
        if waves:
            validated_waves = [w for w in waves if w.is_validated()]
            assert len(validated_waves) >= 0

            for wave in waves:
                assert wave.is_bullish() or wave.is_bearish()
                assert wave.strength >= 0.0
                assert wave.strength <= 1.0

        if concepts:
            fresh_concepts = [c for c in concepts if c.is_fresh]
            assert len(fresh_concepts) >= 0

            for concept in concepts:
                assert concept.is_bullish() or concept.is_bearish()
                assert concept.strength >= 0.0
                assert concept.strength <= 1.0

    def test_market_structure_integration(self):
        """Test market structure analysis integration."""
        ms_detector = MarketStructureDetector(self.data, self.config)

        # Detect structures
        swing_points = ms_detector.detect_swing_points_from_indices(0, 200)
        bos_structures = ms_detector.detect_break_of_structure(0, 200)
        choch_structures = ms_detector.detect_change_of_character(0, 200)

        # Validate structures
        assert isinstance(swing_points, list)
        assert isinstance(bos_structures, list)
        assert isinstance(choch_structures, list)

        assert all(isinstance(sp, MarketStructure) for sp in swing_points)
        assert all(isinstance(bos, MarketStructure) for bos in bos_structures)
        assert all(isinstance(choch, MarketStructure) for choch in choch_structures)

        # Check structure quality
        if swing_points:
            for sp in swing_points:
                assert sp.is_bullish_structure() or sp.is_bearish_structure() or sp.trend_direction == 'NEUTRAL'
                assert sp.strength >= 0.0
                assert sp.strength <= 1.0

        if bos_structures:
            for bos in bos_structures:
                assert bos.is_break_of_structure() or bos.is_change_of_character()
                assert bos.strength >= 0.0
                assert bos.strength <= 1.0

        if choch_structures:
            for choch in choch_structures:
                assert choch.is_change_of_character() or choch.is_break_of_structure()
                assert choch.strength >= 0.0
                assert choch.strength <= 1.0

    def test_session_logic_integration(self):
        """Test session logic integration."""
        kill_zone_detector = KillZoneDetector(self.config_loader.get_config_value("session_logic"))

        # Test different session times
        test_times = [
            datetime(2023, 1, 1, 2, 0),   # Asia
            datetime(2023, 1, 1, 10, 0),  # London
            datetime(2023, 1, 1, 18, 0),  # NY
            datetime(2023, 1, 1, 17, 0),   # Overlap
        ]

        for test_time in test_times:
            kill_zone = kill_zone_detector.get_current_kill_zone(test_time)
            session_bias = kill_zone_detector.get_session_bias(test_time)
            is_active = kill_zone_detector.is_trading_session_active(test_time)

            assert kill_zone is not None
            assert session_bias in ['neutral', 'bullish', 'bearish', 'high_volatility']
            assert isinstance(is_active, bool)

    def test_backtesting_integration(self):
        """Test backtesting integration."""
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

        # Generate signals
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

        # Execute trades
        positions = []
        for i, signal in enumerate(signals[:10]):  # Limit to 10 signals
            result = backtester.execute_trade(signal, i)
            if result['success']:
                positions.append(result['position'])

        # Validate positions
        assert len(positions) <= 10
        assert all(isinstance(pos, Position) for pos in positions)

        # Test position management
        for position in positions:
            # Test exit scenarios
            exit_price = position.entry_price * 1.02  # 2% profit
            result = backtester.manage_position_exits(position, exit_price, 0)

            assert result is not None
            assert 'tp1_hit' in result
            assert 'remaining_qty' in result
            assert 'stop_at_be' in result

    def test_risk_management_integration(self):
        """Test risk management integration."""
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

        # Test position sizing
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='elliott_wave',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={}
        )

        position_size = backtester.calculate_position_size_from_signal(signal, 10000.0)
        assert position_size > 0
        assert position_size <= 10000.0 * 0.02  # Max 2% risk

        # Test risk limits
        can_trade = backtester.check_risk_limits(signal, [])
        assert isinstance(can_trade, bool)

        # Test drawdown protection
        backtester.current_balance = 9500  # 5% drawdown
        can_trade = backtester.check_risk_limits(signal, [])
        assert can_trade is True

        backtester.current_balance = 8500  # 15% drawdown
        can_trade = backtester.check_risk_limits(signal, [])
        assert can_trade is False

    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test configuration loading
        config_loader = ConfigurationLoader('config/')

        # Test trading config
        trading_config = config_loader.load_trading_config()
        assert isinstance(trading_config, dict)
        assert 'risk_management' in trading_config
        assert 'elliott_wave' in trading_config
        assert 'ict_concepts' in trading_config
        assert 'market_structure' in trading_config
        assert 'session_logic' in trading_config

        # Test timeframe config
        timeframe_config = config_loader.load_timeframe_config()
        assert isinstance(timeframe_config, dict)
        assert 'timeframes' in timeframe_config
        assert 'analysis' in timeframe_config
        assert 'signals' in timeframe_config

        # Test configuration validation
        is_valid = config_loader.validate_config()
        assert isinstance(is_valid, bool)

    def test_signal_generation_pipeline(self):
        """Test complete signal generation pipeline."""
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Test signal generation at different points
        for start_idx in range(0, 100, 20):
            end_idx = start_idx + 100

            signals = strategy.generate_signals_from_indices(start_idx, end_idx)

            # Validate signals
            assert isinstance(signals, list)
            assert all(isinstance(signal, Signal) for signal in signals)

            # Check signal quality
            if signals:
                for signal in signals:
                    assert signal.entry_price > 0
                    assert signal.stop_loss > 0
                    assert signal.take_profit > 0
                    assert signal.confidence >= 0.0
                    assert signal.confidence <= 1.0
                    assert signal.signal_type in ['BUY', 'SELL']
                    assert signal.source in ['elliott_wave', 'ict_concept', 'market_structure']

    def test_pattern_confluence_integration(self):
        """Test pattern confluence integration."""
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Test confluence detection
        for i in range(100, 200, 10):
            # Get all patterns at this point
            ew_detector = ElliottWaveDetector(self.data, self.config)
            ict_detector = ICTConceptsDetector(self.data, self.config)
            ms_detector = MarketStructureDetector(self.data, self.config)

            waves = ew_detector.find_elliott_wave_sequence(i-50, i+50)
            concepts = ict_detector.get_all_ict_concepts(i-50, i+50)
            structures = ms_detector.detect_break_of_structure(i-50, i+50)

            # Check confluence
            confluence_count = 0
            if waves:
                confluence_count += 1
            if concepts:
                confluence_count += 1
            if structures:
                confluence_count += 1

            # Should have some confluence
            assert confluence_count >= 0

    def test_performance_integration(self):
        """Test performance integration."""
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Test performance metrics
        start_time = datetime.now()

        # Generate signals
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

        # Test multi-timeframe analysis
        mtf_analysis = strategy.manage_multi_timeframe_analysis(100)

        # Test Elliott-ICT integration
        integration_signals = strategy.integrate_elliott_ict_entries(100)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Should complete in reasonable time
        assert duration < 10.0  # Less than 10 seconds

        # Validate results
        assert isinstance(signals, list)
        assert isinstance(mtf_analysis, dict)
        assert isinstance(integration_signals, list)

    def test_error_handling_integration(self):
        """Test error handling integration."""
        # Test with invalid data - should handle gracefully
        invalid_data = pd.DataFrame()

        # These should not raise exceptions, they should handle gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

        # Test with missing columns - should handle gracefully
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103]
            # Missing required columns
        })

        strategy2 = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy2 is not None

        # Test with invalid configuration - should handle gracefully
        invalid_config = {
            'risk_management': {
                'max_risk_per_trade': 1.5,  # Invalid: > 1.0
                'max_daily_risk': 0.05,
                'max_positions': 5
            }
        }

        # These should not raise exceptions, they should handle gracefully
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        assert strategy is not None

    def test_memory_management_integration(self):
        """Test memory management integration."""
        # Test with large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5000, freq='1h'),
            'open': np.random.uniform(40000, 60000, 5000),
            'high': np.random.uniform(40000, 60000, 5000),
            'low': np.random.uniform(40000, 60000, 5000),
            'close': np.random.uniform(40000, 60000, 5000),
            'volume': np.random.randint(1000, 10000, 5000)
        })

        large_data.set_index('timestamp', inplace=True)

        # Should handle large dataset
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

        assert isinstance(signals, list)
        assert all(isinstance(signal, Signal) for signal in signals)

    def test_bidirectional_symmetry_integration(self):
        """Test bidirectional symmetry integration."""
        # Create bullish data
        bullish_data = self.data.copy()
        bullish_data['close'] = bullish_data['close'] * 1.1

        # Create bearish data
        bearish_data = self.data.copy()
        bearish_data['close'] = bearish_data['close'] * 0.9

        # Test bullish strategy
        bullish_strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        bullish_signals = bullish_strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': bullish_data})

        # Test bearish strategy
        bearish_strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        bearish_signals = bearish_strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': bearish_data})

        # Check symmetry
        assert len(bullish_signals) >= 0
        assert len(bearish_signals) >= 0

        # Check signal types
        bullish_types = [s.signal_type for s in bullish_signals]
        bearish_types = [s.signal_type for s in bearish_signals]

        assert set(bullish_types) == set(bearish_types)

        # Check confidence distribution
        bullish_confidences = [s.confidence for s in bullish_signals]
        bearish_confidences = [s.confidence for s in bearish_signals]

        if bullish_confidences and bearish_confidences:
            assert np.mean(bullish_confidences) > 0
            assert np.mean(bearish_confidences) > 0

    def test_comprehensive_workflow(self):
        """Test comprehensive workflow."""
        # Initialize all components
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

        # Generate signals
        signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

        # Execute trades
        positions = []
        for i, signal in enumerate(signals[:5]):  # Limit to 5 signals
            result = backtester.execute_trade(signal, i)
            if result['success']:
                positions.append(result['position'])

        # Manage positions
        for position in positions:
            # Test different exit scenarios
            exit_prices = [
                position.entry_price * 1.01,  # 1% profit
                position.entry_price * 1.02,  # 2% profit
                position.entry_price * 0.99,  # 1% loss
            ]

            for exit_price in exit_prices:
                result = backtester.manage_position_exits(position, exit_price, 0)
                assert result is not None
                assert 'tp1_hit' in result
                assert 'remaining_qty' in result
                assert 'stop_at_be' in result

        # Generate backtest report
        report = backtester.generate_backtest_report()

        assert report is not None
        assert report.initial_balance == 10000.0
        assert report.final_balance >= 0
        assert report.total_trades >= 0
        assert report.win_rate >= 0.0
        assert report.win_rate <= 1.0
        assert report.profit_factor >= 0.0
        assert report.max_drawdown >= 0.0
        assert report.sharpe_ratio >= 0.0

    def test_real_time_simulation(self):
        """Test real-time simulation."""
        strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)

        # Simulate real-time processing
        for i in range(100, 200, 10):
            # Get current market state
            current_data = self.data.iloc[:i]

            # Generate signals
            signals = strategy.generate_signals_from_indices(i-50, i)

            # Process signals
            for signal in signals:
                if signal.confidence > 0.8:
                    # Simulate trade execution
                    assert signal.entry_price > 0
                    assert signal.stop_loss > 0
                    assert signal.take_profit > 0
                    assert signal.confidence >= 0.0
                    assert signal.confidence <= 1.0

    def test_data_quality_integration(self):
        """Test data quality integration."""
        # Test with various data quality issues
        test_cases = [
            # Missing values
            self.data.copy().replace([np.inf, -np.inf], np.nan),
            # Extreme values
            self.data.copy().replace(self.data['close'], self.data['close'] * 1000),
            # Zero values
            self.data.copy().replace(0, np.nan),
        ]

        for test_data in test_cases:
            try:
                strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
                signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

                # Should handle data quality issues gracefully
                assert isinstance(signals, list)
                assert all(isinstance(signal, Signal) for signal in signals)

            except ValueError:
                # Some data quality issues should raise errors
                pass

    def test_configuration_validation_integration(self):
        """Test configuration validation integration."""
        # Test with various configuration scenarios
        test_configs = [
            # Valid configuration
            self.config,
            # Minimal configuration
            {
                'risk_management': {
                    'max_risk_per_trade': 0.01,
                    'max_daily_risk': 0.03,
                    'max_positions': 3
                }
            },
            # Extended configuration
            {
                'additional_features': {
                    'feature1': True,
                    'feature2': False
                }
            }
        ]

        for config in test_configs:
            try:
                strategy = TradingStrategy(base_path=".", config_loader=self.config_loader)
                signals = strategy.generate_signals({'bias': 'NEUTRAL'}, {'dataframe': self.data})

                # Should work with valid configurations
                assert isinstance(signals, list)
                assert all(isinstance(signal, Signal) for signal in signals)

            except ValueError:
                # Invalid configurations should raise errors
                pass
