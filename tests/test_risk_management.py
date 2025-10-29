"""
Comprehensive tests for portfolio-level risk management features.

Tests:
- Correlation checks vs active positions
- Trade frequency limits (min time between trades, max trades/day)
- Stop loss cooldown periods
- Integration with existing risk management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from trading_strategy.data_structures import Signal, PositionState
from trading_strategy.config_loader import ConfigLoader
from backtester import BacktestEngine


class TestCorrelationRiskManagement:
    """Test correlation-based risk management."""

    def setup_method(self):
        """Set up test environment."""
        self.backtester = BacktestEngine(
            base_path="/test/path",
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Mock config loader with proper dataclass structure
        from trading_strategy.config_loader import CorrelationConfig, FrequencyLimitsConfig, RiskManagementConfig

        correlation_config = CorrelationConfig(
            window_days=30,
            threshold=0.7,
            enabled=True
        )

        frequency_config = FrequencyLimitsConfig(
            min_time_between_trades_minutes=30,
            max_trades_per_day=10,
            stop_loss_cooldown_hours=2,
            enabled=True
        )

        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            max_concurrent_positions=5,
            atr_period=14,
            volatility_factor=True,
            slippage_percent=0.0005,
            tp1_percent=0.3,
            tp2_percent=0.4,
            tp3_percent=0.3,
            stop_to_breakeven=True,
            stop_to_tp1=True,
            max_drawdown_percent=0.15,
            drawdown_recovery_percent=0.05,
            correlation=correlation_config,
            frequency_limits=frequency_config
        )

        self.backtester.config_loader = Mock()
        self.backtester.config_loader.get_risk_management_config.return_value = risk_config
        self.backtester.config_loader.get_session_config.return_value = Mock()

        # Initialize risk config
        self.backtester.risk_config = risk_config
        self.backtester.session_config = self.backtester.config_loader.get_session_config()

        # Set up parameters
        self.backtester.correlation_window_days = risk_config.correlation.window_days
        self.backtester.correlation_threshold = risk_config.correlation.threshold
        self.backtester.correlation_enabled = risk_config.correlation.enabled

        self.backtester.min_time_between_trades_minutes = risk_config.frequency_limits.min_time_between_trades_minutes
        self.backtester.max_trades_per_day = risk_config.frequency_limits.max_trades_per_day
        self.backtester.stop_loss_cooldown_hours = risk_config.frequency_limits.stop_loss_cooldown_hours
        self.backtester.frequency_limits_enabled = risk_config.frequency_limits.enabled

    def test_correlation_check_no_active_positions(self):
        """Test correlation check passes when no active positions."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # No active positions
        self.backtester.active_positions = []

        result = self.backtester._check_correlation_limits(signal)
        assert result is True

    def test_correlation_check_low_correlation(self):
        """Test correlation check passes with low correlation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Set up active positions
        self.backtester.active_positions = ['ETHUSDT']

        # Mock low correlation
        with patch.object(self.backtester, '_calculate_correlation', return_value=0.3):
            result = self.backtester._check_correlation_limits(signal)
            assert result is True

    def test_correlation_check_high_correlation(self):
        """Test correlation check fails with high correlation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Set up active positions
        self.backtester.active_positions = ['ETHUSDT']

        # Mock high correlation
        with patch.object(self.backtester, '_calculate_correlation', return_value=0.8):
            result = self.backtester._check_correlation_limits(signal)
            assert result is False

    def test_correlation_calculation(self):
        """Test correlation calculation between pairs."""
        # Set up price history for two pairs
        base_time = datetime.now()

        # Create highly correlated price data
        prices1 = []
        prices2 = []

        for i in range(100):
            timestamp = base_time + timedelta(hours=i)
            # BTC price with trend
            price1 = 50000 + i * 10
            # ETH price highly correlated with BTC (same trend direction)
            price2 = 3000 + i * 0.6

            prices1.append((timestamp, price1))
            prices2.append((timestamp, price2))

        self.backtester.price_history['BTCUSDT'] = prices1
        self.backtester.price_history['ETHUSDT'] = prices2

        correlation = self.backtester._calculate_correlation('BTCUSDT', 'ETHUSDT')

        # Should have high correlation (close to 1.0 for perfect correlation)
        assert correlation > 0.8
        assert correlation <= 1.0

    def test_correlation_calculation_insufficient_data(self):
        """Test correlation calculation with insufficient data."""
        # Set up minimal price history
        base_time = datetime.now()
        self.backtester.price_history['BTCUSDT'] = [(base_time, 50000)]
        self.backtester.price_history['ETHUSDT'] = [(base_time, 3000)]

        correlation = self.backtester._calculate_correlation('BTCUSDT', 'ETHUSDT')

        # Should return 0.0 for insufficient data
        assert correlation == 0.0

    def test_price_history_update(self):
        """Test price history update and window management."""
        pair = 'BTCUSDT'
        base_time = datetime.now()

        # Add prices over 35 days (should keep only last 30)
        for i in range(35):
            timestamp = base_time + timedelta(days=i)
            price = 50000 + i * 100
            self.backtester._update_price_history(pair, timestamp, price)

        # Should have approximately 30 days of data (window management keeps >= cutoff_time)
        assert len(self.backtester.price_history[pair]) >= 30
        assert len(self.backtester.price_history[pair]) <= 31  # Allow for boundary condition

        # Check that oldest entry is from day 4 or 5 (35 - 30, with boundary tolerance)
        oldest_timestamp = self.backtester.price_history[pair][0][0]
        expected_oldest_min = base_time + timedelta(days=4)
        expected_oldest_max = base_time + timedelta(days=5)
        assert expected_oldest_min <= oldest_timestamp <= expected_oldest_max


class TestTradeFrequencyLimits:
    """Test trade frequency limit management."""

    def setup_method(self):
        """Set up test environment."""
        self.backtester = BacktestEngine(
            base_path="/test/path",
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Mock config loader with proper dataclass structure
        from trading_strategy.config_loader import CorrelationConfig, FrequencyLimitsConfig, RiskManagementConfig

        correlation_config = CorrelationConfig(
            window_days=30,
            threshold=0.7,
            enabled=True
        )

        frequency_config = FrequencyLimitsConfig(
            min_time_between_trades_minutes=30,
            max_trades_per_day=10,
            stop_loss_cooldown_hours=2,
            enabled=True
        )

        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            max_concurrent_positions=5,
            atr_period=14,
            volatility_factor=True,
            slippage_percent=0.0005,
            tp1_percent=0.3,
            tp2_percent=0.4,
            tp3_percent=0.3,
            stop_to_breakeven=True,
            stop_to_tp1=True,
            max_drawdown_percent=0.15,
            drawdown_recovery_percent=0.05,
            correlation=correlation_config,
            frequency_limits=frequency_config
        )

        self.backtester.config_loader = Mock()
        self.backtester.config_loader.get_risk_management_config.return_value = risk_config
        self.backtester.config_loader.get_session_config.return_value = Mock()

        # Initialize risk config
        self.backtester.risk_config = risk_config
        self.backtester.session_config = self.backtester.config_loader.get_session_config()

        # Set up parameters
        self.backtester.correlation_window_days = risk_config.correlation.window_days
        self.backtester.correlation_threshold = risk_config.correlation.threshold
        self.backtester.correlation_enabled = risk_config.correlation.enabled

        self.backtester.min_time_between_trades_minutes = risk_config.frequency_limits.min_time_between_trades_minutes
        self.backtester.max_trades_per_day = risk_config.frequency_limits.max_trades_per_day
        self.backtester.stop_loss_cooldown_hours = risk_config.frequency_limits.stop_loss_cooldown_hours
        self.backtester.frequency_limits_enabled = risk_config.frequency_limits.enabled

    def test_min_time_between_trades_pass(self):
        """Test minimum time between trades passes."""
        pair = 'BTCUSDT'
        base_time = datetime.now()

        # First trade
        signal1 = Signal(
            timestamp=base_time,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Second trade after sufficient time
        signal2 = Signal(
            timestamp=base_time + timedelta(minutes=35),
            signal_type='BUY',
            entry_type='TEST',
            price=102.0,
            stop_loss=97.0,
            take_profits=[107.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Record first trade
        self.backtester.last_trade_times[pair] = base_time

        result = self.backtester._check_trade_frequency_limits(signal2)
        assert result is True

    def test_min_time_between_trades_fail(self):
        """Test minimum time between trades fails."""
        pair = 'BTCUSDT'
        base_time = datetime.now()

        # First trade
        signal1 = Signal(
            timestamp=base_time,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Second trade too soon
        signal2 = Signal(
            timestamp=base_time + timedelta(minutes=15),
            signal_type='BUY',
            entry_type='TEST',
            price=102.0,
            stop_loss=97.0,
            take_profits=[107.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Record first trade
        self.backtester.last_trade_times[pair] = base_time

        result = self.backtester._check_trade_frequency_limits(signal2)
        assert result is False

    def test_max_trades_per_day_pass(self):
        """Test maximum trades per day passes."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Set daily count below limit
        current_date = signal.timestamp.date()
        self.backtester.daily_trade_counts[str(current_date)] = 5

        result = self.backtester._check_trade_frequency_limits(signal)
        assert result is True

    def test_max_trades_per_day_fail(self):
        """Test maximum trades per day fails."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Set daily count at limit
        current_date = signal.timestamp.date()
        self.backtester.daily_trade_counts[str(current_date)] = 10

        result = self.backtester._check_trade_frequency_limits(signal)
        assert result is False

    def test_stop_loss_cooldown_pass(self):
        """Test stop loss cooldown passes."""
        pair = 'BTCUSDT'
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # No cooldown set
        result = self.backtester._check_stop_loss_cooldown(signal)
        assert result is True

    def test_stop_loss_cooldown_fail(self):
        """Test stop loss cooldown fails."""
        pair = 'BTCUSDT'
        base_time = datetime.now()

        signal = Signal(
            timestamp=base_time,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Set cooldown that hasn't expired
        cooldown_end = base_time + timedelta(hours=1)
        self.backtester.stop_loss_cooldowns[pair] = cooldown_end

        result = self.backtester._check_stop_loss_cooldown(signal)
        assert result is False

    def test_stop_loss_cooldown_expired(self):
        """Test stop loss cooldown passes after expiration."""
        pair = 'BTCUSDT'
        base_time = datetime.now()

        signal = Signal(
            timestamp=base_time + timedelta(hours=3),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        # Set cooldown that has expired
        cooldown_end = base_time + timedelta(hours=2)
        self.backtester.stop_loss_cooldowns[pair] = cooldown_end

        result = self.backtester._check_stop_loss_cooldown(signal)
        assert result is True


class TestTradeTracking:
    """Test trade tracking functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.backtester = BacktestEngine(
            base_path="/test/path",
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Mock config loader with proper dataclass structure
        from trading_strategy.config_loader import CorrelationConfig, FrequencyLimitsConfig, RiskManagementConfig

        correlation_config = CorrelationConfig(
            window_days=30,
            threshold=0.7,
            enabled=True
        )

        frequency_config = FrequencyLimitsConfig(
            min_time_between_trades_minutes=30,
            max_trades_per_day=10,
            stop_loss_cooldown_hours=2,
            enabled=True
        )

        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            max_concurrent_positions=5,
            atr_period=14,
            volatility_factor=True,
            slippage_percent=0.0005,
            tp1_percent=0.3,
            tp2_percent=0.4,
            tp3_percent=0.3,
            stop_to_breakeven=True,
            stop_to_tp1=True,
            max_drawdown_percent=0.15,
            drawdown_recovery_percent=0.05,
            correlation=correlation_config,
            frequency_limits=frequency_config
        )

        self.backtester.config_loader = Mock()
        self.backtester.config_loader.get_risk_management_config.return_value = risk_config
        self.backtester.config_loader.get_session_config.return_value = Mock()

        # Initialize risk config
        self.backtester.risk_config = risk_config
        self.backtester.session_config = self.backtester.config_loader.get_session_config()

        # Set up parameters
        self.backtester.stop_loss_cooldown_hours = risk_config.frequency_limits.stop_loss_cooldown_hours

    def test_update_trade_tracking_normal_trade(self):
        """Test trade tracking for normal trade."""
        pair = 'BTCUSDT'
        timestamp = datetime.now()

        signal = Signal(
            timestamp=timestamp,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        trade_result = {
            'final_balance': 10100,
            'journal_entry': {
                'exit_reason': 'TAKE_PROFIT_1',
                'position_state': PositionState(
                    entry_qty=1000,
                    remaining_qty=0,  # Position closed
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profits=[105.0]
                )
            }
        }

        self.backtester._update_trade_tracking(signal, trade_result)

        # Check tracking updates
        assert self.backtester.last_trade_times[pair] == timestamp
        assert self.backtester.daily_trade_counts[str(timestamp.date())] == 1
        assert pair not in self.backtester.active_positions  # Position closed
        assert pair not in self.backtester.stop_loss_cooldowns  # No stop loss

    def test_update_trade_tracking_stop_loss(self):
        """Test trade tracking for stop loss trade."""
        pair = 'BTCUSDT'
        timestamp = datetime.now()

        signal = Signal(
            timestamp=timestamp,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        trade_result = {
            'final_balance': 9900,
            'journal_entry': {
                'exit_reason': 'STOP_LOSS',
                'position_state': PositionState(
                    entry_qty=1000,
                    remaining_qty=0,  # Position closed
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profits=[105.0]
                )
            }
        }

        self.backtester._update_trade_tracking(signal, trade_result)

        # Check tracking updates
        assert self.backtester.last_trade_times[pair] == timestamp
        assert self.backtester.daily_trade_counts[str(timestamp.date())] == 1
        assert pair not in self.backtester.active_positions  # Position closed

        # Check cooldown set
        expected_cooldown_end = timestamp + timedelta(hours=2)
        assert self.backtester.stop_loss_cooldowns[pair] == expected_cooldown_end

    def test_update_trade_tracking_partial_exit(self):
        """Test trade tracking for partial exit."""
        pair = 'BTCUSDT'
        timestamp = datetime.now()

        signal = Signal(
            timestamp=timestamp,
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': pair}
        )

        trade_result = {
            'final_balance': 10050,
            'journal_entry': {
                'exit_reason': 'TAKE_PROFIT_1',
                'position_state': PositionState(
                    entry_qty=1000,
                    remaining_qty=700,  # Partial exit
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profits=[105.0, 110.0]
                )
            }
        }

        self.backtester._update_trade_tracking(signal, trade_result)

        # Check tracking updates
        assert self.backtester.last_trade_times[pair] == timestamp
        assert self.backtester.daily_trade_counts[str(timestamp.date())] == 1
        assert pair in self.backtester.active_positions  # Position still open
        assert pair not in self.backtester.stop_loss_cooldowns  # No stop loss


class TestRiskManagementIntegration:
    """Test integration of all risk management features."""

    def setup_method(self):
        """Set up test environment."""
        self.backtester = BacktestEngine(
            base_path="/test/path",
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Mock config loader with proper dataclass structure
        from trading_strategy.config_loader import CorrelationConfig, FrequencyLimitsConfig, RiskManagementConfig

        correlation_config = CorrelationConfig(
            window_days=30,
            threshold=0.7,
            enabled=True
        )

        frequency_config = FrequencyLimitsConfig(
            min_time_between_trades_minutes=30,
            max_trades_per_day=10,
            stop_loss_cooldown_hours=2,
            enabled=True
        )

        risk_config = RiskManagementConfig(
            max_risk_per_trade=0.02,
            max_daily_risk=0.05,
            max_concurrent_positions=5,
            atr_period=14,
            volatility_factor=True,
            slippage_percent=0.0005,
            tp1_percent=0.3,
            tp2_percent=0.4,
            tp3_percent=0.3,
            stop_to_breakeven=True,
            stop_to_tp1=True,
            max_drawdown_percent=0.15,
            drawdown_recovery_percent=0.05,
            correlation=correlation_config,
            frequency_limits=frequency_config
        )

        self.backtester.config_loader = Mock()
        self.backtester.config_loader.get_risk_management_config.return_value = risk_config
        self.backtester.config_loader.get_session_config.return_value = Mock()

        # Initialize risk config
        self.backtester.risk_config = risk_config
        self.backtester.session_config = self.backtester.config_loader.get_session_config()

        # Set up all parameters
        self.backtester.correlation_window_days = risk_config.correlation.window_days
        self.backtester.correlation_threshold = risk_config.correlation.threshold
        self.backtester.correlation_enabled = risk_config.correlation.enabled

        self.backtester.min_time_between_trades_minutes = risk_config.frequency_limits.min_time_between_trades_minutes
        self.backtester.max_trades_per_day = risk_config.frequency_limits.max_trades_per_day
        self.backtester.stop_loss_cooldown_hours = risk_config.frequency_limits.stop_loss_cooldown_hours
        self.backtester.frequency_limits_enabled = risk_config.frequency_limits.enabled

    def test_all_risk_checks_pass(self):
        """Test all risk checks pass."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Mock all checks to pass
        with patch.object(self.backtester, '_is_optimal_session', return_value=True), \
             patch.object(self.backtester, '_check_correlation_limits', return_value=True), \
             patch.object(self.backtester, '_check_trade_frequency_limits', return_value=True), \
             patch.object(self.backtester, '_check_stop_loss_cooldown', return_value=True):

            result = self.backtester._check_risk_limits(signal)
            assert result is True

    def test_correlation_check_fails(self):
        """Test correlation check failure blocks trade."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Mock correlation check to fail
        with patch.object(self.backtester, '_is_optimal_session', return_value=True), \
             patch.object(self.backtester, '_check_correlation_limits', return_value=False), \
             patch.object(self.backtester, '_check_trade_frequency_limits', return_value=True), \
             patch.object(self.backtester, '_check_stop_loss_cooldown', return_value=True):

            result = self.backtester._check_risk_limits(signal)
            assert result is False

    def test_frequency_check_fails(self):
        """Test frequency check failure blocks trade."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Mock frequency check to fail
        with patch.object(self.backtester, '_is_optimal_session', return_value=True), \
             patch.object(self.backtester, '_check_correlation_limits', return_value=True), \
             patch.object(self.backtester, '_check_trade_frequency_limits', return_value=False), \
             patch.object(self.backtester, '_check_stop_loss_cooldown', return_value=True):

            result = self.backtester._check_risk_limits(signal)
            assert result is False

    def test_cooldown_check_fails(self):
        """Test cooldown check failure blocks trade."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Mock cooldown check to fail
        with patch.object(self.backtester, '_is_optimal_session', return_value=True), \
             patch.object(self.backtester, '_check_correlation_limits', return_value=True), \
             patch.object(self.backtester, '_check_trade_frequency_limits', return_value=True), \
             patch.object(self.backtester, '_check_stop_loss_cooldown', return_value=False):

            result = self.backtester._check_risk_limits(signal)
            assert result is False

    def test_disabled_features_bypass_checks(self):
        """Test disabled features bypass their checks."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='TEST',
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0],
            risk_reward=2.0,
            confidence=0.8,
            metadata={'pair': 'BTCUSDT'}
        )

        # Disable correlation and frequency checks
        self.backtester.correlation_enabled = False
        self.backtester.frequency_limits_enabled = False

        # Mock session check to pass
        with patch.object(self.backtester, '_is_optimal_session', return_value=True):
            result = self.backtester._check_risk_limits(signal)
            assert result is True


if __name__ == '__main__':
    pytest.main([__file__])
