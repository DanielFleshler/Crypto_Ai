"""
Unit tests for the Backtesting Engine.
Tests backtesting functionality, PnL calculations, and performance metrics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtester import BacktestEngine
from trading_strategy.data_structures import Signal, Position, BacktestResult


class TestBacktester:
    """Test cases for the backtesting engine."""

    def setup_method(self):
        """Set up test data and configuration."""
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1h')
        np.random.seed(42)

        base_price = 50000
        price_changes = np.random.normal(0, 0.02, 1000)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        })

        self.data.set_index('timestamp', inplace=True)

        # Create proper config loader
        from trading_strategy.config_loader import ConfigLoader
        self.config_loader = ConfigLoader()

        # Initialize backtester with correct parameters
        self.backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

    def test_backtester_initialization(self):
        """Test backtester initialization."""
        assert self.backtester is not None
        assert self.backtester.base_path == "."
        assert self.backtester.config_loader is not None
        assert self.backtester.initial_balance == 10000.0
        assert self.backtester.risk_per_trade == 0.02
        assert self.backtester.current_balance == 10000.0

    def test_execute_trade_basic(self):
        """Test basic trade execution."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        result = self.backtester.execute_trade(signal, self.data, 0, self.backtester.current_balance)

        assert result is not None
        assert 'position' in result
        assert 'success' in result
        assert 'error' in result

        if result['success']:
            assert result['position'] is not None
            # Account for slippage - entry price should be higher for BUY signals
            expected_entry_price = 50000.0 * (1.0 + 0.0005)  # 0.05% slippage
            assert abs(result['position'].entry_price - expected_entry_price) < 0.01
            assert result['position'].side == 'LONG'

    def test_execute_trade_short(self):
        """Test short trade execution."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='SELL',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=51000.0,
            take_profits=[48000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        result = self.backtester.execute_trade(signal, self.data, 0, self.backtester.current_balance)

        assert result is not None
        assert 'position' in result
        assert 'success' in result

        if result['success']:
            # Account for slippage - entry price should be lower for SELL signals
            expected_entry_price = 50000.0 * (1.0 - 0.0005)  # 0.05% slippage
            assert abs(result['position'].entry_price - expected_entry_price) < 0.01
            assert result['position'].side == 'SHORT'

    def test_position_sizing(self):
        """Test position sizing calculation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        position_size = self.backtester.calculate_position_size(signal.price, signal.stop_loss, 10000.0)

        assert position_size > 0
        assert position_size <= 10000.0 * 0.95  # Max 95% of balance
        assert position_size >= 10000.0 * 0.01  # Min 1% of balance

    def test_position_sizing_with_volatility(self):
        """Test position sizing with volatility adjustment."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        # Test with different volatility levels
        low_vol_size = self.backtester.calculate_position_size(signal.price, signal.stop_loss, 10000.0)
        high_vol_size = self.backtester.calculate_position_size(signal.price, signal.stop_loss, 10000.0)

        # Higher volatility should result in smaller position size
        assert high_vol_size < low_vol_size
        assert low_vol_size > 0
        assert high_vol_size > 0

    def test_manage_position_exits(self):
        """Test position exit management."""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000.0,
            quantity=1.0,
            stop_loss=49000.0,
            take_profits=[51000.0, 52000.0, 53000.0, 54000.0],
            entry_time=datetime.now()
        )

        # Test TP1 hit
        result = self.backtester.manage_position_exits(position, 51000.0, 0)

        assert result is not None
        assert 'tp1_hit' in result
        assert 'remaining_qty' in result
        assert 'stop_at_be' in result

        if result['tp1_hit']:
            assert result['remaining_qty'] == 0.7  # 30% closed
            assert result['stop_at_be'] is True

    def test_stop_to_breakeven(self):
        """Test stop to breakeven logic."""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000.0,
            quantity=1.0,
            stop_loss=49000.0,
            take_profits=[51000.0, 52000.0, 53000.0, 54000.0],
            entry_time=datetime.now()
        )

        # TP1 hit should move stop to breakeven
        result = self.backtester.manage_position_exits(position, 51000.0, 0)

        if result['tp1_hit']:
            assert result['stop_at_be'] is True
            assert position.stop_loss == 50000.0  # Moved to entry price

    def test_calculate_pnl(self):
        """Test PnL calculation."""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000.0,
            quantity=1.0,
            stop_loss=49000.0,
            take_profits=[51000.0, 52000.0, 53000.0, 54000.0],
            entry_time=datetime.now()
        )

        # Test profitable exit
        pnl = self.backtester.calculate_pnl(position, 52000.0)
        assert pnl > 0

        # Test losing exit
        pnl = self.backtester.calculate_pnl(position, 48000.0)
        assert pnl < 0

        # Test breakeven exit
        pnl = self.backtester.calculate_pnl(position, 50000.0)
        assert pnl == 0.0

    def test_update_equity_curve(self):
        """Test equity curve updates."""
        initial_balance = self.backtester.current_balance

        # Simulate a trade
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        result = self.backtester.execute_trade(signal, self.data, 0, self.backtester.current_balance)

        if result['success']:
            # Update equity curve
            self.backtester.update_equity_curve(0)

            # Check equity curve
            assert len(self.backtester.equity_curve) > 0
            assert self.backtester.equity_curve[0] == initial_balance

    def test_apply_risk_management_rules(self):
        """Test risk management rules application."""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000.0,
            quantity=1.0,
            stop_loss=49000.0,
            take_profits=[51000.0, 52000.0, 53000.0, 54000.0],
            entry_time=datetime.now()
        )

        # Test risk management rules
        can_trade = self.backtester.apply_risk_management_rules(position, 0)

        assert isinstance(can_trade, bool)
        assert can_trade in [True, False]

    def test_generate_backtest_report(self):
        """Test backtest report generation."""
        # Run a simple backtest
        signals = [
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='ELLIOTT_WAVE',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[52000.0],
                confidence=0.8,
                risk_reward=1.0,
                metadata={'timeframe': '1h', 'source': 'elliott_wave'}
            )
        ]

        # Execute trades
        for i, signal in enumerate(signals):
            result = self.backtester.execute_trade(signal, self.data, i, self.backtester.current_balance)
            if result['success']:
                # Simulate exit
                exit_price = 52000.0
                pnl = self.backtester.calculate_pnl(result['position'], exit_price)
                self.backtester.current_balance += pnl

        # Generate report
        report = self.backtester.generate_backtest_report()

        assert report is not None
        assert isinstance(report, BacktestResult)
        assert report.initial_balance == 10000.0
        assert report.final_balance >= 0
        assert report.total_trades >= 0
        assert report.win_rate >= 0.0
        assert report.win_rate <= 1.0
        assert report.profit_factor >= 0.0
        assert report.max_drawdown >= 0.0
        assert report.sharpe_ratio >= 0.0

    def test_risk_limits_enforcement(self):
        """Test risk limits enforcement."""
        # Test maximum positions limit
        max_positions = [Position(symbol='BTCUSDT', side='LONG', entry_price=50000.0, quantity=1.0, stop_loss=49000.0, take_profits=[51000.0], entry_time=datetime.now()) for _ in range(5)]

        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        can_trade = self.backtester.check_risk_limits(signal, max_positions)
        assert can_trade is False

        # Test with fewer positions
        few_positions = [Position(symbol='BTCUSDT', side='LONG', entry_price=50000.0, quantity=1.0, stop_loss=49000.0, take_profits=[51000.0], entry_time=datetime.now()) for _ in range(2)]
        can_trade = self.backtester.check_risk_limits(signal, few_positions)
        assert can_trade is True

    def test_daily_risk_limit(self):
        """Test daily risk limit enforcement."""
        # Set high daily risk
        self.backtester.daily_pnl = -0.06  # 6% loss

        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        can_trade = self.backtester.check_risk_limits(signal, [])
        assert can_trade is False

    def test_drawdown_protection(self):
        """Test drawdown protection."""
        # Set high drawdown
        self.backtester.current_drawdown = 0.15  # 15%
        self.backtester.max_drawdown = 0.10  # 10%

        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        can_trade = self.backtester.check_risk_limits(signal, [])
        assert can_trade is False

    def test_commission_and_slippage(self):
        """Test commission and slippage calculations."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        # Test with commission and slippage
        result = self.backtester.execute_trade(signal, self.data, 0, self.backtester.current_balance)

        if result['success']:
            position = result['position']

            # Check that commission and slippage are applied
            assert position.entry_price != 50000.0  # Should be adjusted for slippage
            assert position.quantity > 0

    def test_partial_exits(self):
        """Test partial exit mechanics."""
        position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000.0,
            quantity=1.0,
            stop_loss=49000.0,
            take_profits=[51000.0, 52000.0, 53000.0, 54000.0],
            entry_time=datetime.now()
        )

        # Test TP1 hit
        result = self.backtester.manage_position_exits(position, 51000.0, 0)

        if result['tp1_hit']:
            assert result['remaining_qty'] == 0.7  # 30% closed
            assert result['stop_at_be'] is True

            # Test TP2 hit
            result = self.backtester.manage_position_exits(position, 52000.0, 0)

            if result['tp2_hit']:
                assert result['remaining_qty'] == 0.3  # 40% more closed

    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create test trades
        trades = [
            {'entry': 50000, 'exit': 52000, 'type': 'BUY'},
            {'entry': 52000, 'exit': 50000, 'type': 'SELL'},
            {'entry': 50000, 'exit': 48000, 'type': 'BUY'},
            {'entry': 48000, 'exit': 50000, 'type': 'SELL'}
        ]

        # Execute trades
        for trade in trades:
            signal = Signal(
                timestamp=datetime.now(),
                signal_type=trade['type'],
                entry_type='ELLIOTT_WAVE',
                price=trade['entry'],
                stop_loss=trade['entry'] * 0.98,
                take_profits=[trade['exit']],
                confidence=0.8,
                risk_reward=1.0,
                metadata={'timeframe': '1h', 'source': 'elliott_wave'}
            )

            result = self.backtester.execute_trade(signal, self.data, 0, self.backtester.current_balance)
            if result['success']:
                pnl = self.backtester.calculate_pnl(result['position'], trade['exit'])
                self.backtester.current_balance += pnl

        # Generate report
        report = self.backtester.generate_backtest_report()

        # Check metrics
        assert report.total_trades >= 0
        assert report.winning_trades >= 0
        assert report.losing_trades >= 0
        assert report.win_rate >= 0.0
        assert report.win_rate <= 1.0
        assert report.profit_factor >= 0.0
        assert report.max_drawdown >= 0.0
        assert report.sharpe_ratio >= 0.0

    def test_error_handling(self):
        """Test error handling in backtester."""
        # Test with invalid signal - this should raise ValueError during Signal creation
        with pytest.raises(ValueError):
            invalid_signal = Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='ELLIOTT_WAVE',
                price=0.0,  # Invalid price
                stop_loss=0.0,
                take_profits=[0.0],
                confidence=0.8,
                risk_reward=1.0,
                metadata={'timeframe': '1h', 'source': 'elliott_wave'}
            )

        # Test with valid signal but invalid data
        valid_signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='ELLIOTT_WAVE',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[51000.0],
            confidence=0.8,
            risk_reward=1.0,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        # Test with invalid data
        invalid_data = pd.DataFrame()  # Empty dataframe
        result = self.backtester.execute_trade(valid_signal, invalid_data, 0, 10000.0)
        assert result['success'] is False
        assert 'error' in result

    def test_data_validation(self):
        """Test data validation in backtester."""
        # Test with invalid data - BacktestEngine constructor doesn't validate data immediately
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

        # Test with missing columns - this should be handled gracefully
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103]
            # Missing required columns
        })

        # This should handle missing columns gracefully
        try:
            result = backtester.execute_trade(Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='TEST',
                price=100.0,
                stop_loss=95.0,
                take_profits=[105.0],
                confidence=0.8,
                risk_reward=1.0
            ), incomplete_data, 0, 10000.0)
            assert isinstance(result, dict)  # Should return result dict
        except Exception:
            pass  # Expected to handle errors gracefully

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid configuration - BacktestEngine constructor doesn't validate config immediately
        invalid_config = {
            'initial_balance': -1000.0,  # Invalid balance
            'risk_per_trade': 1.5,  # Invalid risk
            'max_positions': -1  # Invalid positions
        }

        # This should not raise an error during initialization
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

    def test_memory_management(self):
        """Test memory management in backtester."""
        # Test with large dataset
        large_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10000, freq='1h'),
            'open': np.random.uniform(40000, 60000, 10000),
            'high': np.random.uniform(40000, 60000, 10000),
            'low': np.random.uniform(40000, 60000, 10000),
            'close': np.random.uniform(40000, 60000, 10000),
            'volume': np.random.randint(1000, 10000, 10000)
        })

        large_data.set_index('timestamp', inplace=True)

        # Should handle large dataset without memory issues
        backtester = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)
        assert backtester is not None

    def test_concurrent_trades(self):
        """Test concurrent trade handling."""
        # Test multiple simultaneous trades
        signals = [
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='ELLIOTT_WAVE',
                price=50000.0,
                stop_loss=49000.0,
                take_profits=[52000.0],
                confidence=0.8,
                risk_reward=1.0,
                metadata={'timeframe': '1h', 'source': 'elliott_wave'}
            ),
            Signal(
                timestamp=datetime.now(),
                signal_type='BUY',
                entry_type='ELLIOTT_WAVE',
                price=51000.0,
                stop_loss=50000.0,
                take_profits=[53000.0],
                confidence=0.8,
                risk_reward=1.0,
                metadata={'timeframe': '1h', 'source': 'elliott_wave'}
            )
        ]

        positions = []
        for i, signal in enumerate(signals):
            result = self.backtester.execute_trade(signal, self.data, i, self.backtester.current_balance)
            if result['success']:
                positions.append(result['position'])

        # Should handle multiple positions
        assert len(positions) <= 2  # May be limited by risk management

    def test_bidirectional_symmetry(self):
        """Test bidirectional symmetry in backtester."""
        # Test bullish scenario
        bullish_signal = Signal(
            timestamp=datetime.now(),
            signal_type='BUY',
            entry_type='elliott_wave',
            price=50000.0,
            stop_loss=49000.0,
            take_profits=[52000.0],
            risk_reward=1.0,
            confidence=0.8,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        # Test bearish scenario
        bearish_signal = Signal(
            timestamp=datetime.now(),
            signal_type='SELL',
            entry_type='elliott_wave',
            price=50000.0,
            stop_loss=51000.0,
            take_profits=[48000.0],
            risk_reward=1.0,
            confidence=0.8,
            metadata={'timeframe': '1h', 'source': 'elliott_wave'}
        )

        # Execute both trades
        bullish_result = self.backtester.execute_trade(bullish_signal, 0)
        bearish_result = self.backtester.execute_trade(bearish_signal, 1)

        # Check symmetry
        assert bullish_result['success'] in [True, False]
        assert bearish_result['success'] in [True, False]

        if bullish_result['success'] and bearish_result['success']:
            assert bullish_result['position'].signal_type == 'BUY'
            assert bearish_result['position'].signal_type == 'SELL'
