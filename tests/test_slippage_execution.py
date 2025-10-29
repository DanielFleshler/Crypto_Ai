"""
Test Slippage-Adjusted Stop Execution

Tests the implementation of slippage modeling on stop loss hits to ensure
realistic execution in volatile markets.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from trading_strategy.data_structures import Signal, PositionState
from trading_strategy.config_loader import ConfigLoader
from backtester import BacktestEngine


class TestSlippageExecution:
    """Test slippage-adjusted stop execution functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_loader = ConfigLoader()
        self.risk_config = self.config_loader.get_risk_management_config()

        # Create mock backtest engine
        self.backtest_engine = BacktestEngine(base_path=".", config_loader=self.config_loader, initial_balance=10000.0, risk_per_trade=0.02)

        # Set slippage to 0.05% for testing (matching config)
        # Create a modified config with custom slippage
        from dataclasses import replace
        self.risk_config = replace(self.risk_config, slippage_percent=0.0005)

    def test_slippage_adjusted_stop_price_buy_signal(self):
        """Test slippage adjustment for BUY signals (long positions)."""
        # Create BUY signal
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # Test slippage adjustment
        adjusted_price = self.backtest_engine._get_slippage_adjusted_stop_price(signal, signal.stop_loss)

        # For BUY signals, slippage should make the exit price worse (lower)
        expected_price = 95.0 - (95.0 * 0.0005)  # 95.0 - 0.0475 = 94.9525
        assert abs(adjusted_price - expected_price) < 0.001
        assert adjusted_price < signal.stop_loss  # Should be worse than stop price

    def test_slippage_adjusted_stop_price_sell_signal(self):
        """Test slippage adjustment for SELL signals (short positions)."""
        # Create SELL signal
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="SELL",
            entry_type="SELL",
            price=100.0,
            stop_loss=105.0,
            take_profits=[95.0, 90.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # Test slippage adjustment
        adjusted_price = self.backtest_engine._get_slippage_adjusted_stop_price(signal, signal.stop_loss)

        # For SELL signals, slippage should make the exit price worse (higher)
        expected_price = 105.0 + (105.0 * 0.0005)  # 105.0 + 0.0525 = 105.0525
        assert abs(adjusted_price - expected_price) < 0.001
        assert adjusted_price > signal.stop_loss  # Should be worse than stop price

    def test_slippage_impact_on_pnl_buy_signal(self):
        """Test that slippage reduces PnL for BUY signals hitting stop loss."""
        # Create BUY signal
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        quantity = 100.0

        # Calculate PnL without slippage
        pnl_no_slippage = self.backtest_engine._calculate_pnl(signal, signal.price, signal.stop_loss, quantity)

        # Calculate PnL with slippage
        slippage_adjusted_price = self.backtest_engine._get_slippage_adjusted_stop_price(signal, signal.stop_loss)
        pnl_with_slippage = self.backtest_engine._calculate_pnl(signal, signal.price, slippage_adjusted_price, quantity)

        # PnL with slippage should be worse (more negative) than without slippage
        assert pnl_with_slippage < pnl_no_slippage
        assert pnl_no_slippage == -500.0  # (95 - 100) * 100 = -500
        assert pnl_with_slippage < -500.0  # Should be worse due to slippage

    def test_slippage_impact_on_pnl_sell_signal(self):
        """Test that slippage reduces PnL for SELL signals hitting stop loss."""
        # Create SELL signal
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="SELL",
            entry_type="SELL",
            price=100.0,
            stop_loss=105.0,
            take_profits=[95.0, 90.0],
            risk_reward=2.0,
            confidence=0.8
        )

        quantity = 100.0

        # Calculate PnL without slippage
        pnl_no_slippage = self.backtest_engine._calculate_pnl(signal, signal.price, signal.stop_loss, quantity)

        # Calculate PnL with slippage
        slippage_adjusted_price = self.backtest_engine._get_slippage_adjusted_stop_price(signal, signal.stop_loss)
        pnl_with_slippage = self.backtest_engine._calculate_pnl(signal, signal.price, slippage_adjusted_price, quantity)

        # PnL with slippage should be worse (more negative) than without slippage
        assert pnl_with_slippage < pnl_no_slippage
        assert pnl_no_slippage == -500.0  # (100 - 105) * 100 = -500
        assert pnl_with_slippage < -500.0  # Should be worse due to slippage

    def test_slippage_configuration_zero_slippage(self):
        """Test behavior when slippage is set to zero."""
        # Create a mock config with zero slippage
        mock_config = Mock()
        mock_risk_config = Mock()
        mock_risk_config.slippage_percent = 0.0
        mock_config.get_risk_management_config.return_value = mock_risk_config
        mock_config.get_session_config.return_value = Mock()

        engine_zero_slippage = BacktestEngine(
            base_path="/mock/path",
            config_loader=mock_config,
            initial_balance=10000,
            risk_per_trade=0.01
        )

        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # With zero slippage, adjusted price should equal stop price
        adjusted_price = engine_zero_slippage._get_slippage_adjusted_stop_price(signal, signal.stop_loss)
        assert adjusted_price == signal.stop_loss

    def test_slippage_configuration_high_slippage(self):
        """Test behavior with high slippage percentage."""
        # Create a mock config with high slippage (1%)
        mock_config = Mock()
        mock_risk_config = Mock()
        mock_risk_config.slippage_percent = 0.01
        mock_config.get_risk_management_config.return_value = mock_risk_config
        mock_config.get_session_config.return_value = Mock()

        engine_high_slippage = BacktestEngine(
            base_path="/mock/path",
            config_loader=mock_config,
            initial_balance=10000,
            risk_per_trade=0.01
        )

        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # Test slippage adjustment
        adjusted_price = engine_high_slippage._get_slippage_adjusted_stop_price(signal, signal.stop_loss)

        # Should be 1% worse than stop price
        expected_price = 95.0 - (95.0 * 0.01)  # 95.0 - 0.95 = 94.05
        assert abs(adjusted_price - expected_price) < 0.001

    def test_slippage_integration_with_trade_execution(self):
        """Test slippage integration in full trade execution."""
        # Create test data with low enough prices to hit stop loss
        dates = pd.date_range(start='2023-01-01 10:00:00', periods=100, freq='15min')  # London session
        ohlc_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [94.0] * 100,  # Low enough to hit stop loss
            'close': [100.5] * 100,
            'volume': [1000] * 100
        }, index=dates)

        # Create signal that will hit stop loss
        signal = Signal(
            timestamp=dates[0],
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,  # Stop will be hit by low of 94.0
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # Mock the data loader
        with patch.object(self.backtest_engine.data_loader, 'load_pair_data') as mock_load:
            mock_load.return_value = {'15m': ohlc_data}

            # Mock the trading strategy
            with patch.object(self.backtest_engine.trading_strategy, 'run_analysis') as mock_analysis:
                mock_analysis.return_value = {'signals': [signal]}

                # Run backtest
                result = self.backtest_engine.run_backtest('BTCUSDT', '2023-01-01', '2023-01-02')

                # Verify that slippage was applied
                assert result['total_trades'] == 1

                # Get the trade journal entry
                trade_journal = result['trade_journal']
                assert len(trade_journal) == 1

                trade = trade_journal.iloc[0]

                # Exit price should be worse than stop loss due to slippage
                assert trade['exit_price'] < signal.stop_loss
                assert trade['exit_reason'] == 'STOP_LOSS'

                # PnL should reflect the slippage impact
                expected_pnl_no_slippage = (signal.stop_loss - signal.price) * trade['quantity']
                assert trade['pnl'] < expected_pnl_no_slippage

    def test_slippage_affects_final_balance(self):
        """Test that slippage affects the final balance calculation."""
        # Test slippage calculation directly without full backtest

        # Create signals
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        quantity = 1000.0

        # Test with zero slippage
        mock_config_zero = Mock()
        mock_risk_config_zero = Mock()
        mock_risk_config_zero.slippage_percent = 0.0
        mock_config_zero.get_risk_management_config.return_value = mock_risk_config_zero
        mock_config_zero.get_session_config.return_value = Mock()

        engine_zero = BacktestEngine(
            base_path="/mock/path",
            config_loader=mock_config_zero,
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Test with slippage
        mock_config_slippage = Mock()
        mock_risk_config_slippage = Mock()
        mock_risk_config_slippage.slippage_percent = 0.01  # 1%
        mock_config_slippage.get_risk_management_config.return_value = mock_risk_config_slippage
        mock_config_slippage.get_session_config.return_value = Mock()

        engine_slippage = BacktestEngine(
            base_path="/mock/path",
            config_loader=mock_config_slippage,
            initial_balance=10000,
            risk_per_trade=0.01
        )

        # Calculate PnL with different slippage settings
        pnl_zero_slippage = engine_zero._calculate_pnl(signal, signal.price, signal.stop_loss, quantity)

        slippage_adjusted_price = engine_slippage._get_slippage_adjusted_stop_price(signal, signal.stop_loss)
        pnl_with_slippage = engine_slippage._calculate_pnl(signal, signal.price, slippage_adjusted_price, quantity)

        # PnL with slippage should be worse (more negative)
        assert pnl_with_slippage < pnl_zero_slippage

        # Verify the slippage adjustment is working
        assert slippage_adjusted_price < signal.stop_loss  # For BUY signal

        # Calculate expected final balances
        initial_balance = 10000
        final_balance_zero = initial_balance + pnl_zero_slippage
        final_balance_slippage = initial_balance + pnl_with_slippage

        # Final balance with slippage should be worse
        assert final_balance_slippage < final_balance_zero

    def test_slippage_edge_cases(self):
        """Test edge cases for slippage calculation."""
        signal = Signal(
            timestamp=datetime.now(),
            signal_type="BUY",
            entry_type="BUY",
            price=100.0,
            stop_loss=95.0,
            take_profits=[105.0, 110.0],
            risk_reward=2.0,
            confidence=0.8
        )

        # Test with very small stop price
        small_stop = 0.01
        adjusted_small = self.backtest_engine._get_slippage_adjusted_stop_price(signal, small_stop)
        assert adjusted_small < small_stop
        assert adjusted_small > 0  # Should not go negative

        # Test with very large stop price
        large_stop = 10000.0
        adjusted_large = self.backtest_engine._get_slippage_adjusted_stop_price(signal, large_stop)
        assert adjusted_large < large_stop
        assert adjusted_large > 0  # Should not go negative


if __name__ == "__main__":
    pytest.main([__file__])
