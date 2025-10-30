#!/usr/bin/env python3
"""
Quick validation backtest to verify counter-trend FVG fix
"""

from backtester import BacktestEngine
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.config_loader import ConfigLoader

def main():
    print("="*70)
    print("VALIDATION BACKTEST: Counter-Trend FVG Fix")
    print("Period: 2023 Q1 (BEARISH market)")
    print("="*70)
    print()

    # Initialize
    data_loader = DataLoader(base_path='data/raw')
    config_loader = ConfigLoader()
    strategy = TradingStrategy(
        data_loader=data_loader,
        pair='BTCUSDT',
        start_date='2023-01-01',
        end_date='2023-03-31'
    )

    backtester = BacktestEngine(
        trading_strategy=strategy,
        data_loader=data_loader,
        initial_balance=10000
    )

    # Run backtest
    result = backtester.run_backtest(
        pair='BTCUSDT',
        start_date='2023-01-01',
        end_date='2023-03-31',
        timeframe='1h'
    )

    # Display results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Trades: {len(result.get('trade_journal', []))}")
    print(f"Final Balance: ${result.get('final_balance', 10000):.2f}")
    print(f"P&L: ${result.get('final_balance', 10000) - 10000:+.2f}")
    print(f"Return: {((result.get('final_balance', 10000) / 10000) - 1) * 100:+.2f}%")
    print()

    # Check signal generation
    print("EXPECTED: ~100-150 SELL signals in BEARISH Q1 2023")
    print("BEFORE FIX: Only 1-3 signals (99% filtered out)")
    print()

    if len(result.get('trade_journal', [])) > 10:
        print("✅ FIX WORKING: Significantly more signals generated!")
    else:
        print("⚠️  Still low signal count - investigate further")

if __name__ == '__main__':
    main()
