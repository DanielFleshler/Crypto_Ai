import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtester import BacktestEngine

def test_backtester():
    """Test the backtester with assertions"""
    print("Testing Backtester...")
    
    engine = BacktestEngine(base_path=".",
                            initial_balance=10000, risk_per_trade=0.01)
    
    try:
        results = engine.run_backtest("BTCUSDT","2024-01-01","2025-10-21")
        
        # Assertions to validate results
        assert 'final_balance' in results, "Missing final_balance in results"
        assert 'total_trades' in results, "Missing total_trades in results"
        assert 'win_rate' in results, "Missing win_rate in results"
        assert 'profit_factor' in results, "Missing profit_factor in results"
        assert 'max_drawdown' in results, "Missing max_drawdown in results"
        assert 'sharpe_ratio' in results, "Missing sharpe_ratio in results"
        assert 'equity_curve' in results, "Missing equity_curve in results"
        assert 'trade_journal' in results, "Missing trade_journal in results"
        
        # Validate data types and ranges
        assert isinstance(results['final_balance'], (int, float)), "final_balance should be numeric"
        assert isinstance(results['total_trades'], int), "total_trades should be integer"
        assert 0 <= results['win_rate'] <= 1, "win_rate should be between 0 and 1"
        assert results['profit_factor'] >= 0, "profit_factor should be non-negative"
        assert 0 <= results['max_drawdown'] <= 1, "max_drawdown should be between 0 and 1"
        assert isinstance(results['sharpe_ratio'], (int, float)), "sharpe_ratio should be numeric"
        assert isinstance(results['equity_curve'], list), "equity_curve should be a list"
        assert isinstance(results['trade_journal'], object), "trade_journal should be a DataFrame"
        
        print("✅ All assertions passed!")
        print("Final Balance:", results['final_balance'])
        print("Total Trades:", results['total_trades'])
        print("Win Rate:", results['win_rate'])
        print("Profit Factor:", results['profit_factor'])
        print("Max Drawdown:", results['max_drawdown'])
        print("Sharpe Ratio:", results['sharpe_ratio'])
        
        # Save trade journal
        results['trade_journal'].to_csv('trade_journal.csv',index=False)
        print("✅ Trade journal saved to trade_journal.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_backtester()
    if not success:
        sys.exit(1)
