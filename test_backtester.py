# test_backtester.py
# עותק-הדבק את הקוד הזה כ-test_backtester.py והרץ כדי לבדוק את מנוע ה-Backtesting

from backtester import BacktestEngine

# הגדר נתיב בסיס לנתונים
base_path = "/Users/danielfleshler/Desktop/Code/Crypto_bot_trader"

# אתחול מנוע Backtest
engine = BacktestEngine(
    base_path=base_path,
    initial_balance=10000,
    risk_per_trade=0.01
)

# הפעלת Backtest
results = engine.run_backtest(
    pair="BTCUSDT",
    start_date="2024-01-01",
    end_date="2024-06-30"
)

# הצגת התוצאות
print("Final Balance:", results['final_balance'])
print("Total Trades:", results['total_trades'])
print("Win Rate:", results['win_rate'])
print("Profit Factor:", results['profit_factor'])
print("Max Drawdown:", results['max_drawdown'])
print("Sharpe Ratio:", results['sharpe_ratio'])

# שמירת Trade Journal ל-CSV
results['trade_journal'].to_csv('trade_journal.csv', index=False)
print("Trade journal saved to trade_journal.csv")