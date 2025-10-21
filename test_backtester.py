from backtester import BacktestEngine

engine = BacktestEngine(base_path=".",
                        initial_balance=10000, risk_per_trade=0.01)
results = engine.run_backtest("BTCUSDT","2024-01-01","2025-10-21")
print("Final Balance:", results['final_balance'])
print("Total Trades:", results['total_trades'])
print("Win Rate:", results['win_rate'])
print("Profit Factor:", results['profit_factor'])
print("Max Drawdown:", results['max_drawdown'])
print("Sharpe Ratio:", results.get('sharpe_ratio'))
results['trade_journal'].to_csv('trade_journal.csv',index=False)
