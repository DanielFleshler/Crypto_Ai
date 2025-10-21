# backtester.py

import pandas as pd
import numpy as np
from typing import List, Dict
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy, Signal

class BacktestEngine:
    def __init__(self, base_path: str,
                 initial_balance: float = 10000,
                 risk_per_trade: float = 0.01):
        self.data_loader = DataLoader(base_path)
        self.strategy = TradingStrategy(base_path)
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade

    def run_backtest(self, pair: str,
                     start_date: str, end_date: str) -> Dict:
        # 1. Load data
        ohlc = self.data_loader.load_pair_data(
            pair, ['15m'], start_date, end_date)['15m']

        # 2. Generate signals
        analysis = self.strategy.run_analysis(pair, start_date, end_date)
        signals: List[Signal] = analysis['signals']

        balance = self.initial_balance
        equity_curve = []
        trade_journal = []

        # Set start_time as index for proper time-based operations
        if 'start_time' in ohlc.columns:
            ohlc = ohlc.set_index('start_time')
        
        times = list(ohlc.index)

        # 3. Simulate each signal
        for sig in signals:
            entry_time = sig.timestamp
            if entry_time not in times:
                continue
            entry_idx = times.index(entry_time)
            entry_price = sig.price

            # Position sizing
            risk_amount = balance * self.risk_per_trade
            risk = abs(entry_price - sig.stop_loss)
            qty = risk_amount / risk if risk > 0 else 0

            position = {'open': True}

            # Trade loop
            pnl = 0
            for idx in range(entry_idx+1, len(times)):
                row = ohlc.iloc[idx]
                price_high = row['high']
                price_low = row['low']
                current_time = times[idx]

                # Stop Loss
                if sig.signal_type == 'BUY' and price_low <= sig.stop_loss:
                    pnl = (sig.stop_loss - entry_price) * qty
                    self._record_trade(sig, current_time, sig.stop_loss, pnl, trade_journal)
                    position['open'] = False
                    break
                if sig.signal_type == 'SELL' and price_high >= sig.stop_loss:
                    pnl = (entry_price - sig.stop_loss) * qty
                    self._record_trade(sig, current_time, sig.stop_loss, pnl, trade_journal)
                    position['open'] = False
                    break

                # Take Profits
                for tp in sig.take_profits:
                    if position['open'] and sig.signal_type=='BUY' and price_high >= tp:
                        pnl = (tp - entry_price) * qty
                        self._record_trade(sig, current_time, tp, pnl, trade_journal)
                        position['open'] = False
                        break
                    if position['open'] and sig.signal_type=='SELL' and price_low <= tp:
                        pnl = (entry_price - tp) * qty
                        self._record_trade(sig, current_time, tp, pnl, trade_journal)
                        position['open'] = False
                        break
                if not position['open']:
                    break

            balance += pnl
            equity_curve.append(balance)

        # 4. Metrics
        df_journal = pd.DataFrame(trade_journal)
        
        # Handle case when no trades were executed
        if len(df_journal) == 0:
            wins = 0
            losses = 0
            win_rate = 0
            profit_factor = 0
            sharpe = 0
            returns = pd.Series([0])
        else:
            wins = df_journal[df_journal['pnl']>0]['pnl'].sum()
            losses = -df_journal[df_journal['pnl']<0]['pnl'].sum()
            win_rate = len(df_journal[df_journal['pnl']>0]) / len(df_journal) if len(df_journal) else 0
            profit_factor = wins / losses if losses>0 else np.inf
            returns = df_journal['pnl'] / self.initial_balance
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std()>0 else np.nan
        
        dd = self._max_drawdown(equity_curve)

        return {
            'final_balance': balance,
            'total_trades': len(df_journal),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': dd,
            'equity_curve': equity_curve,
            'trade_journal': df_journal
        }

    def _record_trade(self, sig: Signal, time, price, pnl, journal):
        journal.append({
            'entry_time': sig.timestamp,
            'exit_time': time,
            'entry_price': sig.price,
            'exit_price': price,
            'direction': sig.signal_type,
            'pnl': pnl,
            'rr': pnl / abs(sig.price - sig.stop_loss) if sig.stop_loss else np.nan,
            'entry_type': sig.entry_type
        })

    def _max_drawdown(self, equity_curve: List[float]) -> float:
        peak = equity_curve[0] if equity_curve else self.initial_balance
        max_dd = 0
        for x in equity_curve:
            peak = max(peak, x)
            dd = (peak - x) / peak
            max_dd = max(max_dd, dd)
        return max_dd
