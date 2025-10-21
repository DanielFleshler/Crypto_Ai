import pandas as pd
import numpy as np
from typing import List, Dict
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.data_structures import Signal

class BacktestEngine:
    def __init__(self, base_path: str,
                 initial_balance: float = 10000,
                 risk_per_trade: float = 0.01):
        self.dl=DataLoader(base_path)
        self.ts=TradingStrategy(base_path)
        self.initial_balance=initial_balance
        self.risk_per_trade=risk_per_trade

    def run_backtest(self,pair,sd,ed)->Dict:
        ohlc=self.dl.load_pair_data(pair,['15m'],sd,ed)['15m']
        res=self.ts.run_analysis(pair,sd,ed)
        signals=res['signals']
        bal=self.initial_balance; eq=[]; journal=[]
        
        # Create timestamp-to-index mapping for O(1) lookups
        times=list(ohlc.index)
        timestamp_to_index = {ts: i for i, ts in enumerate(times)}
        
        for sig in signals:
            # Find the closest timestamp in the OHLC data
            if sig.timestamp in timestamp_to_index:
                i = timestamp_to_index[sig.timestamp]
            else:
                # Find the closest timestamp
                time_diffs = [abs((pd.Timestamp(sig.timestamp) - pd.Timestamp(t)).total_seconds()) for t in times]
                if time_diffs:
                    i = time_diffs.index(min(time_diffs))
                else:
                    continue
            
            ep=sig.price
            risk=abs(ep-sig.stop_loss)
            
            # Handle division by zero and validate qty
            if risk <= 0:
                print(f"Warning: Risk is zero or negative for signal at {sig.timestamp}, skipping trade")
                continue
                
            qty = (bal * self.risk_per_trade) / risk
            # Limit qty to 8 decimal places to prevent exchange errors
            qty = round(qty, 8)
            
            if qty <= 0:
                print(f"Warning: Calculated quantity is zero or negative, skipping trade")
                continue
                
            pnl=0
            exit_price = None
            exit_reason = None
            
            for j in range(i+1,len(times)):
                hi,lo=ohlc.iloc[j][['high','low']]
                t=times[j]
                
                # Check stop loss
                if sig.signal_type=='BUY' and lo<=sig.stop_loss:
                    pnl=(sig.stop_loss-ep)*qty
                    exit_price = sig.stop_loss
                    exit_reason = 'STOP_LOSS'
                    break
                if sig.signal_type=='SELL' and hi>=sig.stop_loss:
                    pnl=(ep-sig.stop_loss)*qty
                    exit_price = sig.stop_loss
                    exit_reason = 'STOP_LOSS'
                    break
                    
                # Check take profits
                for tp in sig.take_profits:
                    if sig.signal_type=='BUY' and hi>=tp:
                        pnl=(tp-ep)*qty
                        exit_price = tp
                        exit_reason = 'TAKE_PROFIT'
                        break
                    if sig.signal_type=='SELL' and lo<=tp:
                        pnl=(ep-tp)*qty
                        exit_price = tp
                        exit_reason = 'TAKE_PROFIT'
                        break
                if pnl!=0: break
                
            # Enhanced journal with trade details
            journal.append({
                'timestamp': sig.timestamp,
                'entry_type': sig.entry_type,
                'signal_type': sig.signal_type,
                'entry_price': ep,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'quantity': qty,
                'pnl': pnl,
                'risk_reward': sig.risk_reward,
                'confidence': sig.confidence
            })
            bal+=pnl; eq.append(bal)
            
        dfj=pd.DataFrame(journal)
        
        # Handle empty journal case
        if len(dfj) == 0:
            wins = 0
            losses = 0
            wr = 0
            pf = 0
            sharpe_ratio = 0
        else:
            wins=dfj[dfj['pnl']>0]['pnl'].sum()
            losses=-dfj[dfj['pnl']<0]['pnl'].sum()
            wr=len(dfj[dfj['pnl']>0])/len(dfj)
            pf=wins/losses if losses>0 else np.inf
            
            # Calculate Sharpe ratio
            if len(dfj) > 1:
                returns = dfj['pnl'].values
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
                
        dd=self._max_dd(eq)
        return {'final_balance':bal,'total_trades':len(dfj),
                'win_rate':wr,'profit_factor':pf,'max_drawdown':dd,
                'sharpe_ratio':sharpe_ratio,'equity_curve':eq,'trade_journal':dfj}

    def _max_dd(self,eq):
        peak=eq[0] if eq else self.initial_balance; mdd=0
        for x in eq:
            peak=max(peak,x); mdd=max(mdd,(peak-x)/peak)
        return mdd
