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
        times=list(ohlc.index)
        for sig in signals:
            # Find the closest timestamp in the OHLC data
            if sig.timestamp not in times:
                # Find the closest timestamp
                time_diffs = [abs((pd.Timestamp(sig.timestamp) - pd.Timestamp(t)).total_seconds()) for t in times]
                if time_diffs:
                    i = time_diffs.index(min(time_diffs))
                else:
                    continue
            else:
                i=times.index(sig.timestamp)
            ep=sig.price
            risk=abs(ep-sig.stop_loss); qty=(bal*self.risk_per_trade)/risk if risk>0 else 0
            pnl=0
            for j in range(i+1,len(times)):
                hi,lo=ohlc.iloc[j][['high','low']]
                t=times[j]
                if sig.signal_type=='BUY' and lo<=sig.stop_loss:
                    pnl=(sig.stop_loss-ep)*qty; journal.append({'pnl':pnl}); break
                if sig.signal_type=='SELL' and hi>=sig.stop_loss:
                    pnl=(ep-sig.stop_loss)*qty; journal.append({'pnl':pnl}); break
                for tp in sig.take_profits:
                    if sig.signal_type=='BUY' and hi>=tp:
                        pnl=(tp-ep)*qty; journal.append({'pnl':pnl}); break
                    if sig.signal_type=='SELL' and lo<=tp:
                        pnl=(ep-tp)*qty; journal.append({'pnl':pnl}); break
                if pnl!=0: break
            bal+=pnl; eq.append(bal)
        dfj=pd.DataFrame(journal)
        
        # Handle empty journal case
        if len(dfj) == 0:
            wins = 0
            losses = 0
            wr = 0
            pf = 0
        else:
            wins=dfj[dfj['pnl']>0]['pnl'].sum()
            losses=-dfj[dfj['pnl']<0]['pnl'].sum()
            wr=len(dfj[dfj['pnl']>0])/len(dfj)
            pf=wins/losses if losses>0 else np.inf
        dd=self._max_dd(eq)
        return {'final_balance':bal,'total_trades':len(dfj),
                'win_rate':wr,'profit_factor':pf,'max_drawdown':dd,
                'equity_curve':eq,'trade_journal':dfj}

    def _max_dd(self,eq):
        peak=eq[0] if eq else self.initial_balance; mdd=0
        for x in eq:
            peak=max(peak,x); mdd=max(mdd,(peak-x)/peak)
        return mdd
