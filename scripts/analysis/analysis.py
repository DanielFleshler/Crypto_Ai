#!/usr/bin/env python3
"""
Analysis & Diagnostics Hub
Consolidated script for all analysis, diagnostics, and risk assessment tasks.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.config_loader import ConfigLoader


class AnalysisHub:
    """Main analysis class consolidating all diagnostic and analysis functions."""
    
    def __init__(self, base_path: str = 'data/raw'):
        self.data_loader = DataLoader(base_path=base_path)
        self.config_loader = ConfigLoader()
        self.base_path = Path(base_path)
    
    def diagnose_signal_bias(self, pair: str = 'BTCUSDT', 
                           start_date: str = '2023-01-01', 
                           end_date: str = '2023-03-31',
                           timeframes: List[str] = ['1h', '15m']) -> Dict:
        """
        Diagnose signal generation bias (e.g., why 99% signals are BUY in bear market).
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary with diagnostic results
        """
        print("="*70)
        print(f"SIGNAL BIAS DIAGNOSTIC: {pair}")
        print(f"Period: {start_date} to {end_date}")
        print("="*70)
        print()
        
        # Load data
        data = self.data_loader.load_pair_data(pair, timeframes, start_date, end_date)
        
        # Analyze FVG detection
        ict_detector = ICTConceptsDetector(self.config_loader)
        results = {}
        
        for tf in timeframes:
            print(f"\nAnalyzing {tf} timeframe...")
            print("-" * 50)
            
            tf_data = data[tf]
            fvgs = ict_detector.detect_fvg(tf_data)
            
            bullish_fvgs = [f for f in fvgs if f.is_bullish()]
            bearish_fvgs = [f for f in fvgs if f.is_bearish()]
            
            results[tf] = {
                'total_fvgs': len(fvgs),
                'bullish_fvgs': len(bullish_fvgs),
                'bearish_fvgs': len(bearish_fvgs),
                'bullish_pct': (len(bullish_fvgs)/len(fvgs)*100) if fvgs else 0,
                'bearish_pct': (len(bearish_fvgs)/len(fvgs)*100) if fvgs else 0,
                'imbalance_ratio': len(bullish_fvgs)/len(bearish_fvgs) if bearish_fvgs else float('inf')
            }
            
            print(f"Total FVGs: {results[tf]['total_fvgs']}")
            print(f"Bullish FVGs: {results[tf]['bullish_fvgs']} ({results[tf]['bullish_pct']:.1f}%)")
            print(f"Bearish FVGs: {results[tf]['bearish_fvgs']} ({results[tf]['bearish_pct']:.1f}%)")
            
            # Check market trend
            price_change = ((tf_data['close'].iloc[-1] - tf_data['close'].iloc[0]) / 
                          tf_data['close'].iloc[0] * 100)
            
            if price_change < -10:
                market_trend = "BEARISH"
            elif price_change > 10:
                market_trend = "BULLISH"
            else:
                market_trend = "SIDEWAYS"
                
            results[tf]['market_trend'] = market_trend
            results[tf]['price_change_pct'] = price_change
            
            print(f"Market trend: {market_trend} ({price_change:+.1f}%)")
            
            # Issue detection
            if market_trend == "BEARISH" and results[tf]['imbalance_ratio'] > 2:
                print("\n⚠️  ISSUE DETECTED: Bullish bias in bearish market!")
                print(f"   Bullish FVGs outnumber Bearish by {results[tf]['imbalance_ratio']:.1f}x")
        
        # Provide recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        recommendations = []
        
        for tf, data in results.items():
            if data['market_trend'] == "BEARISH" and data['imbalance_ratio'] > 2:
                recommendations.append(
                    f"- {tf}: Implement counter-trend logic for bearish markets"
                )
        
        if recommendations:
            print("\n".join(recommendations))
            print("\nSuggested fixes:")
            print("1. In BEARISH bias: Trade bearish FVGs (continuation)")
            print("2. OR: Use bullish FVGs as SELL entry zones (wait for rejection)")
            print("3. Adjust HTF bias filtering to not exclude opposing FVGs")
        else:
            print("✓ No significant signal bias detected")
        
        return results
    
    def analyze_risk_levels(self, win_rate: float = 0.3347, 
                          avg_rr: float = 3.0,
                          total_trades: int = 43,
                          consecutive_losses: int = 11,
                          initial_balance: float = 10000) -> Dict:
        """
        Analyze impact of different risk per trade levels with Kelly Criterion.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_rr: Average risk-reward ratio
            total_trades: Total number of trades
            consecutive_losses: Maximum consecutive losses observed
            initial_balance: Starting balance
            
        Returns:
            Dictionary with risk analysis results
        """
        print("="*70)
        print("RISK PER TRADE ANALYSIS")
        print(f"Win Rate: {win_rate*100:.2f}%, R:R: 1:{avg_rr}")
        print("="*70)
        print()
        
        risk_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
        results = {'risk_levels': {}}
        
        for risk_pct in risk_levels:
            print(f"Risk Per Trade: {risk_pct*100}%")
            print("-" * 50)
            
            # Calculate drawdown from consecutive losses
            balance = initial_balance
            for i in range(consecutive_losses):
                loss = balance * risk_pct
                balance -= loss
            
            drawdown_pct = ((initial_balance - balance) / initial_balance) * 100
            
            # Expected value per trade
            ev_per_trade = (win_rate * (risk_pct * avg_rr)) - ((1-win_rate) * risk_pct)
            total_ev = ev_per_trade * total_trades * initial_balance
            
            # Risk assessment
            if drawdown_pct > 50:
                risk_level = "EXTREME"
            elif drawdown_pct > 30:
                risk_level = "HIGH"
            elif drawdown_pct > 15:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            results['risk_levels'][risk_pct] = {
                'drawdown_pct': drawdown_pct,
                'balance_after_losses': balance,
                'expected_value': total_ev,
                'expected_return_pct': (total_ev/initial_balance)*100,
                'risk_level': risk_level
            }
            
            print(f"  Worst drawdown ({consecutive_losses} losses): {drawdown_pct:.2f}%")
            print(f"  Balance after drawdown: ${balance:.2f}")
            print(f"  Risk level: {risk_level}")
            print(f"  Expected value: ${total_ev:+.2f} ({(total_ev/initial_balance)*100:+.2f}%)")
            print()
        
        # Kelly Criterion calculation
        print("="*70)
        print("KELLY CRITERION OPTIMAL RISK")
        print("="*70)
        
        b = avg_rr
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        half_kelly = kelly_fraction / 2
        quarter_kelly = kelly_fraction / 4
        
        results['kelly'] = {
            'full_kelly': kelly_fraction,
            'half_kelly': half_kelly,
            'quarter_kelly': quarter_kelly,
            'recommended_range': (max(0.01, quarter_kelly), min(0.10, half_kelly))
        }
        
        print(f"Full Kelly: {kelly_fraction*100:.2f}% per trade")
        print(f"Half Kelly (conservative): {half_kelly*100:.2f}% per trade")
        print(f"Quarter Kelly (very conservative): {quarter_kelly*100:.2f}% per trade")
        print()
        
        if kelly_fraction <= 0:
            print("⚠️  WARNING: Negative Kelly = Strategy has negative expectation!")
            print("    Don't trade this strategy until win rate or R:R improves")
        else:
            rec_min = results['kelly']['recommended_range'][0]
            rec_max = results['kelly']['recommended_range'][1]
            print(f"Recommended range: {rec_min*100:.1f}% to {rec_max*100:.1f}% per trade")
        
        return results
    
    def analyze_market_regime(self, pair: str, timeframe: str,
                            start_date: str, end_date: str) -> Dict:
        """
        Analyze market regime and characteristics.
        
        Args:
            pair: Trading pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with market regime analysis
        """
        print(f"\nAnalyzing market regime for {pair} ({start_date} to {end_date})...")
        
        # Load data
        data = self.data_loader.load_pair_data(pair, [timeframe], start_date, end_date)
        df = data[timeframe]
        
        # Calculate metrics
        returns = df['close'].pct_change()
        
        # Trend analysis
        x = np.arange(len(df))
        y = df['close'].values
        trend_slope = np.polyfit(x, y, 1)[0]
        
        # Volatility
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized for hourly
        
        # Price change
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).mean()
        atr_pct = (atr / df['close'].mean()) * 100
        
        # Classify regime
        if abs(price_change) < 10:
            trend = 'sideways'
        elif price_change > 10:
            trend = 'bull'
        else:
            trend = 'bear'
        
        if volatility > 1.5:
            vol_regime = 'high_volatility'
        elif volatility < 0.5:
            vol_regime = 'low_volatility'
        else:
            vol_regime = 'normal_volatility'
        
        regime = f"{trend}_{vol_regime}"
        
        return {
            'regime': regime,
            'trend': trend,
            'volatility_regime': vol_regime,
            'price_change_pct': price_change,
            'annualized_volatility': volatility,
            'trend_slope': trend_slope,
            'atr_pct': atr_pct,
            'total_candles': len(df)
        }
    
    def analyze_session_performance(self, pair: str, 
                                  start_date: str, 
                                  end_date: str) -> Dict:
        """
        Analyze historical price movements by trading session.
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with session performance metrics
        """
        print(f"\nAnalyzing session performance for {pair}...")
        
        # Load hourly data
        data = self.data_loader.load_pair_data(pair, ['1h'], start_date, end_date)
        df = data['1h']
        
        # Define sessions (UTC times)
        sessions = {
            'asia': (0, 8),
            'london': (8, 13),
            'london_ny': (13, 16),
            'ny': (16, 21),
            'off_hours': (21, 24)
        }
        
        results = {}
        
        for session_name, (start_hour, end_hour) in sessions.items():
            # Filter data for session
            if start_hour < end_hour:
                session_mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
            else:  # Handle off_hours wrap
                session_mask = (df.index.hour >= start_hour) | (df.index.hour < end_hour)
            
            session_data = df[session_mask]
            
            if len(session_data) > 0:
                # Calculate metrics
                returns = session_data['close'].pct_change()
                
                results[session_name] = {
                    'avg_return': returns.mean() * 100,
                    'volatility': returns.std() * 100,
                    'total_candles': len(session_data),
                    'positive_hours': (returns > 0).sum(),
                    'negative_hours': (returns < 0).sum(),
                    'win_rate': (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0,
                    'avg_range_pct': ((session_data['high'] - session_data['low']) / session_data['close'] * 100).mean()
                }
        
        # Display results
        print("\n" + "-"*70)
        print(f"{'Session':<15} {'Avg Return':<12} {'Volatility':<12} {'Win Rate':<10} {'Avg Range'}")
        print("-"*70)
        
        for session, metrics in results.items():
            print(f"{session.upper():<15} "
                  f"{metrics['avg_return']:>10.3f}% "
                  f"{metrics['volatility']:>10.3f}% "
                  f"{metrics['win_rate']:>8.1f}% "
                  f"{metrics['avg_range_pct']:>8.2f}%")
        
        return results
    
    def comprehensive_analysis(self, pair: str = 'BTCUSDT',
                             start_date: str = '2023-01-01',
                             end_date: str = '2023-12-31') -> Dict:
        """
        Run comprehensive analysis including all diagnostic functions.
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Market regime analysis
        print("\n1. MARKET REGIME ANALYSIS")
        results['market_regime'] = self.analyze_market_regime(pair, '1h', start_date, end_date)
        
        # 2. Signal bias diagnostic
        print("\n2. SIGNAL BIAS DIAGNOSTIC")
        results['signal_bias'] = self.diagnose_signal_bias(pair, start_date, end_date)
        
        # 3. Session performance
        print("\n3. SESSION PERFORMANCE ANALYSIS")
        results['session_performance'] = self.analyze_session_performance(pair, start_date, end_date)
        
        # 4. Risk analysis (using default parameters)
        print("\n4. RISK LEVEL ANALYSIS")
        results['risk_analysis'] = self.analyze_risk_levels()
        
        return results


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Trading Strategy Analysis Hub')
    parser.add_argument('command', choices=['signal-bias', 'risk-analysis', 'market-regime', 
                                          'session-performance', 'comprehensive'],
                       help='Analysis command to run')
    parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--start', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe for analysis')
    
    # Risk analysis specific arguments
    parser.add_argument('--win-rate', type=float, default=0.3347, help='Win rate (0-1)')
    parser.add_argument('--rr-ratio', type=float, default=3.0, help='Risk-reward ratio')
    parser.add_argument('--trades', type=int, default=43, help='Total trades')
    parser.add_argument('--max-losses', type=int, default=11, help='Max consecutive losses')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    
    args = parser.parse_args()
    
    # Initialize analysis hub
    analyzer = AnalysisHub()
    
    # Execute requested command
    if args.command == 'signal-bias':
        results = analyzer.diagnose_signal_bias(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end
        )
    
    elif args.command == 'risk-analysis':
        results = analyzer.analyze_risk_levels(
            win_rate=args.win_rate,
            avg_rr=args.rr_ratio,
            total_trades=args.trades,
            consecutive_losses=args.max_losses,
            initial_balance=args.balance
        )
    
    elif args.command == 'market-regime':
        results = analyzer.analyze_market_regime(
            pair=args.pair,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end
        )
        print(f"\nMarket Regime: {results['regime'].upper()}")
        print(f"Price Change: {results['price_change_pct']:+.2f}%")
        print(f"Volatility: {results['annualized_volatility']:.2f}")
    
    elif args.command == 'session-performance':
        results = analyzer.analyze_session_performance(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end
        )
    
    elif args.command == 'comprehensive':
        results = analyzer.comprehensive_analysis(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end
        )
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
