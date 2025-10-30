#!/usr/bin/env python3
"""
Realistic Backtest Simulation
Comprehensive testing across multiple market regimes with detailed performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from backtester import BacktestEngine
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.config_loader import ConfigLoader
import warnings
warnings.filterwarnings('ignore')


class RealisticBacktester:
    """
    Comprehensive backtesting system for production-ready validation
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.base_path = Path(__file__).parent
        self.data_loader = DataLoader(base_path='data/raw')
        self.config_loader = ConfigLoader()
        self.results = {}
        
    def identify_market_regime(self, df: pd.DataFrame, window: int = 90) -> str:
        """
        Identify market regime using multiple indicators
        """
        # Calculate returns and volatility
        returns = df['close'].pct_change()
        
        # Price trend (90-day)
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[-window]) / 
                       df['close'].iloc[-window]) * 100
        
        # Volatility (annualized)
        volatility = returns.rolling(window).std() * np.sqrt(365 * 24)
        current_vol = volatility.iloc[-1]
        
        # Moving averages
        ma_50 = df['close'].rolling(50).mean()
        ma_200 = df['close'].rolling(200).mean()
        
        # Trend strength
        if len(df) >= 200:
            trend_strength = (ma_50.iloc[-1] - ma_200.iloc[-1]) / ma_200.iloc[-1] * 100
        else:
            trend_strength = price_change
        
        # Classify regime
        regime_parts = []
        
        # Trend classification
        if price_change > 20:
            regime_parts.append("strong_bull")
        elif price_change > 10:
            regime_parts.append("bull")
        elif price_change < -20:
            regime_parts.append("strong_bear")
        elif price_change < -10:
            regime_parts.append("bear")
        else:
            regime_parts.append("sideways")
        
        # Volatility classification
        if current_vol > 1.2:
            regime_parts.append("high_vol")
        elif current_vol < 0.4:
            regime_parts.append("low_vol")
        else:
            regime_parts.append("normal_vol")
        
        return "_".join(regime_parts)
    
    def define_test_periods(self) -> Dict[str, Dict]:
        """
        Define comprehensive test periods covering various market conditions
        """
        return {
            # Training period (In-sample)
            "2021_Bull_Run": {
                "start": "2021-01-01",
                "end": "2021-05-31",
                "type": "in_sample",
                "description": "Major bull run to ATH"
            },
            "2021_Crash": {
                "start": "2021-05-01",
                "end": "2021-07-31",
                "type": "in_sample",
                "description": "50% crash from ATH"
            },
            "2021_Recovery": {
                "start": "2021-08-01",
                "end": "2021-11-30",
                "type": "in_sample",
                "description": "Recovery to new ATH"
            },
            
            # 2022 - Bear market (In-sample)
            "2022_Q1_Bear": {
                "start": "2022-01-01",
                "end": "2022-03-31",
                "type": "in_sample",
                "description": "Beginning of bear market"
            },
            "2022_Q2_Crash": {
                "start": "2022-04-01",
                "end": "2022-06-30",
                "type": "in_sample",
                "description": "Luna/FTX crash period"
            },
            "2022_Q3_Sideways": {
                "start": "2022-07-01",
                "end": "2022-09-30",
                "type": "in_sample",
                "description": "Low volatility sideways"
            },
            "2022_Q4_Capitulation": {
                "start": "2022-10-01",
                "end": "2022-12-31",
                "type": "in_sample",
                "description": "Final capitulation"
            },
            
            # 2023-2024 - Out of sample
            "2023_Q1_Recovery": {
                "start": "2023-01-01",
                "end": "2023-03-31",
                "type": "out_of_sample",
                "description": "Early recovery signs"
            },
            "2023_Q2_Rally": {
                "start": "2023-04-01",
                "end": "2023-06-30",
                "type": "out_of_sample",
                "description": "Strong rally"
            },
            "2023_Q3_Consolidation": {
                "start": "2023-07-01",
                "end": "2023-09-30",
                "type": "out_of_sample",
                "description": "Consolidation phase"
            },
            "2023_Q4_Breakout": {
                "start": "2023-10-01",
                "end": "2023-12-31",
                "type": "out_of_sample",
                "description": "Breakout preparation"
            },
            "2024_Bull_Market": {
                "start": "2024-01-01",
                "end": "2024-10-18",
                "type": "out_of_sample",
                "description": "New bull market"
            }
        }
    
    def run_period_backtest(self, pair: str, period_name: str, 
                          period_info: Dict) -> Dict:
        """
        Run backtest for a specific period with detailed metrics
        """
        print(f"\n{'='*80}")
        print(f"Testing Period: {period_name}")
        print(f"Date Range: {period_info['start']} to {period_info['end']}")
        print(f"Type: {period_info['type'].upper()}")
        print(f"Description: {period_info['description']}")
        print("="*80)
        
        try:
            # Initialize components
            strategy = TradingStrategy(
                base_path=str(self.base_path),
                config_loader=self.config_loader
            )
            
            engine = BacktestEngine(
                base_path=str(self.base_path),
                config_loader=self.config_loader,
                initial_balance=self.initial_balance
            )
            
            # Run backtest
            result = engine.run_backtest(
                pair=pair,
                start_date=period_info['start'],
                end_date=period_info['end']
            )
            
            # Load data to identify regime
            data = self.data_loader.load_pair_data(
                pair, ['1h'], period_info['start'], period_info['end']
            )
            
            if '1h' in data and not data['1h'].empty:
                regime = self.identify_market_regime(data['1h'])
            else:
                regime = "unknown"
            
            # Calculate detailed metrics
            metrics = self.calculate_comprehensive_metrics(result, regime)
            
            # Add period info
            metrics['period_name'] = period_name
            metrics['period_type'] = period_info['type']
            metrics['start_date'] = period_info['start']
            metrics['end_date'] = period_info['end']
            metrics['market_regime'] = regime
            
            # Display key metrics
            self.display_period_results(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"ERROR in {period_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'period_name': period_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def calculate_comprehensive_metrics(self, backtest_result: Dict, 
                                      regime: str) -> Dict:
        """
        Calculate all required performance metrics
        """
        # Extract trade data
        trade_journal = backtest_result.get('trade_journal', pd.DataFrame())
        trades = trade_journal.to_dict('records') if not trade_journal.empty else []
        equity_curve = backtest_result.get('equity_curve', [self.initial_balance])
        
        if not trades:
            return self._empty_metrics()
        
        # Basic statistics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        
        # Win rate
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss calculations
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        net_profit = total_profit - total_loss
        
        # Profit factor
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Average trade statistics
        avg_win = (total_profit / num_wins) if num_wins > 0 else 0
        avg_loss = (total_loss / num_losses) if num_losses > 0 else 0
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        
        # Expectancy
        expectancy = ((win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss))
        
        # Risk metrics
        returns = pd.Series([equity_curve[i]/equity_curve[i-1] - 1 
                           for i in range(1, len(equity_curve))])
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_dd = 0
        max_dd_duration = 0
        current_dd_start = None
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if current_dd_start is not None:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    current_dd_start = None
            else:
                if current_dd_start is None:
                    current_dd_start = i
                dd = ((peak - value) / peak) * 100
                max_dd = max(max_dd, dd)
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252 * 24) if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Calmar ratio (return / max drawdown)
        total_return = ((equity_curve[-1] - equity_curve[0]) / equity_curve[0]) * 100
        calmar_ratio = (total_return / max_dd) if max_dd > 0 else float('inf')
        
        # Win/Loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        is_win_streak = None
        
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                if is_win_streak:
                    current_streak += 1
                else:
                    is_win_streak = True
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                if is_win_streak is False:
                    current_streak += 1
                else:
                    is_win_streak = False
                    current_streak = 1
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Session performance
        session_performance = self.analyze_session_performance(trades)
        
        # Risk-reward analysis
        risk_rewards = []
        for trade in trades:
            if 'entry_price' in trade and 'stop_loss' in trade and 'exit_price' in trade:
                risk = abs(trade['entry_price'] - trade['stop_loss'])
                reward = abs(trade['exit_price'] - trade['entry_price'])
                if risk > 0:
                    risk_rewards.append(reward / risk)
        
        avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0
        
        return {
            # Basic metrics
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            
            # Financial metrics
            'net_profit': net_profit,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'expectancy': expectancy,
            'expectancy_pct': (expectancy / self.initial_balance) * 100,
            
            # Risk metrics
            'max_drawdown_pct': max_dd,
            'max_dd_duration': max_dd_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Trade quality
            'avg_risk_reward': avg_risk_reward,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            
            # Session analysis
            'session_performance': session_performance,
            
            # Additional info
            'final_balance': equity_curve[-1] if equity_curve else self.initial_balance,
            'market_regime': regime
        }
    
    def analyze_session_performance(self, trades: List[Dict]) -> Dict:
        """
        Analyze performance by trading session
        """
        sessions = {
            'asia': {'trades': 0, 'wins': 0, 'pnl': 0, 'hours': (0, 8)},
            'london': {'trades': 0, 'wins': 0, 'pnl': 0, 'hours': (8, 13)},
            'london_ny': {'trades': 0, 'wins': 0, 'pnl': 0, 'hours': (13, 16)},
            'ny': {'trades': 0, 'wins': 0, 'pnl': 0, 'hours': (16, 21)},
            'off_hours': {'trades': 0, 'wins': 0, 'pnl': 0, 'hours': (21, 24)}
        }
        
        for trade in trades:
            timestamp = trade.get('timestamp')
            if not timestamp:
                continue
                
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            hour = timestamp.hour
            
            # Determine session
            session_name = None
            for name, info in sessions.items():
                start_hour, end_hour = info['hours']
                if start_hour <= hour < end_hour:
                    session_name = name
                    break
            
            if session_name:
                sessions[session_name]['trades'] += 1
                sessions[session_name]['pnl'] += trade.get('pnl', 0)
                if trade.get('pnl', 0) > 0:
                    sessions[session_name]['wins'] += 1
        
        # Calculate win rates and avg pnl
        for session in sessions.values():
            if session['trades'] > 0:
                session['win_rate'] = (session['wins'] / session['trades']) * 100
                session['avg_pnl'] = session['pnl'] / session['trades']
            else:
                session['win_rate'] = 0
                session['avg_pnl'] = 0
        
        return sessions
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'net_profit': 0,
            'total_return_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_trade': 0,
            'expectancy': 0,
            'expectancy_pct': 0,
            'max_drawdown_pct': 0,
            'max_dd_duration': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'avg_risk_reward': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'session_performance': {},
            'final_balance': self.initial_balance,
            'market_regime': 'unknown'
        }
    
    def display_period_results(self, metrics: Dict):
        """Display formatted results for a period"""
        print(f"\nResults for {metrics.get('period_name', 'Unknown')}:")
        print(f"Market Regime: {metrics.get('market_regime', 'Unknown')}")
        print("-" * 60)
        
        # Performance metrics
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Expectancy: ${metrics['expectancy']:.2f} ({metrics['expectancy_pct']:.2f}%)")
        
        # Returns
        print(f"\nTotal Return: {metrics['total_return_pct']:.2f}%")
        print(f"Net Profit: ${metrics['net_profit']:.2f}")
        print(f"Final Balance: ${metrics['final_balance']:.2f}")
        
        # Risk metrics
        print(f"\nMax Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        
        # Trade quality
        print(f"\nAvg Risk/Reward: {metrics['avg_risk_reward']:.2f}")
        print(f"Max Win Streak: {metrics['max_win_streak']}")
        print(f"Max Loss Streak: {metrics['max_loss_streak']}")
    
    def generate_comprehensive_report(self, all_results: List[Dict]):
        """Generate final comprehensive report"""
        print("\n" + "="*100)
        print("COMPREHENSIVE BACKTEST REPORT")
        print("="*100)
        
        # Separate in-sample and out-of-sample results
        in_sample = [r for r in all_results if r.get('period_type') == 'in_sample' and r.get('status') != 'failed']
        out_sample = [r for r in all_results if r.get('period_type') == 'out_of_sample' and r.get('status') != 'failed']
        
        # Summary statistics
        print("\n1. OVERALL PERFORMANCE")
        print("-" * 80)
        
        # In-sample performance
        if in_sample:
            avg_win_rate_is = np.mean([r['win_rate'] for r in in_sample])
            avg_pf_is = np.mean([r['profit_factor'] for r in in_sample])
            avg_sharpe_is = np.mean([r['sharpe_ratio'] for r in in_sample])
            max_dd_is = max([r['max_drawdown_pct'] for r in in_sample])
            
            print(f"\nIn-Sample (Training) Performance:")
            print(f"  Average Win Rate: {avg_win_rate_is:.2f}%")
            print(f"  Average Profit Factor: {avg_pf_is:.2f}")
            print(f"  Average Sharpe Ratio: {avg_sharpe_is:.2f}")
            print(f"  Maximum Drawdown: {max_dd_is:.2f}%")
        
        # Out-of-sample performance
        if out_sample:
            avg_win_rate_oos = np.mean([r['win_rate'] for r in out_sample])
            avg_pf_oos = np.mean([r['profit_factor'] for r in out_sample])
            avg_sharpe_oos = np.mean([r['sharpe_ratio'] for r in out_sample])
            max_dd_oos = max([r['max_drawdown_pct'] for r in out_sample])
            
            print(f"\nOut-of-Sample (Test) Performance:")
            print(f"  Average Win Rate: {avg_win_rate_oos:.2f}%")
            print(f"  Average Profit Factor: {avg_pf_oos:.2f}")
            print(f"  Average Sharpe Ratio: {avg_sharpe_oos:.2f}")
            print(f"  Maximum Drawdown: {max_dd_oos:.2f}%")
            
            # Performance degradation check
            if in_sample:
                degradation = ((avg_pf_is - avg_pf_oos) / avg_pf_is) * 100 if avg_pf_is > 0 else 0
                print(f"\n  Performance Degradation: {degradation:.1f}%")
                if degradation > 30:
                    print("  ⚠️  WARNING: Significant performance degradation - possible overfitting")
                elif degradation > 15:
                    print("  ⚠️  CAUTION: Moderate performance degradation")
                else:
                    print("  ✅ Good consistency between training and test periods")
        
        # Performance by market regime
        print("\n2. PERFORMANCE BY MARKET REGIME")
        print("-" * 80)
        
        regime_performance = {}
        for result in all_results:
            if result.get('status') == 'failed':
                continue
            regime = result.get('market_regime', 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result)
        
        for regime, results in regime_performance.items():
            avg_trades = np.mean([r['total_trades'] for r in results])
            avg_win_rate = np.mean([r['win_rate'] for r in results])
            avg_pf = np.mean([r['profit_factor'] for r in results])
            
            print(f"\n{regime.upper()}:")
            print(f"  Periods tested: {len(results)}")
            print(f"  Avg trades/period: {avg_trades:.1f}")
            print(f"  Avg win rate: {avg_win_rate:.2f}%")
            print(f"  Avg profit factor: {avg_pf:.2f}")
        
        # Session performance
        print("\n3. PERFORMANCE BY SESSION")
        print("-" * 80)
        
        all_session_data = {}
        for result in all_results:
            if result.get('status') == 'failed':
                continue
            session_perf = result.get('session_performance', {})
            for session, data in session_perf.items():
                if session not in all_session_data:
                    all_session_data[session] = []
                all_session_data[session].append(data)
        
        print(f"\n{'Session':<15} {'Trades':<10} {'Win Rate':<12} {'Avg P&L':<12}")
        print("-" * 50)
        
        for session, data_list in all_session_data.items():
            total_trades = sum(d['trades'] for d in data_list)
            avg_win_rate = np.mean([d['win_rate'] for d in data_list if d['trades'] > 0])
            avg_pnl = np.mean([d['avg_pnl'] for d in data_list if d['trades'] > 0])
            
            print(f"{session.upper():<15} {total_trades:<10} {avg_win_rate:<12.2f} ${avg_pnl:<11.2f}")
        
        # Final assessment
        print("\n4. FINAL ASSESSMENT")
        print("-" * 80)
        
        # Calculate overall score
        all_valid_results = [r for r in all_results if r.get('status') != 'failed']
        if all_valid_results:
            overall_pf = np.mean([r['profit_factor'] for r in all_valid_results])
            overall_sharpe = np.mean([r['sharpe_ratio'] for r in all_valid_results])
            overall_win_rate = np.mean([r['win_rate'] for r in all_valid_results])
            max_drawdown = max([r['max_drawdown_pct'] for r in all_valid_results])
            
            # Score calculation
            score = 0
            score += 20 if overall_win_rate >= 40 else (10 if overall_win_rate >= 30 else 0)
            score += 30 if overall_pf >= 2.0 else (20 if overall_pf >= 1.5 else (10 if overall_pf >= 1.2 else 0))
            score += 20 if max_drawdown <= 15 else (10 if max_drawdown <= 25 else 0)
            score += 20 if overall_sharpe >= 1.0 else (10 if overall_sharpe >= 0.5 else 0)
            score += 10 if out_sample and degradation <= 15 else 0
            
            print(f"\nStrategy Score: {score}/100")
            
            if score >= 80:
                print("Rating: EXCELLENT - Ready for live trading with careful monitoring")
            elif score >= 60:
                print("Rating: GOOD - Consider paper trading first")
            elif score >= 40:
                print("Rating: FAIR - Needs optimization before live trading")
            else:
                print("Rating: POOR - Significant improvements required")
            
            print(f"\nKey Metrics Summary:")
            print(f"  Profit Factor: {overall_pf:.2f} {'✅' if overall_pf >= 1.5 else '❌'}")
            print(f"  Max Drawdown: {max_drawdown:.2f}% {'✅' if max_drawdown <= 15 else '⚠️' if max_drawdown <= 25 else '❌'}")
            print(f"  Sharpe Ratio: {overall_sharpe:.2f} {'✅' if overall_sharpe >= 1.0 else '⚠️' if overall_sharpe >= 0.5 else '❌'}")
            print(f"  Win Rate: {overall_win_rate:.2f}% {'✅' if overall_win_rate >= 40 else '⚠️' if overall_win_rate >= 30 else '❌'}")
        
        print("\n" + "="*100)
    
    def save_results(self, all_results: List[Dict], filename: str = None):
        """Save detailed results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"realistic_backtest_results_{timestamp}.json"
        
        # Convert any non-serializable objects
        def convert_for_json(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        clean_results = json.loads(json.dumps(all_results, default=convert_for_json))
        
        with open(filename, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'initial_balance': self.initial_balance,
                'results': clean_results,
                'summary': self._generate_summary(all_results)
            }, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _generate_summary(self, all_results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        valid_results = [r for r in all_results if r.get('status') != 'failed']
        in_sample = [r for r in valid_results if r.get('period_type') == 'in_sample']
        out_sample = [r for r in valid_results if r.get('period_type') == 'out_of_sample']
        
        return {
            'total_periods_tested': len(all_results),
            'successful_tests': len(valid_results),
            'failed_tests': len(all_results) - len(valid_results),
            'in_sample_periods': len(in_sample),
            'out_of_sample_periods': len(out_sample),
            'overall_metrics': {
                'avg_profit_factor': np.mean([r['profit_factor'] for r in valid_results]) if valid_results else 0,
                'avg_win_rate': np.mean([r['win_rate'] for r in valid_results]) if valid_results else 0,
                'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in valid_results]) if valid_results else 0,
                'max_drawdown': max([r['max_drawdown_pct'] for r in valid_results]) if valid_results else 0,
                'total_trades': sum([r['total_trades'] for r in valid_results]) if valid_results else 0
            }
        }
    
    def run_comprehensive_backtest(self):
        """Main method to run the complete backtest"""
        print("\n" + "="*100)
        print("STARTING REALISTIC BACKTEST SIMULATION")
        print("Testing Elliott Wave + ICT Strategy across multiple market regimes")
        print("="*100)
        
        # Define test periods
        test_periods = self.define_test_periods()
        
        # Run backtests
        all_results = []
        pair = 'BTCUSDT'
        
        for period_name, period_info in test_periods.items():
            result = self.run_period_backtest(pair, period_name, period_info)
            all_results.append(result)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results


def main():
    """Main entry point"""
    # Initialize backtester
    backtester = RealisticBacktester(initial_balance=10000)
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest()
    
    print("\n✅ Realistic backtest simulation complete!")
    print("\nNext steps:")
    print("1. Review the detailed results in the JSON file")
    print("2. Analyze periods where strategy underperformed")
    print("3. Consider parameter optimization for weak regimes")
    print("4. Implement suggested improvements before live trading")


if __name__ == '__main__':
    main()
