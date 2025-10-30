#!/usr/bin/env python3
"""
Unified Backtesting Interface
Consolidates all backtesting functionality into a single, powerful script.

Features:
- Quick validation backtests
- Single period analysis
- Multi-period regime testing
- Walk-forward analysis
- Parameter optimization
- Comprehensive reporting
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from backtester import BacktestEngine
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.config_loader import ConfigLoader


class UnifiedBacktester:
    """Unified backtesting interface with multiple modes and options."""
    
    def __init__(self, base_path: str = '.', initial_balance: float = 10000):
        self.base_path = Path(base_path)
        self.initial_balance = initial_balance
        self.data_loader = DataLoader(base_path='data/raw')
        self.config_loader = ConfigLoader()
        
    def quick_validation(self, pair: str = 'BTCUSDT',
                        start_date: str = '2023-01-01',
                        end_date: str = '2023-03-31',
                        expected_trades: int = 100) -> Dict:
        """
        Quick validation backtest to verify strategy changes.
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            expected_trades: Expected minimum number of trades
            
        Returns:
            Validation results
        """
        print("="*70)
        print("QUICK VALIDATION BACKTEST")
        print(f"Period: {start_date} to {end_date}")
        print("="*70)
        print()
        
        # Initialize backtester
        backtester = BacktestEngine(
            base_path=str(self.base_path),
            initial_balance=self.initial_balance
        )
        
        # Run backtest
        result = backtester.run_backtest(
            pair=pair,
            start_date=start_date,
            end_date=end_date
        )
        
        # Analyze results
        total_trades = len(result.get('trade_journal', []))
        final_balance = result.get('final_balance', self.initial_balance)
        pnl = final_balance - self.initial_balance
        return_pct = (pnl / self.initial_balance) * 100
        
        print(f"Total Trades: {total_trades}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"P&L: ${pnl:+.2f}")
        print(f"Return: {return_pct:+.2f}%")
        print()
        
        # Validation check
        if total_trades >= expected_trades:
            print(f"✅ PASSED: {total_trades} trades >= {expected_trades} expected")
            validation_passed = True
        else:
            print(f"❌ FAILED: {total_trades} trades < {expected_trades} expected")
            validation_passed = False
        
        return {
            'validation_passed': validation_passed,
            'total_trades': total_trades,
            'expected_trades': expected_trades,
            'final_balance': final_balance,
            'pnl': pnl,
            'return_pct': return_pct
        }
    
    def single_period_backtest(self, pair: str, start_date: str, end_date: str,
                             timeframe: str = '1h', detailed: bool = True) -> Dict:
        """
        Run a detailed backtest for a single period.
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            timeframe: Primary timeframe
            detailed: Whether to calculate detailed metrics
            
        Returns:
            Backtest results
        """
        print(f"\nRunning single period backtest: {pair} ({start_date} to {end_date})")
        print("-" * 70)
        
        engine = BacktestEngine(
            base_path=str(self.base_path),
            initial_balance=self.initial_balance,
            risk_per_trade=self.config_loader.get_risk_management_config().max_risk_per_trade
        )
        
        result = engine.run_backtest(pair, start_date, end_date)
        
        if detailed:
            # Calculate advanced metrics
            metrics = self._calculate_advanced_metrics(result)
            result['advanced_metrics'] = metrics
            
            # Display detailed results
            self._display_detailed_results(result, metrics)
        
        return result
    
    def multi_period_backtest(self, pair: str = 'BTCUSDT',
                            periods: Optional[Dict] = None) -> Dict:
        """
        Run backtests across multiple market regimes.
        
        Args:
            pair: Trading pair
            periods: Dictionary of period definitions
            
        Returns:
            Results for all periods
        """
        if periods is None:
            # Default periods covering different market regimes
            periods = {
                'Bull_2021': ('2021-07-01', '2021-11-30'),
                'Bear_2022': ('2022-01-01', '2022-12-31'),
                'Sideways_2023': ('2023-01-01', '2023-12-31'),
                'Recent_2024': ('2024-01-01', '2024-10-18')
            }
        
        print("\n" + "="*80)
        print("MULTI-PERIOD BACKTEST ANALYSIS")
        print(f"Testing {pair} across {len(periods)} market regimes")
        print("="*80)
        
        all_results = {}
        
        for period_name, (start_date, end_date) in periods.items():
            print(f"\n{'='*60}")
            print(f"Period: {period_name}")
            print(f"{'='*60}")
            
            try:
                result = self.single_period_backtest(
                    pair=pair,
                    start_date=start_date,
                    end_date=end_date,
                    detailed=True
                )
                all_results[period_name] = result
                
            except Exception as e:
                print(f"ERROR in {period_name}: {str(e)}")
                all_results[period_name] = {'error': str(e), 'status': 'failed'}
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def walk_forward_analysis(self, pair: str, start_date: str, end_date: str,
                            train_months: int = 6, test_months: int = 1) -> Dict:
        """
        Perform walk-forward analysis.
        
        Args:
            pair: Trading pair
            start_date: Overall start date
            end_date: Overall end date
            train_months: Training period length in months
            test_months: Testing period length in months
            
        Returns:
            Walk-forward analysis results
        """
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS")
        print(f"Train: {train_months} months, Test: {test_months} months")
        print("="*80)
        
        results = []
        current_date = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        step = 1
        while current_date < end_dt:
            # Define train period
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            
            # Define test period
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > end_dt:
                break
            
            print(f"\nStep {step}:")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test: {test_start.date()} to {test_end.date()}")
            
            # Run in-sample (training) backtest
            train_result = self.single_period_backtest(
                pair=pair,
                start_date=str(train_start.date()),
                end_date=str(train_end.date()),
                detailed=False
            )
            
            # Run out-of-sample (test) backtest
            test_result = self.single_period_backtest(
                pair=pair,
                start_date=str(test_start.date()),
                end_date=str(test_end.date()),
                detailed=False
            )
            
            results.append({
                'step': step,
                'train_period': (str(train_start.date()), str(train_end.date())),
                'test_period': (str(test_start.date()), str(test_end.date())),
                'train_result': train_result,
                'test_result': test_result
            })
            
            # Move to next period
            current_date = current_date + pd.DateOffset(months=test_months)
            step += 1
        
        # Analyze walk-forward results
        self._analyze_walk_forward_results(results)
        
        return results
    
    def parameter_optimization(self, pair: str, start_date: str, end_date: str,
                             param_ranges: Optional[Dict] = None) -> Dict:
        """
        Optimize strategy parameters.
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Optimization results
        """
        if param_ranges is None:
            # Default parameter ranges
            param_ranges = {
                'risk_per_trade': [0.01, 0.02, 0.03, 0.05],
                'rr_ratio': [2.0, 2.5, 3.0, 4.0],
                'lookback_periods': [20, 50, 100, 200]
            }
        
        print("\n" + "="*80)
        print("PARAMETER OPTIMIZATION")
        print("="*80)
        
        results = []
        best_result = None
        best_sharpe = -float('inf')
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            print(f"\nTesting: {params}")
            
            # Modify config temporarily
            # Note: This is simplified - in practice you'd want to properly handle config updates
            
            try:
                result = self.single_period_backtest(
                    pair=pair,
                    start_date=start_date,
                    end_date=end_date,
                    detailed=True
                )
                
                sharpe = result.get('advanced_metrics', {}).get('sharpe_ratio', 0)
                
                results.append({
                    'params': params,
                    'sharpe_ratio': sharpe,
                    'total_return': result.get('return_pct', 0),
                    'max_drawdown': result.get('advanced_metrics', {}).get('max_drawdown_pct', 0),
                    'total_trades': result.get('advanced_metrics', {}).get('total_trades', 0)
                })
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = params
                    
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        print("\n" + "-"*70)
        print("OPTIMIZATION RESULTS")
        print("-"*70)
        print(f"Best parameters: {best_result}")
        print(f"Best Sharpe ratio: {best_sharpe:.3f}")
        
        return {
            'best_params': best_result,
            'best_sharpe': best_sharpe,
            'all_results': results
        }
    
    def _calculate_advanced_metrics(self, backtest_result: Dict) -> Dict:
        """Calculate advanced performance metrics."""
        trade_journal_df = backtest_result.get('trade_journal', pd.DataFrame())
        trades = trade_journal_df.to_dict('records') if not trade_journal_df.empty else []
        equity_curve = backtest_result.get('equity_curve', [])
        
        if not trades:
            return self._empty_metrics()
        
        # Win/loss statistics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit/Loss metrics
        total_profit = sum([t.get('pnl', 0) for t in winning_trades])
        total_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        avg_win = (total_profit / num_wins) if num_wins > 0 else 0
        avg_loss = (total_loss / num_losses) if num_losses > 0 else 0
        
        # Expectancy
        expectancy = ((win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss))
        
        # Drawdown calculation
        max_drawdown_pct = self._calculate_max_drawdown(equity_curve)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'consecutive_wins': max_consecutive_wins,
            'consecutive_losses': max_consecutive_losses
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float], periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        returns = pd.Series([equity_curve[i] / equity_curve[i-1] - 1 
                           for i in range(1, len(equity_curve))])
        
        if returns.std() == 0:
            return 0.0
        
        return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    
    def _calculate_consecutive_stats(self, trades: List[Dict]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _display_detailed_results(self, result: Dict, metrics: Dict):
        """Display detailed backtest results."""
        print("\n" + "-"*60)
        print("BACKTEST RESULTS")
        print("-"*60)
        
        # Performance metrics
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Expectancy: ${metrics['expectancy']:.2f}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Consecutive Wins: {metrics['consecutive_wins']}")
        print(f"Consecutive Losses: {metrics['consecutive_losses']}")
        
        # Returns
        final_balance = result.get('final_balance', self.initial_balance)
        total_pnl = final_balance - self.initial_balance
        return_pct = (total_pnl / self.initial_balance) * 100
        
        print(f"\nReturns:")
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"Return: {return_pct:+.2f}%")
    
    def _generate_comparison_report(self, results: Dict):
        """Generate comparison report for multi-period results."""
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80)
        
        # Summary table
        print(f"\n{'Period':<20} {'Trades':<8} {'Win%':<8} {'PF':<8} {'DD%':<8} {'Sharpe':<8} {'Return%':<10}")
        print("-"*80)
        
        for period_name, result in results.items():
            if result.get('status') == 'failed':
                print(f"{period_name:<20} {'ERROR':<8}")
                continue
            
            metrics = result.get('advanced_metrics', {})
            final_balance = result.get('final_balance', self.initial_balance)
            return_pct = ((final_balance - self.initial_balance) / self.initial_balance) * 100
            
            print(f"{period_name:<20} "
                  f"{metrics.get('total_trades', 0):<8} "
                  f"{metrics.get('win_rate', 0):<8.2f} "
                  f"{metrics.get('profit_factor', 0):<8.2f} "
                  f"{metrics.get('max_drawdown_pct', 0):<8.2f} "
                  f"{metrics.get('sharpe_ratio', 0):<8.2f} "
                  f"{return_pct:<10.2f}")
        
        # Overall statistics
        successful_results = [r for r in results.values() if r.get('status') != 'failed']
        
        if successful_results:
            avg_win_rate = np.mean([r.get('advanced_metrics', {}).get('win_rate', 0) 
                                   for r in successful_results])
            avg_sharpe = np.mean([r.get('advanced_metrics', {}).get('sharpe_ratio', 0) 
                                for r in successful_results])
            
            print(f"\n{'='*80}")
            print("OVERALL STATISTICS")
            print(f"{'='*80}")
            print(f"Average Win Rate: {avg_win_rate:.2f}%")
            print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    
    def _analyze_walk_forward_results(self, results: List[Dict]):
        """Analyze walk-forward results."""
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("="*80)
        
        # Extract performance metrics
        train_returns = []
        test_returns = []
        
        for result in results:
            train_balance = result['train_result'].get('final_balance', self.initial_balance)
            train_return = ((train_balance - self.initial_balance) / self.initial_balance) * 100
            train_returns.append(train_return)
            
            test_balance = result['test_result'].get('final_balance', self.initial_balance)
            test_return = ((test_balance - self.initial_balance) / self.initial_balance) * 100
            test_returns.append(test_return)
        
        # Calculate statistics
        avg_train_return = np.mean(train_returns)
        avg_test_return = np.mean(test_returns)
        
        print(f"Average In-Sample Return: {avg_train_return:+.2f}%")
        print(f"Average Out-of-Sample Return: {avg_test_return:+.2f}%")
        
        # Check for overfitting
        degradation = ((avg_train_return - avg_test_return) / avg_train_return) * 100 if avg_train_return > 0 else 0
        
        if degradation > 50:
            print(f"\n⚠️  WARNING: Significant performance degradation ({degradation:.1f}%)")
            print("   This suggests potential overfitting")
        elif degradation > 20:
            print(f"\n⚠️  CAUTION: Moderate performance degradation ({degradation:.1f}%)")
        else:
            print(f"\n✅ Good consistency between in-sample and out-of-sample results")


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(description='Unified Backtesting Interface')
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Backtest mode')
    
    # Quick validation
    quick_parser = subparsers.add_parser('quick', help='Quick validation backtest')
    quick_parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    quick_parser.add_argument('--start', default='2023-01-01', help='Start date')
    quick_parser.add_argument('--end', default='2023-03-31', help='End date')
    quick_parser.add_argument('--expected-trades', type=int, default=100, 
                            help='Expected minimum trades')
    
    # Single period
    single_parser = subparsers.add_parser('single', help='Single period backtest')
    single_parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    single_parser.add_argument('--start', required=True, help='Start date')
    single_parser.add_argument('--end', required=True, help='End date')
    single_parser.add_argument('--timeframe', default='1h', help='Timeframe')
    
    # Multi period
    multi_parser = subparsers.add_parser('multi', help='Multi-period backtest')
    multi_parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    
    # Walk forward
    walk_parser = subparsers.add_parser('walk-forward', help='Walk-forward analysis')
    walk_parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    walk_parser.add_argument('--start', required=True, help='Start date')
    walk_parser.add_argument('--end', required=True, help='End date')
    walk_parser.add_argument('--train-months', type=int, default=6, 
                           help='Training period months')
    walk_parser.add_argument('--test-months', type=int, default=1, 
                           help='Testing period months')
    
    # Optimization
    opt_parser = subparsers.add_parser('optimize', help='Parameter optimization')
    opt_parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    opt_parser.add_argument('--start', required=True, help='Start date')
    opt_parser.add_argument('--end', required=True, help='End date')
    
    # Common arguments
    parser.add_argument('--balance', type=float, default=10000, 
                       help='Initial balance')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = UnifiedBacktester(initial_balance=args.balance)
    
    # Execute command
    if args.command == 'quick':
        results = backtester.quick_validation(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            expected_trades=args.expected_trades
        )
    
    elif args.command == 'single':
        results = backtester.single_period_backtest(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            timeframe=args.timeframe
        )
    
    elif args.command == 'multi':
        results = backtester.multi_period_backtest(pair=args.pair)
    
    elif args.command == 'walk-forward':
        results = backtester.walk_forward_analysis(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end,
            train_months=args.train_months,
            test_months=args.test_months
        )
    
    elif args.command == 'optimize':
        results = backtester.parameter_optimization(
            pair=args.pair,
            start_date=args.start,
            end_date=args.end
        )
    
    else:
        parser.print_help()
        return
    
    # Save results if requested
    if args.save_results and results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{args.command}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            # Convert any non-serializable objects
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {filename}")
    
    print("\n✓ Backtest complete!")


if __name__ == '__main__':
    main()
