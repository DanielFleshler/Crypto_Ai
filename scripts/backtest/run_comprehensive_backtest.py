"""
Comprehensive Backtest Analysis
Tests trading strategy across multiple market regimes with detailed performance metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from backtester import BacktestEngine
from trading_strategy.config_loader import ConfigLoader

class ComprehensiveBacktester:
    """
    Runs comprehensive backtests across market regimes and analyzes performance.
    """

    def __init__(self, base_path: str, initial_balance: float = 10000):
        self.base_path = Path(base_path)
        self.initial_balance = initial_balance
        self.results = {}

    def load_parquet_data(self, pair: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load and combine parquet files for the specified period.

        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '15m')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Combined DataFrame with OHLCV data
        """
        data_path = self.base_path / 'data' / 'raw' / pair / timeframe

        if not data_path.exists():
            raise ValueError(f"Data path not found: {data_path}")

        # Get all parquet files in the directory
        parquet_files = sorted(list(data_path.glob('*.parquet')))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")

        # Load and combine all files
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=False)

        # Ensure datetime index
        if not isinstance(combined_df.index, pd.DatetimeIndex):
            # Try common timestamp column names
            time_columns = ['timestamp', 'open_time', 'start_time', 'time', 'datetime']
            time_col = None

            for col in time_columns:
                if col in combined_df.columns:
                    time_col = col
                    break

            if time_col:
                combined_df[time_col] = pd.to_datetime(combined_df[time_col])
                combined_df.set_index(time_col, inplace=True)
            else:
                # Try to convert index to datetime
                combined_df.index = pd.to_datetime(combined_df.index)

        # Sort by timestamp
        combined_df.sort_index(inplace=True)

        # Convert index to timezone-naive if it has timezone info
        if combined_df.index.tz is not None:
            combined_df.index = combined_df.index.tz_localize(None)

        # Filter by date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        combined_df = combined_df.loc[start:end]

        # Standardize column names
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        combined_df.rename(columns=column_mapping, inplace=True)

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in combined_df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        return combined_df[required_cols]

    def identify_market_regime(self, df: pd.DataFrame) -> str:
        """
        Identify market regime based on price action and volatility.

        Args:
            df: OHLCV DataFrame

        Returns:
            Market regime: 'bull', 'bear', 'sideways', 'high_vol', 'low_vol'
        """
        # Calculate returns
        returns = df['close'].pct_change()

        # Calculate trend (linear regression slope)
        x = np.arange(len(df))
        y = df['close'].values
        trend = np.polyfit(x, y, 1)[0]

        # Calculate volatility (annualized standard deviation)
        volatility = returns.std() * np.sqrt(365 * 24)  # For hourly data

        # Calculate price change percentage
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

        # Classify regime
        if abs(price_change) < 10:
            regime = 'sideways'
        elif price_change > 10:
            regime = 'bull'
        else:
            regime = 'bear'

        # Add volatility classification
        if volatility > 1.5:
            regime += '_high_vol'
        elif volatility < 0.5:
            regime += '_low_vol'

        return regime

    def calculate_advanced_metrics(self, backtest_result: Dict) -> Dict:
        """
        Calculate advanced performance metrics.

        Args:
            backtest_result: Raw backtest results

        Returns:
            Dictionary with advanced metrics
        """
        # CRITICAL FIX: Backtester returns 'trade_journal' as DataFrame, not 'trades' as list
        trade_journal_df = backtest_result.get('trade_journal', pd.DataFrame())
        trades = trade_journal_df.to_dict('records') if not trade_journal_df.empty else []
        equity_curve = backtest_result.get('equity_curve', [])

        if not trades:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'expectancy': 0.0,
                'sharpe_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'recovery_factor': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        # Basic stats
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

        # Profit/Loss stats
        total_profit = sum([t.get('pnl', 0) for t in winning_trades])
        total_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))

        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0

        avg_win = (total_profit / num_wins) if num_wins > 0 else 0
        avg_loss = (total_loss / num_losses) if num_losses > 0 else 0

        expectancy = ((win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss))

        # Drawdown
        if equity_curve:
            # CRITICAL FIX: equity_curve contains floats (balances), not dicts
            equity_values = equity_curve  # Already a list of balance values
            peak = equity_values[0]
            max_dd = 0

            for value in equity_values:
                if value > peak:
                    peak = value
                dd = ((peak - value) / peak) * 100
                if dd > max_dd:
                    max_dd = dd

            max_drawdown_pct = max_dd
        else:
            max_drawdown_pct = 0

        # Sharpe Ratio
        if equity_curve and len(equity_curve) > 1:
            # FIXED: equity_curve is a list of floats, not dicts
            returns = pd.Series([equity_curve[i] / equity_curve[i-1] - 1
                                for i in range(1, len(equity_curve))])
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Consecutive wins/losses
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

        # Largest win/loss
        largest_win = max([t.get('pnl', 0) for t in trades]) if trades else 0
        largest_loss = min([t.get('pnl', 0) for t in trades]) if trades else 0

        # Recovery factor
        # Calculate net profit from final_balance and initial_balance
        final_balance = backtest_result.get('final_balance', 10000)
        initial_balance = 10000  # Assume default initial balance
        net_profit = final_balance - initial_balance
        recovery_factor = (net_profit / max_drawdown_pct) if max_drawdown_pct > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'consecutive_wins': max_consecutive_wins,
            'consecutive_losses': max_consecutive_losses,
            'recovery_factor': recovery_factor,
            'total_trades': total_trades,
            'winning_trades': num_wins,
            'losing_trades': num_losses
        }

    def analyze_by_session(self, trades: List[Dict]) -> Dict:
        """
        Analyze performance by trading session.

        Args:
            trades: List of trade dictionaries

        Returns:
            Performance metrics by session
        """
        session_stats = {
            'asia': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'london': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ny': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'london_ny': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'off_hours': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }

        for trade in trades:
            # Get entry timestamp (FIXED: use 'timestamp' not 'entry_time')
            entry_time = trade.get('timestamp') or trade.get('entry_time')
            if not entry_time:
                continue

            # Convert to datetime if it's a string
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)

            # Determine session based on hour (UTC)
            hour = entry_time.hour if hasattr(entry_time, 'hour') else 0

            if 0 <= hour < 8:
                session = 'asia'
            elif 8 <= hour < 13:
                session = 'london'
            elif 13 <= hour < 16:
                session = 'london_ny'
            elif 16 <= hour < 21:
                session = 'ny'
            else:
                session = 'off_hours'

            session_stats[session]['trades'] += 1
            if trade.get('pnl', 0) > 0:
                session_stats[session]['wins'] += 1
            session_stats[session]['total_pnl'] += trade.get('pnl', 0)

        # Calculate win rates
        for session in session_stats:
            total = session_stats[session]['trades']
            if total > 0:
                session_stats[session]['win_rate'] = (session_stats[session]['wins'] / total) * 100
            else:
                session_stats[session]['win_rate'] = 0

        return session_stats

    def run_backtest_for_period(self, pair: str, timeframe: str,
                                start_date: str, end_date: str,
                                period_name: str) -> Dict:
        """
        Run backtest for a specific period.

        Args:
            pair: Trading pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            period_name: Name for this period

        Returns:
            Backtest results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running backtest: {period_name}")
        print(f"Pair: {pair}, Timeframe: {timeframe}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*60}")

        # Load data
        df = self.load_parquet_data(pair, timeframe, start_date, end_date)
        print(f"Loaded {len(df)} candles")

        # Identify market regime
        regime = self.identify_market_regime(df)
        print(f"Market regime: {regime}")

        # Run backtest using the existing data structure
        try:
            engine = BacktestEngine(
                base_path=str(self.base_path),
                initial_balance=self.initial_balance,
                risk_per_trade=0.02
            )

            result = engine.run_backtest(pair, start_date, end_date)

            # Calculate advanced metrics
            metrics = self.calculate_advanced_metrics(result)

            # Analyze by session (FIXED: use trade_journal DataFrame)
            trade_journal_df = result.get('trade_journal', pd.DataFrame())
            trades_list = trade_journal_df.to_dict('records') if not trade_journal_df.empty else []
            session_stats = self.analyze_by_session(trades_list)

            # Compile results
            comprehensive_result = {
                'period_name': period_name,
                'pair': pair,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'market_regime': regime,
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'expectancy': metrics['expectancy'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'largest_win': metrics['largest_win'],
                'largest_loss': metrics['largest_loss'],
                'consecutive_wins': metrics['consecutive_wins'],
                'consecutive_losses': metrics['consecutive_losses'],
                'recovery_factor': metrics['recovery_factor'],
                'total_pnl': result.get('final_balance', self.initial_balance) - self.initial_balance,
                'final_balance': result.get('final_balance', self.initial_balance),
                'return_pct': ((result.get('final_balance', self.initial_balance) - self.initial_balance) / self.initial_balance) * 100,
                'session_performance': session_stats,
                'raw_result': result
            }

            print(f"\n{'-'*60}")
            print(f"RESULTS:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.2f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Total P&L: ${result.get('total_pnl', 0):.2f}")
            print(f"  Return: {comprehensive_result['return_pct']:.2f}%")
            print(f"{'-'*60}")

            return comprehensive_result

        except Exception as e:
            print(f"ERROR during backtest: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'period_name': period_name,
                'error': str(e),
                'status': 'failed'
            }

    def generate_report(self, results: Dict) -> str:
        """
        Generate comprehensive analysis report.

        Args:
            results: Dictionary of backtest results

        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*80)
        report.append("COMPREHENSIVE BACKTEST ANALYSIS REPORT")
        report.append("="*80)

        # Summary table
        report.append("\n" + "-"*80)
        report.append("SUMMARY BY PERIOD")
        report.append("-"*80)
        report.append(f"{'Period':<20} {'Regime':<15} {'Trades':<8} {'Win%':<8} {'PF':<8} {'DD%':<8} {'Return%':<10}")
        report.append("-"*80)

        for period_name, result in results.items():
            if result.get('status') == 'failed':
                report.append(f"{period_name:<20} {'ERROR':<15} {'-':<8} {'-':<8} {'-':<8} {'-':<8} {'-':<10}")
                continue

            report.append(
                f"{period_name:<20} "
                f"{result.get('market_regime', 'N/A'):<15} "
                f"{result.get('total_trades', 0):<8} "
                f"{result.get('win_rate', 0):<8.2f} "
                f"{result.get('profit_factor', 0):<8.2f} "
                f"{result.get('max_drawdown_pct', 0):<8.2f} "
                f"{result.get('return_pct', 0):<10.2f}"
            )

        report.append("-"*80)

        # Detailed analysis for each period
        for period_name, result in results.items():
            if result.get('status') == 'failed':
                continue

            report.append(f"\n{'='*80}")
            report.append(f"{period_name.upper()}")
            report.append(f"{'='*80}")
            report.append(f"Market Regime: {result.get('market_regime')}")
            report.append(f"Period: {result.get('start_date')} to {result.get('end_date')}")

            report.append(f"\n{'-'*40}")
            report.append("Performance Metrics:")
            report.append(f"{'-'*40}")
            report.append(f"  Total Trades: {result.get('total_trades')}")
            report.append(f"  Winning Trades: {result.get('winning_trades')}")
            report.append(f"  Losing Trades: {result.get('losing_trades')}")
            report.append(f"  Win Rate: {result.get('win_rate', 0):.2f}%")
            report.append(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
            report.append(f"  Expectancy: ${result.get('expectancy', 0):.2f}")
            report.append(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")

            report.append(f"\n{'-'*40}")
            report.append("Risk Metrics:")
            report.append(f"{'-'*40}")
            report.append(f"  Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%")
            report.append(f"  Recovery Factor: {result.get('recovery_factor', 0):.2f}")
            report.append(f"  Consecutive Wins: {result.get('consecutive_wins', 0)}")
            report.append(f"  Consecutive Losses: {result.get('consecutive_losses', 0)}")

            report.append(f"\n{'-'*40}")
            report.append("Trade Statistics:")
            report.append(f"{'-'*40}")
            report.append(f"  Average Win: ${result.get('avg_win', 0):.2f}")
            report.append(f"  Average Loss: ${result.get('avg_loss', 0):.2f}")
            report.append(f"  Largest Win: ${result.get('largest_win', 0):.2f}")
            report.append(f"  Largest Loss: ${result.get('largest_loss', 0):.2f}")

            report.append(f"\n{'-'*40}")
            report.append("Returns:")
            report.append(f"{'-'*40}")
            report.append(f"  Total P&L: ${result.get('total_pnl', 0):.2f}")
            report.append(f"  Final Balance: ${result.get('final_balance', 0):.2f}")
            report.append(f"  Return: {result.get('return_pct', 0):.2f}%")

            # Session analysis
            session_perf = result.get('session_performance', {})
            if session_perf:
                report.append(f"\n{'-'*40}")
                report.append("Performance by Session:")
                report.append(f"{'-'*40}")
                report.append(f"{'Session':<15} {'Trades':<10} {'Win Rate':<12} {'Total P&L':<12}")
                report.append(f"{'-'*40}")

                for session, stats in session_perf.items():
                    if stats['trades'] > 0:
                        report.append(
                            f"{session.upper():<15} "
                            f"{stats['trades']:<10} "
                            f"{stats['win_rate']:<12.2f} "
                            f"${stats['total_pnl']:<11.2f}"
                        )

        # Overall assessment
        report.append(f"\n{'='*80}")
        report.append("OVERALL ASSESSMENT")
        report.append(f"{'='*80}")

        # Calculate aggregate metrics
        successful_results = [r for r in results.values() if r.get('status') != 'failed']

        if not successful_results:
            report.append("\nNo successful backtests to analyze.")
            return '\n'.join(report)

        total_trades = sum([r.get('total_trades', 0) for r in successful_results])
        avg_win_rate = np.mean([r.get('win_rate', 0) for r in successful_results])
        avg_profit_factor = np.mean([r.get('profit_factor', 0) for r in successful_results])
        avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in successful_results])
        max_dd = max([r.get('max_drawdown_pct', 0) for r in successful_results])

        report.append(f"\nAggregate Statistics:")
        report.append(f"  Total Trades Across All Periods: {total_trades}")
        report.append(f"  Average Win Rate: {avg_win_rate:.2f}%")
        report.append(f"  Average Profit Factor: {avg_profit_factor:.2f}")
        report.append(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
        report.append(f"  Maximum Drawdown (across periods): {max_dd:.2f}%")

        # Quality assessment
        report.append(f"\nStrategy Quality Assessment:")

        quality_scores = []

        if avg_win_rate >= 50:
            report.append(f"  ✓ Win Rate: GOOD ({avg_win_rate:.2f}% >= 50%)")
            quality_scores.append(1)
        else:
            report.append(f"  ✗ Win Rate: NEEDS IMPROVEMENT ({avg_win_rate:.2f}% < 50%)")
            quality_scores.append(0)

        if avg_profit_factor >= 2.0:
            report.append(f"  ✓ Profit Factor: EXCELLENT ({avg_profit_factor:.2f} >= 2.0)")
            quality_scores.append(2)
        elif avg_profit_factor >= 1.5:
            report.append(f"  ✓ Profit Factor: GOOD ({avg_profit_factor:.2f} >= 1.5)")
            quality_scores.append(1)
        else:
            report.append(f"  ✗ Profit Factor: NEEDS IMPROVEMENT ({avg_profit_factor:.2f} < 1.5)")
            quality_scores.append(0)

        if max_dd <= 15:
            report.append(f"  ✓ Max Drawdown: GOOD ({max_dd:.2f}% <= 15%)")
            quality_scores.append(1)
        else:
            report.append(f"  ⚠ Max Drawdown: ACCEPTABLE ({max_dd:.2f}% > 15%)")
            quality_scores.append(0.5)

        if avg_sharpe >= 1.0:
            report.append(f"  ✓ Sharpe Ratio: GOOD ({avg_sharpe:.2f} >= 1.0)")
            quality_scores.append(1)
        else:
            report.append(f"  ✗ Sharpe Ratio: NEEDS IMPROVEMENT ({avg_sharpe:.2f} < 1.0)")
            quality_scores.append(0)

        overall_quality = np.mean(quality_scores) * 100
        report.append(f"\n  Overall Quality Score: {overall_quality:.1f}/100")

        if overall_quality >= 75:
            report.append(f"  Assessment: EXCELLENT - Strategy ready for live trading")
        elif overall_quality >= 50:
            report.append(f"  Assessment: GOOD - Consider optimization before live trading")
        else:
            report.append(f"  Assessment: NEEDS WORK - Significant improvements required")

        report.append(f"\n{'='*80}\n")

        return '\n'.join(report)


def main():
    """
    Run comprehensive backtest analysis.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST ANALYSIS")
    print("Testing strategy across multiple market regimes")
    print("="*80)

    base_path = Path(__file__).parent
    backtester = ComprehensiveBacktester(base_path=str(base_path), initial_balance=10000)

    # Define test periods for different market regimes
    # Note: Data starts from July 2021
    test_periods = {
        # Bull market 2021 (Jul-Nov)
        'Bull_2021': {
            'pair': 'BTCUSDT',
            'timeframe': '1h',
            'start': '2021-07-01',
            'end': '2021-11-30'
        },

        # Bear market 2022
        'Bear_2022': {
            'pair': 'BTCUSDT',
            'timeframe': '1h',
            'start': '2022-01-01',
            'end': '2022-12-31'
        },

        # Sideways/Recovery 2023
        'Sideways_2023': {
            'pair': 'BTCUSDT',
            'timeframe': '1h',
            'start': '2023-01-01',
            'end': '2023-12-31'
        },

        # Bull market 2024 (Out-of-sample)
        'Bull_2024_OOS': {
            'pair': 'BTCUSDT',
            'timeframe': '1h',
            'start': '2024-01-01',
            'end': '2024-10-18'
        }
    }

    results = {}

    # Run backtests for each period
    for period_name, config in test_periods.items():
        try:
            result = backtester.run_backtest_for_period(
                pair=config['pair'],
                timeframe=config['timeframe'],
                start_date=config['start'],
                end_date=config['end'],
                period_name=period_name
            )
            results[period_name] = result
        except Exception as e:
            print(f"\nERROR in {period_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[period_name] = {'status': 'failed', 'error': str(e)}

    # Generate comprehensive report
    report = backtester.generate_report(results)
    print(report)

    # Save report to file
    report_file = base_path / 'backtest_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save detailed results to JSON
    results_file = base_path / 'backtest_results.json'

    # Convert datetime objects to strings for JSON serialization
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetime(item) for item in obj]
        return obj

    json_results = convert_datetime(results)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {results_file}")


if __name__ == '__main__':
    main()
