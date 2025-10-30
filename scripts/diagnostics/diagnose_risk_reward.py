"""
Diagnostic Script: Analyze Risk/Reward Ratios and Entry Quality
Investigates why average loss > average win despite low win rate.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

def analyze_trade_results(results_file: str):
    """Analyze trade results from backtest JSON."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("RISK/REWARD DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    # Extract all trades
    all_trades = []
    
    # Collect trades from all periods
    for period_name, period_data in data.get('periods', {}).items():
        trades = period_data.get('trades', [])
        for trade in trades:
            trade['period'] = period_name
            all_trades.append(trade)
    
    if not all_trades:
        print("❌ No trades found in results!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    
    print(f"\n📊 Total Trades Analyzed: {len(df)}")
    print(f"   Winning Trades: {len(df[df['pnl'] > 0])}")
    print(f"   Losing Trades: {len(df[df['pnl'] <= 0])}")
    
    # Calculate metrics
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    if len(wins) > 0 and len(losses) > 0:
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())
        
        print(f"\n💰 P&L Analysis:")
        print(f"   Average Win:  ${avg_win:.2f}")
        print(f"   Average Loss: ${avg_loss:.2f}")
        print(f"   Win/Loss Ratio: {avg_win/avg_loss:.2f}x")
        
        if avg_loss > avg_win:
            print(f"   ⚠️  PROBLEM: Average loss is {avg_loss/avg_win:.2f}x larger than average win!")
    
    # Analyze R:R ratios from trade metadata
    print(f"\n📐 Risk/Reward Ratio Analysis:")
    
    if 'risk_reward' in df.columns:
        print(f"   Mean R:R Ratio: {df['risk_reward'].mean():.2f}")
        print(f"   Median R:R Ratio: {df['risk_reward'].median():.2f}")
        print(f"   Min R:R Ratio: {df['risk_reward'].min():.2f}")
        print(f"   Max R:R Ratio: {df['risk_reward'].max():.2f}")
        
        # Distribution
        print(f"\n   R:R Distribution:")
        print(f"   < 1.5:  {len(df[df['risk_reward'] < 1.5])} trades ({len(df[df['risk_reward'] < 1.5])/len(df)*100:.1f}%)")
        print(f"   1.5-2:  {len(df[(df['risk_reward'] >= 1.5) & (df['risk_reward'] < 2.0)])} trades")
        print(f"   2-3:    {len(df[(df['risk_reward'] >= 2.0) & (df['risk_reward'] < 3.0)])} trades")
        print(f"   >= 3:   {len(df[df['risk_reward'] >= 3.0])} trades ({len(df[df['risk_reward'] >= 3.0])/len(df)*100:.1f}%)")
    
    # Analyze actual risk and reward
    if 'entry_price' in df.columns and 'stop_loss' in df.columns and 'take_profit' in df.columns:
        print(f"\n🎯 Stop Loss & Take Profit Analysis:")
        
        # Calculate actual risk and reward
        df['actual_risk_pct'] = abs((df['entry_price'] - df['stop_loss']) / df['entry_price']) * 100
        df['actual_reward_pct'] = abs((df['take_profit'] - df['entry_price']) / df['entry_price']) * 100
        
        print(f"   Mean Risk (SL distance): {df['actual_risk_pct'].mean():.2f}%")
        print(f"   Mean Reward (TP distance): {df['actual_reward_pct'].mean():.2f}%")
        
        # Check if risk is too wide
        if df['actual_risk_pct'].mean() > 2.0:
            print(f"   ⚠️  PROBLEM: Average risk ({df['actual_risk_pct'].mean():.2f}%) is TOO WIDE!")
            print(f"      Recommendation: Tighten stop losses to 1.0-1.5% risk")
        
        # Check if reward is too tight
        if df['actual_reward_pct'].mean() < 3.0:
            print(f"   ⚠️  PROBLEM: Average reward ({df['actual_reward_pct'].mean():.2f}%) is TOO TIGHT!")
            print(f"      Recommendation: Extend take profits to 3.0-5.0% reward")
        
        # Calculate actual R:R from structure
        df['calculated_rr'] = df['actual_reward_pct'] / df['actual_risk_pct']
        print(f"\n   Calculated R:R (from actual SL/TP): {df['calculated_rr'].mean():.2f}")
        
        if df['calculated_rr'].mean() < 2.0:
            print(f"   ❌ CRITICAL: Calculated R:R is BELOW 2:1!")
    
    # Analyze by entry type
    if 'entry_type' in df.columns:
        print(f"\n📝 Analysis by Entry Type:")
        for entry_type in df['entry_type'].unique():
            subset = df[df['entry_type'] == entry_type]
            win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            avg_pnl = subset['pnl'].mean()
            
            print(f"\n   {entry_type}:")
            print(f"      Trades: {len(subset)}")
            print(f"      Win Rate: {win_rate:.1f}%")
            print(f"      Avg P&L: ${avg_pnl:.2f}")
            
            if 'risk_reward' in subset.columns:
                print(f"      Avg R:R: {subset['risk_reward'].mean():.2f}")
            
            if 'actual_risk_pct' in subset.columns:
                print(f"      Avg Risk: {subset['actual_risk_pct'].mean():.2f}%")
                print(f"      Avg Reward: {subset['actual_reward_pct'].mean():.2f}%")
    
    # Analyze by session
    if 'session' in df.columns:
        print(f"\n🌍 Analysis by Trading Session:")
        for session in df['session'].unique():
            subset = df[df['session'] == session]
            win_rate = len(subset[subset['pnl'] > 0]) / len(subset) * 100
            
            wins_session = subset[subset['pnl'] > 0]
            losses_session = subset[subset['pnl'] <= 0]
            
            if len(wins_session) > 0 and len(losses_session) > 0:
                avg_win_session = wins_session['pnl'].mean()
                avg_loss_session = abs(losses_session['pnl'].mean())
                
                print(f"\n   {session}:")
                print(f"      Trades: {len(subset)}")
                print(f"      Win Rate: {win_rate:.1f}%")
                print(f"      Avg Win: ${avg_win_session:.2f}")
                print(f"      Avg Loss: ${avg_loss_session:.2f}")
                print(f"      Win/Loss Ratio: {avg_win_session/avg_loss_session:.2f}x")
                
                if avg_loss_session > avg_win_session:
                    print(f"      ⚠️  PROBLEM: Losses are {avg_loss_session/avg_win_session:.2f}x larger in {session}!")
    
    # Find worst offenders (trades with bad R:R)
    print(f"\n🔍 Worst R:R Trades (Bottom 10):")
    if 'risk_reward' in df.columns:
        worst_rr = df.nsmallest(10, 'risk_reward')[['entry_type', 'risk_reward', 'pnl', 'period']]
        print(worst_rr.to_string(index=False))
    
    # Confluence score analysis
    if 'confluence_score' in df.columns:
        print(f"\n⭐ Confluence Score Analysis:")
        print(f"   Mean: {df['confluence_score'].mean():.2f}")
        print(f"   Median: {df['confluence_score'].median():.2f}")
        
        # Compare win rate by confluence score
        high_conf = df[df['confluence_score'] >= df['confluence_score'].median()]
        low_conf = df[df['confluence_score'] < df['confluence_score'].median()]
        
        high_wr = len(high_conf[high_conf['pnl'] > 0]) / len(high_conf) * 100 if len(high_conf) > 0 else 0
        low_wr = len(low_conf[low_conf['pnl'] > 0]) / len(low_conf) * 100 if len(low_conf) > 0 else 0
        
        print(f"\n   High Confluence (>={df['confluence_score'].median():.1f}):")
        print(f"      Trades: {len(high_conf)}")
        print(f"      Win Rate: {high_wr:.1f}%")
        
        print(f"\n   Low Confluence (<{df['confluence_score'].median():.1f}):")
        print(f"      Trades: {len(low_conf)}")
        print(f"      Win Rate: {low_wr:.1f}%")
        
        if low_wr > high_wr:
            print(f"\n   ⚠️  PROBLEM: Low confluence trades perform BETTER!")
            print(f"      This suggests confluence scoring is INVERTED or WRONG!")
    
    print(f"\n{'='*80}")
    print("DIAGNOSTIC RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check each issue
    if len(wins) > 0 and len(losses) > 0:
        avg_win = wins['pnl'].mean()
        avg_loss = abs(losses['pnl'].mean())
        if avg_loss > avg_win:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'Average loss > average win',
                'fix': 'Tighten stop losses OR extend take profits'
            })
    
    if 'actual_risk_pct' in df.columns:
        if df['actual_risk_pct'].mean() > 2.0:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'Stop losses too wide ({df["actual_risk_pct"].mean():.2f}%)',
                'fix': 'Reduce SL distance in _calculate_structure_based_stop_loss'
            })
    
    if 'actual_reward_pct' in df.columns:
        if df['actual_reward_pct'].mean() < 3.0:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'Take profits too tight ({df["actual_reward_pct"].mean():.2f}%)',
                'fix': 'Extend TP distance in _calculate_structure_based_tps'
            })
    
    if 'calculated_rr' in df.columns:
        if df['calculated_rr'].mean() < 2.0:
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': f'Calculated R:R too low ({df["calculated_rr"].mean():.2f})',
                'fix': 'Enforce minimum 2:1 or 3:1 R:R ratio before signal generation'
            })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
            print(f"   → {rec['fix']}")
    else:
        print("\n✅ No critical issues found in risk/reward structure.")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    # Find most recent backtest results
    results_dir = Path('/Users/danielfleshler/Desktop/Code/Crypto_bot_trader/results/backtests')
    
    # Check root directory first
    root_results = list(Path('/Users/danielfleshler/Desktop/Code/Crypto_bot_trader').glob('realistic_backtest_results_*.json'))
    
    if root_results:
        # Use most recent from root
        latest_result = max(root_results, key=lambda p: p.stat().st_mtime)
        print(f"📂 Analyzing: {latest_result.name}\n")
        analyze_trade_results(str(latest_result))
    elif results_dir.exists():
        # Use most recent from results directory
        results = list(results_dir.glob('realistic_backtest_results_*.json'))
        if results:
            latest_result = max(results, key=lambda p: p.stat().st_mtime)
            print(f"📂 Analyzing: {latest_result.name}\n")
            analyze_trade_results(str(latest_result))
        else:
            print("❌ No backtest results found!")
    else:
        print("❌ Results directory not found!")

