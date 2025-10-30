#!/usr/bin/env python3
"""
Strategy Diagnostic Tool
Identifies why signals are being filtered out
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.config_loader import ConfigLoader
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.elliott_wave import ElliottWaveDetector


class StrategyDiagnostic:
    """Diagnose signal generation issues"""
    
    def __init__(self):
        self.data_loader = DataLoader(base_path='data/raw')
        self.config_loader = ConfigLoader()
        
    def diagnose_signal_pipeline(self, pair: str = 'BTCUSDT',
                               start_date: str = '2023-01-01',
                               end_date: str = '2023-03-31'):
        """Diagnose each step of signal generation"""
        print("="*80)
        print("SIGNAL GENERATION PIPELINE DIAGNOSTIC")
        print(f"Period: {start_date} to {end_date}")
        print("="*80)
        
        # Load data
        print("\n1. DATA LOADING")
        print("-" * 40)
        data = self.data_loader.load_pair_data(pair, ['4h', '1h', '15m'], start_date, end_date)
        
        for tf, df in data.items():
            print(f"{tf}: {len(df)} candles")
            if len(df) > 0:
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Check HTF bias
        print("\n2. HTF BIAS DETECTION")
        print("-" * 40)
        
        strategy = TradingStrategy(
            base_path='.',
            config_loader=self.config_loader
        )
        
        # Manually check HTF structures
        ms_detector = MarketStructureDetector(data['4h'], self.config_loader)
        structures_df = ms_detector.detect_market_structure_breaks(data['4h'])
        
        if 'structure_type' in structures_df.columns:
            structure_counts = structures_df['structure_type'].value_counts()
            print(f"Market structures found:")
            for struct_type, count in structure_counts.items():
                print(f"  {struct_type}: {count}")
        else:
            print("No market structures detected")
        
        # Check ICT concepts
        print("\n3. ICT CONCEPT DETECTION")
        print("-" * 40)
        
        ict_detector = ICTConceptsDetector(data['1h'], self.config_loader)
        
        # Count concepts
        fvgs = ict_detector.detect_fvg(data['1h'])
        print(f"FVGs detected: {len(fvgs)}")
        if fvgs:
            bullish_fvgs = [f for f in fvgs if f.is_bullish()]
            bearish_fvgs = [f for f in fvgs if f.is_bearish()]
            print(f"  Bullish: {len(bullish_fvgs)}")
            print(f"  Bearish: {len(bearish_fvgs)}")
        
        # Check filtering stages
        print("\n4. SIGNAL FILTERING ANALYSIS")
        print("-" * 40)
        
        # Get raw config values
        config = self.config_loader.get_entry_confirmation_config()
        print(f"Minimum confirmations required: {config.minimum_confirmations}")
        print(f"Minimum score required: {config.minimum_score}")
        
        risk_config = self.config_loader.get_risk_management_config()
        print(f"\nRisk filters:")
        print(f"  Max concurrent positions: {risk_config.max_concurrent_positions}")
        print(f"  Max daily risk: {risk_config.max_daily_risk*100:.1f}%")
        print(f"  Min RR ratio: {risk_config.minimum_rr_ratio}")
        
        # Check session filters
        session_config = self.config_loader.get_session_config()
        if session_config.filter_by_session:
            print(f"\nSession filtering: ENABLED")
            print(f"  Optimal sessions: {session_config.optimal_sessions}")
        else:
            print(f"\nSession filtering: DISABLED")
        
        # Analyze a sample period in detail
        print("\n5. DETAILED SIGNAL ANALYSIS (First Week)")
        print("-" * 40)
        
        # Take first week of data
        if len(data['1h']) > 168:  # 1 week of hourly data
            week_data = data['1h'].iloc[:168]
            
            # Count potential entry points
            potential_entries = 0
            
            # Simple heuristic: count significant moves
            for i in range(20, len(week_data)):
                # Check for pullback
                high_20 = week_data['high'].iloc[i-20:i].max()
                low_20 = week_data['low'].iloc[i-20:i].min()
                current_close = week_data['close'].iloc[i]
                
                # Bullish pullback
                if current_close < (low_20 + (high_20 - low_20) * 0.618):
                    potential_entries += 1
                
                # Bearish pullback  
                if current_close > (low_20 + (high_20 - low_20) * 0.382):
                    potential_entries += 1
            
            print(f"Potential entry zones identified: {potential_entries}")
            print(f"This suggests ~{potential_entries/7:.1f} opportunities per day")
        
        print("\n6. RECOMMENDATIONS")
        print("-" * 40)
        print("Based on the diagnostic:")
        
        recommendations = []
        
        if len(fvgs) < 50:
            recommendations.append("- FVG detection too restrictive - adjust thresholds")
        
        if config.minimum_confirmations > 2:
            recommendations.append("- Reduce minimum confirmations requirement")
            
        if config.minimum_score > 0.5:
            recommendations.append("- Lower minimum score threshold")
            
        if session_config.filter_by_session:
            recommendations.append("- Consider disabling session filtering initially")
            
        if risk_config.minimum_rr_ratio > 2.0:
            recommendations.append("- Reduce minimum RR ratio requirement")
        
        if not recommendations:
            recommendations.append("- Check data quality and market structure detection")
            recommendations.append("- Verify Elliott Wave pattern detection")
            recommendations.append("- Consider simplifying entry logic")
        
        for rec in recommendations:
            print(rec)
        
        return {
            'data_quality': len(data['1h']) > 0,
            'structures_found': len(structures_df) if 'structure_type' in structures_df.columns else 0,
            'fvgs_found': len(fvgs),
            'potential_entries': potential_entries if 'potential_entries' in locals() else 0,
            'recommendations': recommendations
        }


def main():
    """Run diagnostic"""
    diagnostic = StrategyDiagnostic()
    
    # Test different periods
    test_periods = [
        ("2023 Q1 - No trades period", "2023-01-01", "2023-03-31"),
        ("2021 Recovery - Some trades", "2021-08-01", "2021-08-31"),
        ("2022 Crash - Few trades", "2022-05-01", "2022-05-31"),
    ]
    
    for period_name, start, end in test_periods:
        print(f"\n\n{'='*80}")
        print(f"TESTING: {period_name}")
        print(f"{'='*80}")
        
        results = diagnostic.diagnose_signal_pipeline(
            pair='BTCUSDT',
            start_date=start,
            end_date=end
        )
    
    print("\n\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nKey issues to address:")
    print("1. Signal generation is too restrictive")
    print("2. Need to balance quality vs quantity of signals")
    print("3. Consider progressive filtering approach")
    print("4. May need to adjust for different market regimes")


if __name__ == '__main__':
    main()
