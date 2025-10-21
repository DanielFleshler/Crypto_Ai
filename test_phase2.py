# Test Phase 2 - Entry Types Implementation Test
# Copy and paste this code into a file named test_phase2.py and run it

"""
Phase 2 Test: Testing implementation of 5 entry types
Goal: Ensure that generate_signals() creates real signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import os
import sys

# Add the path to terminal if needed
# sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_strategy'))

from trading_strategy.data_loader import DataLoader
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.kill_zones import KillZoneDetector
from trading_strategy.trading_strategy import TradingStrategy

def create_sample_ohlc_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    Creates mock OHLC data for testing algorithms
    Includes realistic fluctuations, trends, and corrections
    """
    print(f"🔄 Creating {n_periods} candles of mock OHLC data...")
    
    # Start with base price
    start_price = 45000.0
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='15T')
    
    # Creating prices with realistic fluctuations
    np.random.seed(42)  # For reproducibility
    
    prices = [start_price]
    for i in range(1, n_periods):
        # Basic volatility
        change_pct = np.random.normal(0, 0.002)  # 0.2% average volatility
        
        # Adding trends
        if i % 200 == 0:  # Every 200 candles - new trend
            trend = np.random.choice([-1, 1]) * 0.001
        elif i % 50 == 0:  # Every 50 candles - correction
            trend = -trend * 0.5 if 'trend' in locals() else 0
        
        if 'trend' not in locals():
            trend = 0
            
        new_price = prices[-1] * (1 + change_pct + trend)
        prices.append(max(new_price, 1.0))  # Minimum price 1
    
    # Creating OHLC from prices
    ohlc_data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
            
        # Creating realistic high/low
        volatility = abs(np.random.normal(0, 0.001))
        high = max(open_price, close_price) * (1 + volatility)
        low = min(open_price, close_price) * (1 - volatility)
        
        volume = np.random.uniform(100, 1000)
        turnover = volume * close_price
        
        ohlc_data.append({
            'start_time': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume,
            'turnover': turnover
        })
    
    df = pd.DataFrame(ohlc_data)
    df.set_index('start_time', inplace=True)
    
    print(f"✅ Created {len(df)} candles")
    print(f"📊 Price range: {df['low'].min():.0f} - {df['high'].max():.0f}")
    print(f"📈 Start price: {df['close'].iloc[0]:.0f}, End price: {df['close'].iloc[-1]:.0f}")
    
    return df

def test_entry_signals():
    """Testing signal generation"""
    
    print("🚀 Starting Phase 2 test - implementing entry types")
    print("=" * 80)
    
    # Creating test data
    df_15m = create_sample_ohlc_data(1000)
    
    print("\n1️⃣ Initializing system...")
    strategy = TradingStrategy(base_path=".")
    strategy.htf_bias = 'BULLISH'  # Set bias for testing
    
    print("\n2️⃣ Running market structure detection...")
    msd = MarketStructureDetector()
    swing_df = msd.detect_swing_points(df_15m, strength=3)
    structures = msd.detect_market_structure(swing_df)
    
    swing_highs_count = swing_df['swing_high'].sum()
    swing_lows_count = swing_df['swing_low'].sum()
    print(f"   ✅ Identified {swing_highs_count} swing highs, {swing_lows_count} swing lows")
    print(f"   ✅ Identified {len(structures)} market structures")
    
    if swing_highs_count == 0 or swing_lows_count == 0:
        print("   ⚠️ No swing points - reducing strength parameter")
        swing_df = msd.detect_swing_points(df_15m, strength=2)
        structures = msd.detect_market_structure(swing_df)
        swing_highs_count = swing_df['swing_high'].sum() 
        swing_lows_count = swing_df['swing_low'].sum()
        print(f"   ✅ With strength=2: {swing_highs_count} swing highs, {swing_lows_count} swing lows")
    
    print("\n3️⃣ Running ICT detection...")
    ict = ICTConceptsDetector()
    fvgs = ict.detect_fvg(df_15m, min_gap_percent=0.05)  # Reducing threshold
    order_blocks = ict.detect_order_blocks(df_15m, swing_df)
    ote_zones = ict.detect_ote_zones(df_15m, swing_df)
    
    print(f"   ✅ Identified {len(fvgs)} FVGs")
    print(f"   ✅ Identified {len(order_blocks)} Order Blocks") 
    print(f"   ✅ Identified {len(ote_zones)} OTE Zones")
    
    print("\n4️⃣ Running Elliott Wave detection...")
    elliott = ElliottWaveDetector()
    wave1_candidates = elliott.identify_wave_1(df_15m, swing_df)
    
    elliott_sequences = []
    for wave1 in wave1_candidates[:5]:  # Check only first 5
        wave2 = elliott.identify_wave_2(df_15m, wave1, swing_df)
        if wave2:
            wave3 = elliott.identify_wave_3(df_15m, wave1, wave2, swing_df)
            sequence = [wave1, wave2]
            if wave3:
                sequence.append(wave3)
            if elliott.validate_elliott_wave_sequence(sequence):
                elliott_sequences.append(sequence)
    
    print(f"   ✅ Identified {len(wave1_candidates)} Wave 1 candidates")
    print(f"   ✅ Created {len(elliott_sequences)} valid sequences")
    
    print("\n5️⃣ Creating MTF Analysis...")
    mtf_analysis = {
        'swing_points': swing_df,
        'structures': structures,
        'fvgs': fvgs,
        'order_blocks': order_blocks,
        'ote_zones': ote_zones,
        'elliott_sequences': elliott_sequences
    }
    
    print("\n6️⃣ Creating signals...")
    try:
        signals = strategy.generate_signals({}, mtf_analysis)
        
        print(f"   🎯 Created {len(signals)} signals!")
        
        if len(signals) > 0:
            print("\n📊 Signal details:")
            for i, signal in enumerate(signals[:3]):  # Show up to first 3
                print(f"   Signal {i+1}:")
                print(f"     Type: {signal.entry_type}")
                print(f"     Direction: {signal.signal_type}")
                print(f"     Entry price: {signal.price:.2f}")
                print(f"     Stop Loss: {signal.stop_loss:.2f}")
                print(f"     Take Profits: {[f'{tp:.2f}' for tp in signal.take_profits]}")
                print(f"     R:R: {signal.risk_reward:.2f}")
                print(f"     Confidence: {signal.confidence:.2f}")
                print()
            
            # Calculate statistics
            entry_types = {}
            for signal in signals:
                entry_types[signal.entry_type] = entry_types.get(signal.entry_type, 0) + 1
            
            print("📈 Entry type distribution:")
            for entry_type, count in entry_types.items():
                print(f"   {entry_type}: {count} signals")
                
            avg_rr = sum(s.risk_reward for s in signals) / len(signals)
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            print(f"\n📊 Statistics:")
            print(f"   Average R:R: {avg_rr:.2f}")
            print(f"   Average Confidence: {avg_confidence:.2f}")
            
        else:
            print("   ⚠️ No signals created")
            print("   🔍 Check:")
            print("     - HTF Bias set?", strategy.htf_bias)
            print("     - Are there Elliott sequences?", len(elliott_sequences))
            print("     - Are there FVGs/OBs?", len(fvgs), len(order_blocks))
            
    except Exception as e:
        print(f"   ❌ Error creating signals: {str(e)}")
        print(f"   🔍 Stack trace: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ Phase 2 test completed!")
    
    return signals if 'signals' in locals() else []

def test_signal_quality(signals: List):
    """Testing signal quality"""
    if not signals:
        print("❌ No signals to test")
        return
        
    print("\n🔍 Testing signal quality:")
    
    # Testing R:R
    good_rr = sum(1 for s in signals if s.risk_reward >= 2.0)
    print(f"   R:R >= 2.0: {good_rr}/{len(signals)} ({good_rr/len(signals)*100:.1f}%)")
    
    # Testing Confidence
    high_conf = sum(1 for s in signals if s.confidence >= 0.7)
    print(f"   Confidence >= 0.7: {high_conf}/{len(signals)} ({high_conf/len(signals)*100:.1f}%)")
    
    # Testing TP levels
    multi_tp = sum(1 for s in signals if len(s.take_profits) > 1)
    print(f"   Multiple TPs: {multi_tp}/{len(signals)} ({multi_tp/len(signals)*100:.1f}%)")

if __name__ == "__main__":
    """Running the test"""
    print("🎯 Elliott Wave + ICT Trading Strategy - Phase 2 Test")
    print("Goal: Testing implementation of 5 entry types\n")
    
    signals = test_entry_signals()
    test_signal_quality(signals)
    
    print("\n🎉 Test completed!")
    print("📝 Next steps:")
    print("   1. Ensure all 5 entry types are implemented")
    print("   2. Improve confluence scoring")
    print("   3. Add real Parquet data loading")
    print("   4. Move to phase 3 (backtesting)")