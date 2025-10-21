# Test Phase 2 - Entry Types Implementation Test
# עותק-והדבק את הקוד הזה לקובץ בשם test_phase2.py והרץ

"""
בדיקת שלב 2: בדיקת יישום 5 סוגי הכניסות
מטרה: לוודא ש-generate_signals() יוצרת סיגנלים אמיתיים
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import os
import sys

# הוסף את הנתיב לטרמינל אם צריך
# sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_strategy'))

from trading_strategy.data_loader import DataLoader
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.kill_zones import KillZoneDetector
from trading_strategy.trading_strategy import TradingStrategy

def create_sample_ohlc_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    יוצר נתוני OHLC מדומים לבדיקת האלגוריתמים
    כולל תנודות אמיתיות, trends, ותיקונים
    """
    print(f"🔄 יוצר {n_periods} נרות של נתוני OHLC מדומים...")
    
    # נתחיל במחיר בסיס
    start_price = 45000.0
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='15T')
    
    # יצירת מחירים עם תנודות מציאותיות
    np.random.seed(42)  # לשחזור
    
    prices = [start_price]
    for i in range(1, n_periods):
        # תנודתיות בסיסית
        change_pct = np.random.normal(0, 0.002)  # 0.2% תנודתיות ממוצעת
        
        # הוספת מגמות
        if i % 200 == 0:  # כל 200 נרות - מגמה חדשה
            trend = np.random.choice([-1, 1]) * 0.001
        elif i % 50 == 0:  # כל 50 נרות - תיקון
            trend = -trend * 0.5 if 'trend' in locals() else 0
        
        if 'trend' not in locals():
            trend = 0
            
        new_price = prices[-1] * (1 + change_pct + trend)
        prices.append(max(new_price, 1.0))  # מחיר מינימלי 1
    
    # יצירת OHLC מהמחירים
    ohlc_data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i-1]
            
        # יצירת high/low ריאליסטיים
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
    
    print(f"✅ נוצרו {len(df)} נרות")
    print(f"📊 טווח מחירים: {df['low'].min():.0f} - {df['high'].max():.0f}")
    print(f"📈 מחיר התחלה: {df['close'].iloc[0]:.0f}, מחיר סיום: {df['close'].iloc[-1]:.0f}")
    
    return df

def test_entry_signals():
    """בדיקת יצירת סיגנלי כניסה"""
    
    print("🚀 מתחיל בדיקת שלב 2 - יישום סוגי כניסות")
    print("=" * 80)
    
    # יצירת נתוני בדיקה
    df_15m = create_sample_ohlc_data(1000)
    
    print("\n1️⃣ אתחול מערכת...")
    strategy = TradingStrategy()
    strategy.htf_bias = 'BULLISH'  # הגדר bias לבדיקה
    
    print("\n2️⃣ הפעלת זיהוי מבנה שוק...")
    msd = MarketStructureDetector()
    swing_df = msd.detect_swing_points(df_15m, strength=3)
    structures = msd.detect_market_structure(swing_df)
    
    swing_highs_count = swing_df['swing_high'].sum()
    swing_lows_count = swing_df['swing_low'].sum()
    print(f"   ✅ זוהו {swing_highs_count} swing highs, {swing_lows_count} swing lows")
    print(f"   ✅ זוהו {len(structures)} מבני שוק")
    
    if swing_highs_count == 0 or swing_lows_count == 0:
        print("   ⚠️ אין swing points - מקטין את strength parameter")
        swing_df = msd.detect_swing_points(df_15m, strength=2)
        structures = msd.detect_market_structure(swing_df)
        swing_highs_count = swing_df['swing_high'].sum() 
        swing_lows_count = swing_df['swing_low'].sum()
        print(f"   ✅ עם strength=2: {swing_highs_count} swing highs, {swing_lows_count} swing lows")
    
    print("\n3️⃣ הפעלת זיהוי ICT...")
    ict = ICTConceptsDetector()
    fvgs = ict.detect_fvg(df_15m, min_gap_percent=0.05)  # מקטין threshold
    order_blocks = ict.detect_order_blocks(df_15m, swing_df)
    ote_zones = ict.detect_ote_zones(df_15m, swing_df)
    
    print(f"   ✅ זוהו {len(fvgs)} FVGs")
    print(f"   ✅ זוהו {len(order_blocks)} Order Blocks") 
    print(f"   ✅ זוהו {len(ote_zones)} OTE Zones")
    
    print("\n4️⃣ הפעלת זיהוי Elliott Wave...")
    elliott = ElliottWaveDetector()
    wave1_candidates = elliott.identify_wave_1(df_15m, swing_df)
    
    elliott_sequences = []
    for wave1 in wave1_candidates[:5]:  # בדוק רק 5 הראשונים
        wave2 = elliott.identify_wave_2(df_15m, wave1, swing_df)
        if wave2:
            wave3 = elliott.identify_wave_3(df_15m, wave1, wave2, swing_df)
            sequence = [wave1, wave2]
            if wave3:
                sequence.append(wave3)
            if elliott.validate_elliott_wave_sequence(sequence):
                elliott_sequences.append(sequence)
    
    print(f"   ✅ זוהו {len(wave1_candidates)} מועמדי Wave 1")
    print(f"   ✅ נוצרו {len(elliott_sequences)} רצפים תקפים")
    
    print("\n5️⃣ יצירת MTF Analysis...")
    mtf_analysis = {
        'swing_points': swing_df,
        'structures': structures,
        'fvgs': fvgs,
        'order_blocks': order_blocks,
        'ote_zones': ote_zones,
        'elliott_sequences': elliott_sequences
    }
    
    print("\n6️⃣ יצירת סיגנלים...")
    try:
        signals = strategy.generate_signals({}, mtf_analysis)
        
        print(f"   🎯 נוצרו {len(signals)} סיגנלים!")
        
        if len(signals) > 0:
            print("\n📊 פירוט סיגנלים:")
            for i, signal in enumerate(signals[:3]):  # הצג עד 3 הראשונים
                print(f"   סיגנל {i+1}:")
                print(f"     סוג: {signal.entry_type}")
                print(f"     כיוון: {signal.signal_type}")
                print(f"     מחיר כניסה: {signal.price:.2f}")
                print(f"     Stop Loss: {signal.stop_loss:.2f}")
                print(f"     Take Profits: {[f'{tp:.2f}' for tp in signal.take_profits]}")
                print(f"     R:R: {signal.risk_reward:.2f}")
                print(f"     Confidence: {signal.confidence:.2f}")
                print()
            
            # חישוב סטטיסטיקות
            entry_types = {}
            for signal in signals:
                entry_types[signal.entry_type] = entry_types.get(signal.entry_type, 0) + 1
            
            print("📈 התפלגות סוגי כניסות:")
            for entry_type, count in entry_types.items():
                print(f"   {entry_type}: {count} סיגנלים")
                
            avg_rr = sum(s.risk_reward for s in signals) / len(signals)
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            print(f"\n📊 סטטיסטיקות:")
            print(f"   R:R ממוצע: {avg_rr:.2f}")
            print(f"   Confidence ממוצע: {avg_confidence:.2f}")
            
        else:
            print("   ⚠️ לא נוצרו סיגנלים")
            print("   🔍 בדוק:")
            print("     - HTF Bias מוגדר?", strategy.htf_bias)
            print("     - יש Elliott sequences?", len(elliott_sequences))
            print("     - יש FVGs/OBs?", len(fvgs), len(order_blocks))
            
    except Exception as e:
        print(f"   ❌ שגיאה ביצירת סיגנלים: {str(e)}")
        print(f"   🔍 Stack trace: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ בדיקת שלב 2 הושלמה!")
    
    return signals if 'signals' in locals() else []

def test_signal_quality(signals: List):
    """בדיקת איכות הסיגנלים"""
    if not signals:
        print("❌ אין סיגנלים לבדיקה")
        return
        
    print("\n🔍 בדיקת איכות סיגנלים:")
    
    # בדיקת R:R
    good_rr = sum(1 for s in signals if s.risk_reward >= 2.0)
    print(f"   R:R >= 2.0: {good_rr}/{len(signals)} ({good_rr/len(signals)*100:.1f}%)")
    
    # בדיקת Confidence
    high_conf = sum(1 for s in signals if s.confidence >= 0.7)
    print(f"   Confidence >= 0.7: {high_conf}/{len(signals)} ({high_conf/len(signals)*100:.1f}%)")
    
    # בדיקת TP levels
    multi_tp = sum(1 for s in signals if len(s.take_profits) > 1)
    print(f"   Multiple TPs: {multi_tp}/{len(signals)} ({multi_tp/len(signals)*100:.1f}%)")

if __name__ == "__main__":
    """הפעלת הבדיקה"""
    print("🎯 Elliott Wave + ICT Trading Strategy - Phase 2 Test")
    print("מטרה: בדיקת יישום 5 סוגי הכניסות\n")
    
    signals = test_entry_signals()
    test_signal_quality(signals)
    
    print("\n🎉 בדיקה הושלמה!")
    print("📝 השלבים הבאים:")
    print("   1. וודא שכל 5 סוגי הכניסות מיושמים")
    print("   2. שפר את confluence scoring")
    print("   3. הוסף real Parquet data loading")
    print("   4. עבור לשלב 3 (backtesting)")