#!/usr/bin/env python3
"""
Diagnostic: Investigate why 99% of ICT signals are BUY in bearish market
"""

from trading_strategy.data_loader import DataLoader
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.config_loader import ConfigLoader

def main():
    # Load 2023 data (bearish period)
    data_loader = DataLoader(base_path='data/raw')
    config_loader = ConfigLoader()

    print("="*70)
    print("DIAGNOSTIC: ICT Signal Generation Bias")
    print("Period: 2023 (BEARISH market)")
    print("="*70)
    print()

    # Load data
    data = data_loader.load_pair_data(
        'BTCUSDT',
        ['1h', '15m'],
        '2023-01-01',
        '2023-03-31'  # Just Q1 for faster testing
    )

    print(f"Loaded {len(data['15m'])} candles of 15m data")
    print()

    # Analyze FVG detection
    ict_detector = ICTConceptsDetector(config_loader)
    ict_detector.data = data['15m']

    fvgs = ict_detector.detect_fvg(data['15m'])

    bullish_fvgs = [f for f in fvgs if f.is_bullish()]
    bearish_fvgs = [f for f in fvgs if f.is_bearish()]

    print("FVG Detection Results:")
    print("-" * 50)
    print(f"Total FVGs detected: {len(fvgs)}")
    print(f"Bullish FVGs: {len(bullish_fvgs)} ({(len(bullish_fvgs)/len(fvgs)*100) if fvgs else 0:.1f}%)")
    print(f"Bearish FVGs: {len(bearish_fvgs)} ({(len(bearish_fvgs)/len(fvgs)*100) if fvgs else 0:.1f}%)")
    print()

    # Sample some FVGs
    print("Sample Bullish FVGs (first 5):")
    for fvg in bullish_fvgs[:5]:
        print(f"  {fvg.timestamp}: {fvg.concept_type} ${fvg.start_price:.2f}-${fvg.end_price:.2f}")

    print()
    print("Sample Bearish FVGs (first 5):")
    for fvg in bearish_fvgs[:5]:
        print(f"  {fvg.timestamp}: {fvg.concept_type} ${fvg.start_price:.2f}-${fvg.end_price:.2f}")

    print()
    print("="*70)
    print("KEY INSIGHT")
    print("="*70)

    if len(bullish_fvgs) > len(bearish_fvgs) * 5:
        print("⚠️  FOUND ISSUE: Massive imbalance in FVG detection!")
        print(f"   Bullish FVGs outnumber Bearish by {len(bullish_fvgs)/len(bearish_fvgs) if bearish_fvgs else 'INF'}x")
        print()
        print("EXPLANATION:")
        print("  In a BEARISH downtrend:")
        print("  - Price moves down aggressively (creates bullish retracement FVGs)")
        print("  - Bullish FVGs form when price gaps down then bounces")
        print("  - Bearish FVGs form when price gaps up (rare in downtrend)")
        print()
        print("  This is NORMAL market behavior but causes problem:")
        print("  - ICT strategy detects more bullish FVGs")
        print("  - Tries to enter BUY signals (retracement trades)")
        print("  - But HTF bias is BEARISH, so signals get filtered out")
        print("  - Result: Very few signals pass")
        print()
        print("SOLUTION:")
        print("  In BEARISH bias, we should:")
        print("  1. Trade bearish FVGs (continuation)")
        print("  2. OR trade bullish FVGs but as SELL opportunities")
        print("     (wait for retracement into bullish FVG, then short the rejection)")
    else:
        print("✓ FVG detection is balanced")

    print()
    print("="*70)
    print("RECOMMENDED FIXES")
    print("="*70)
    print("1. Add 'counter-trend entry' logic:")
    print("   - BEARISH bias + bullish FVG = Wait for rejection, then SELL")
    print("   - BULLISH bias + bearish FVG = Wait for rejection, then BUY")
    print()
    print("2. Adjust HTF bias filtering:")
    print("   - Don't filter out opposing FVGs")
    print("   - Instead, use them as reversal/rejection zones")
    print()
    print("3. Or simplify: Trade only with-trend FVGs:")
    print("   - Keep current filter but improve bear FVG detection")
    print("   - Focus on consolidations and mini-ranges for bear FVGs")

if __name__ == '__main__':
    main()
