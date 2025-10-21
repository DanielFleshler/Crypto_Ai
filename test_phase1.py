import pandas as pd
from trading_strategy.data_loader import DataLoader
from trading_strategy.market_structure import MarketStructureDetector
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.kill_zones import KillZoneDetector
from trading_strategy.trading_strategy import TradingStrategy

def main():
    base_path = "."  # Current directory (project root)
    pair = "BTCUSDT"
    start_date = "2025-09-01"
    end_date   = "2025-09-30"
    timeframe  = "15m"

    print("1️⃣ טוען נתונים...")
    dl = DataLoader(base_path=base_path)
    data = dl.load_pair_data(pair, [timeframe], start_date, end_date)
    df = data[timeframe]
    print(f"Rows loaded: {len(df)}")
    print(df.head(), "\n")

    print("2️⃣ בדיקת Swing Points ו-BOS/CHoCH...")
    msd = MarketStructureDetector()
    swing_df = msd.detect_swing_points(df, strength=2)
    structures = msd.detect_market_structure(swing_df)
    print(f"Swing Highs: {swing_df['swing_high'].sum()}, Swing Lows: {swing_df['swing_low'].sum()}")
    print(f"Structures detected: {len(structures)}")
    for s in structures[:5]:
        print(" ", s)
    print()

    print("3️⃣ בדיקת FVG ו-Order Blocks...")
    ict = ICTConceptsDetector()
    fvgs = ict.detect_fvg(df, min_gap_percent=0.1)
    obs  = ict.detect_order_blocks(df, swing_df)
    print(f"FVGs detected: {len(fvgs)}, Order Blocks: {len(obs)}")
    print("  Examples FVG:", fvgs[:3])
    print("  Examples OB:",  obs[:3], "\n")

    print("4️⃣ בדיקת Elliott Wave (גלים 1–3)...")
    ewd = ElliottWaveDetector()
    wave1s = ewd.identify_wave_1(df, swing_df)
    wave2  = wave1s and ewd.identify_wave_2(df, wave1s[0], swing_df)
    wave3  = wave2 and ewd.identify_wave_3(df, wave1s[0], wave2, swing_df)
    print(f"Wave 1 candidates: {len(wave1s)}")
    print("Wave 2 identified:", wave2)
    print("Wave 3 identified:", wave3, "\n")

    print("5️⃣ בדיקת Kill Zones...")
    kzd = KillZoneDetector()
    df_z = kzd.mark_kill_zones(df)
    print(df_z[['kill_zone','is_asia','is_london','is_ny']].head(10))
    print("Kill Zones count:")
    print(df_z['kill_zone'].value_counts(), "\n")

    print("6️⃣ בדיקת run_analysis() במערכת הראשית...")
    ts = TradingStrategy(base_path=base_path)
    res = ts.run_analysis(pair=pair, start_date=start_date, end_date=end_date)
    print("run_analysis() result:", res)

if __name__ == "__main__":
    main()
