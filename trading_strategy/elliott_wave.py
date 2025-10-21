from typing import Dict, List
from .data_structures import ElliottWave

class ElliottWaveDetector:
    def __init__(self):
        self.fibs={'retracement':[0.236,0.382,0.5,0.618,0.786],
                   'extension':[1.0,1.272,1.414,1.618,2.0,2.272,2.618]}

    def calculate_fibonacci_retracement(self, s,e)->Dict:
        d=abs(e-s); r={}
        for f in self.fibs['retracement']:
            r[f'fib_{f}']= s+(e-s)*f if s<e else s-(s-e)*f
        return r

    def calculate_fibonacci_extension(self,s,e,x)->Dict:
        d=abs(e-s); r={}
        for f in self.fibs['extension']:
            r[f'ext_{f}']= x+d*f if s<e else x-d*f
        return r

    def identify_wave_1(self, df, swing_df)->List[ElliottWave]:
        swings=[]
        for idx,row in swing_df.iterrows():
            if row['swing_high'] or row['swing_low']:
                swings.append((idx,row['swing_high_price']
                              if row['swing_high'] else row['swing_low_price']))
        seq=[]
        for i in range(1,len(swings)):
            prev,cur=swings[i-1],swings[i]
            pct=abs(cur[1]-prev[1])/prev[1]*100
            if pct>3:
                seq.append(ElliottWave(1,'IMPULSE',prev[0],cur[0],
                                       prev[1],cur[1],'current',{}))
        return seq

    def identify_wave_2(self, df, w1, swing_df):
        fibs=self.calculate_fibonacci_retracement(w1.start_price,
                                                  w1.end_price)
        idx=df.index.get_loc(w1.end_time)
        for i in range(idx+1,min(idx+50,len(df))):
            row=swing_df.iloc[i]
            if row['swing_low'] and w1.start_price< w1.end_price:
                p=row['swing_low_price']
                # Simple retracement check - wave 2 should be between 23.6% and 78.6%
                if fibs['fib_0.236']<=p<=fibs['fib_0.786']:
                    return ElliottWave(2,'CORRECTIVE',
                                       w1.end_time,df.index[i],
                                       w1.end_price,p,'current',fibs)
            if row['swing_high'] and w1.start_price>w1.end_price:
                p=row['swing_high_price']
                if fibs['fib_0.786']<=p<=fibs['fib_0.236']:
                    return ElliottWave(2,'CORRECTIVE',
                                       w1.end_time,df.index[i],
                                       w1.end_price,p,'current',fibs)
        return None

    def identify_wave_3(self, df, w1, w2, swing_df):
        exts=self.calculate_fibonacci_extension(
            w1.start_price,w1.end_price,w2.end_price)
        idx=df.index.get_loc(w2.end_time)
        for i in range(idx+1,min(idx+100,len(df))):
            row=swing_df.iloc[i]
            if row['swing_high'] and w1.start_price < w1.end_price:
                p=row['swing_high_price']
                # Wave 3 should be higher than wave 1 end, but more flexible
                if p > w2.end_price and p >= w1.end_price * 0.99:
                    return ElliottWave(3,'IMPULSE',
                                       w2.end_time,df.index[i],
                                       w2.end_price,p,'current',exts)
            if row['swing_low'] and w1.start_price > w1.end_price:
                p=row['swing_low_price']
                if p < w2.end_price and p <= w1.end_price * 1.01:
                    return ElliottWave(3,'IMPULSE',
                                       w2.end_time,df.index[i],
                                       w2.end_price,p,'current',exts)
        return None

    def validate_elliott_wave_sequence(self, waves: List[ElliottWave]) -> bool:
        if len(waves)<3: return False
        w1,w2,w3=waves[0],waves[1],waves[2]
        # For bullish sequence: w1 up, w2 down, w3 up
        if w1.start_price < w1.end_price:  # Bullish wave 1
            if w2.end_price < w2.start_price and w3.end_price > w3.start_price:  # Corrective w2, impulse w3
                return True
        # For bearish sequence: w1 down, w2 up, w3 down
        elif w1.start_price > w1.end_price:  # Bearish wave 1
            if w2.end_price > w2.start_price and w3.end_price < w3.start_price:  # Corrective w2, impulse w3
                return True
        return False
