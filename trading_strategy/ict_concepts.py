import numpy as np
from typing import List
from .data_structures import ICTConcept

class ICTConceptsDetector:
    def detect_fvg(self, df, min_gap_percent=0.1)->List[ICTConcept]:
        res=[]
        for i in range(2,len(df)):
            c1, c3 = df.iloc[i-2], df.iloc[i]
            if c1['high'] < c3['low']:
                gap=(c3['low']-c1['high']); pct=gap/c1['high']*100
                if pct>=min_gap_percent:
                    res.append(ICTConcept(df.index[i],'FVG_BULLISH',
                        c1['high'],c3['low'],'current',min(pct/2,1.0)))
            elif c1['low'] > c3['high']:
                gap=(c1['low']-c3['high']); pct=gap/c1['low']*100
                if pct>=min_gap_percent:
                    res.append(ICTConcept(df.index[i],'FVG_BEARISH',
                        c3['high'],c1['low'],'current',min(pct/2,1.0)))
        return res

    def detect_order_blocks(self, df, swing_df)->List[ICTConcept]:
        res=[]
        sl, sh = swing_df[swing_df['swing_low']], swing_df[swing_df['swing_high']]
        for idx,row in sl.iterrows():
            loc = df.index.get_loc(idx)
            for j in range(loc-1, max(loc-10,0), -1):
                c=df.iloc[j]
                if c['close']<c['open'] and c['low']<=row['swing_low_price']*1.01:
                    res.append(ICTConcept(df.index[j],'OB_BULLISH',
                                          c['low'],c['high'],'current',0.7))
                    break
        for idx,row in sh.iterrows():
            loc=df.index.get_loc(idx)
            for j in range(loc-1,max(loc-10,0),-1):
                c=df.iloc[j]
                if c['close']>c['open'] and c['high']>=row['swing_high_price']*0.99:
                    res.append(ICTConcept(df.index[j],'OB_BEARISH',
                                          c['low'],c['high'],'current',0.7))
                    break
        return res

    def detect_breaker_blocks(self, df, obs)->List[ICTConcept]:
        res=[]
        for ob in obs:
            idx=df.index.get_loc(ob.timestamp)
            for i in range(idx+1,len(df)):
                c=df.iloc[i]
                if ob.concept_type=='OB_BULLISH' and c['low']<ob.start_price:
                    res.append(ICTConcept(ob.timestamp,'BB_BEARISH',
                                           ob.start_price,ob.end_price,'current',0.8))
                    break
                if ob.concept_type=='OB_BEARISH' and c['high']>ob.end_price:
                    res.append(ICTConcept(ob.timestamp,'BB_BULLISH',
                                           ob.start_price,ob.end_price,'current',0.8))
                    break
        return res

    def calculate_ote_levels(self, s,e):
        d=e-s
        return {'ote_start':s+0.62*d,'ote_end':s+0.79*d,
                'fib_50':s+0.50*d,'fib_618':s+0.618*d,'fib_786':s+0.786*d}

    def detect_ote_zones(self, df, swing_df, lookback=20)->List[ICTConcept]:
        res=[]; highs=swing_df[swing_df['swing_high']].tail(lookback)
        lows=swing_df[swing_df['swing_low']].tail(lookback)
        for ih,hr in highs.iterrows():
            for il,lr in lows.iterrows():
                if abs((ih-il).days)<=10 and hr['swing_high_price']>lr['swing_low_price']:
                    levels=self.calculate_ote_levels(
                        lr['swing_low_price'],hr['swing_high_price'])
                    res.append(ICTConcept(max(ih,il),'OTE_BULLISH',
                                           levels['ote_start'],levels['ote_end'],
                                           'current',0.9))
        return res
