import pandas as pd

class KillZoneDetector:
    def identify_kill_zone(self, ts: pd.Timestamp)->str:
        h=ts.hour
        if h<8: return 'ASIA'
        if 8<=h<16:
            return 'LONDON_NY_OVERLAP' if h<16 and h>=13 else 'LONDON'
        if 16<=h<21: return 'NEW_YORK'
        return 'OFF_HOURS'

    def mark_kill_zones(self, df: pd.DataFrame)->pd.DataFrame:
        df=df.copy()
        df['kill_zone']=df.index.map(self.identify_kill_zone)
        df['is_asia']=df['kill_zone']=='ASIA'
        df['is_london']=df['kill_zone'].isin(['LONDON','LONDON_NY_OVERLAP'])
        df['is_ny']=df['kill_zone'].isin(['NEW_YORK','LONDON_NY_OVERLAP'])
        df['is_overlap']=df['kill_zone']=='LONDON_NY_OVERLAP'
        return df
