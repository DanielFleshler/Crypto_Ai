import os
import pandas as pd
from typing import List, Dict

class DataLoader:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.timeframes = ['5m','15m','30m','1h','4h','1d']

    def load_pair_data(self, pair: str,
                       timeframes: List[str] = None,
                       start_date: str = None,
                       end_date: str = None) -> Dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = self.timeframes

        pair_data = {}
        ym_months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y-%m")
        for tf in timeframes:
            dfs = []
            for ym in ym_months:
                path = os.path.join(self.base_path, "data", "raw", pair, tf, f"{pair}_{tf}_{ym}.parquet")
                if os.path.exists(path):
                    dfm = pd.read_parquet(path)
                    dfs.append(dfm)
            if dfs:
                df = pd.concat(dfs)
                df.sort_values('start_time', inplace=True)
                df.set_index('start_time', inplace=True)
                pair_data[tf] = df[start_date:end_date]
            else:
                cols = ['open','high','low','close','volume','turnover']
                pair_data[tf] = pd.DataFrame(columns=cols)
        return pair_data
