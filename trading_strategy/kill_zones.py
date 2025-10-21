# Phase 1.5: Kill Zones Detection (Trading Sessions)
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class KillZone:
    name: str
    start_time: datetime
    end_time: datetime
    description: str
class KillZoneDetector:
    """
    Detect trading session kill zones: Asia, London, New York
    Based on your strategy: Asia builds liquidity, London fake breakouts, NY real moves
    """
    
    def __init__(self):
        # Kill zone times in UTC (adjust for your timezone if needed)
        self.kill_zones = {
            'ASIA': {
                'start': '00:00',
                'end': '08:00',
                'description': 'Liquidity building, range formation'
            },
            'LONDON': {
                'start': '08:00', 
                'end': '16:00',
                'description': 'Fake breakouts, liquidity grabs'
            },
            'NEW_YORK': {
                'start': '13:00',  # Overlaps with London
                'end': '21:00',
                'description': 'Real institutional moves'
            }
        }
    
    def identify_kill_zone(self, timestamp: pd.Timestamp) -> str:
        """
        Identify which kill zone a timestamp belongs to
        
        Args:
            timestamp: Pandas timestamp
            
        Returns:
            Kill zone name ('ASIA', 'LONDON', 'NEW_YORK', 'OFF_HOURS')
        """
        hour = timestamp.hour
        
        # Asia Kill Zone: 00:00 - 08:00 UTC
        if 0 <= hour < 8:
            return 'ASIA'
        
        # London Kill Zone: 08:00 - 16:00 UTC  
        elif 8 <= hour < 16:
            return 'LONDON'
        
        # New York Kill Zone: 13:00 - 21:00 UTC (overlaps with London)
        elif 13 <= hour < 21:
            if hour < 16:
                return 'LONDON_NY_OVERLAP'  # Most volatile period
            else:
                return 'NEW_YORK'
        
        else:
            return 'OFF_HOURS'
    
    def mark_kill_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add kill zone information to DataFrame
        
        Args:
            df: OHLC DataFrame with datetime index or start_time column
            
        Returns:
            DataFrame with kill zone columns added
        """
        df = df.copy()
        
        # Check if we have a datetime index or start_time column
        if hasattr(df.index, 'hour'):  # Datetime index
            df['kill_zone'] = df.index.map(self.identify_kill_zone)
        elif 'start_time' in df.columns:  # start_time column
            df['kill_zone'] = df['start_time'].map(self.identify_kill_zone)
        else:
            # If no datetime info available, set all to OFF_HOURS
            df['kill_zone'] = 'OFF_HOURS'
        
        # Add binary columns for each session
        df['is_asia'] = df['kill_zone'] == 'ASIA'
        df['is_london'] = df['kill_zone'].isin(['LONDON', 'LONDON_NY_OVERLAP'])
        df['is_ny'] = df['kill_zone'].isin(['NEW_YORK', 'LONDON_NY_OVERLAP'])
        df['is_overlap'] = df['kill_zone'] == 'LONDON_NY_OVERLAP'
        
        return df
    
    def get_session_liquidity_levels(self, df: pd.DataFrame, session: str) -> Dict[str, float]:
        """
        Get liquidity levels (highs/lows) for a specific session
        
        Args:
            df: DataFrame with kill zones marked
            session: 'ASIA', 'LONDON', 'NEW_YORK'
            
        Returns:
            Dict with session high/low levels
        """
        session_data = None
        
        if session == 'ASIA':
            session_data = df[df['is_asia']]
        elif session == 'LONDON':
            session_data = df[df['is_london']]
        elif session == 'NEW_YORK':
            session_data = df[df['is_ny']]
        
        if session_data is None or len(session_data) == 0:
            return {'high': None, 'low': None}
        
        return {
            'high': session_data['high'].max(),
            'low': session_data['low'].min(),
            'session_start': session_data.index[0],
            'session_end': session_data.index[-1]
        }

print("🕐 KillZoneDetector Class Created")
print("✅ Asia, London, New York session detection")
print("✅ Session overlap identification")
print("✅ Liquidity level extraction per session")
print("⏰ Ready for session-based analysis")