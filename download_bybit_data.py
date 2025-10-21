#!/usr/bin/env python3
"""
Bybit Trading Data Downloader
Downloads historical trading data for cryptocurrency pairs from Bybit API
"""

import requests
import pandas as pd
import yaml
import time
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BybitDataDownloader:
    def __init__(self):
        self.base_url = "https://api.bybit.com/v5/market/kline"
        self.timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.start_date = datetime(2021, 1, 1)
        self.end_date = datetime(2025, 10, 19)
        
    def load_pairs(self, config_path):
        """Load trading pairs from YAML config file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['pairs']
    
    def convert_pair_format(self, pair):
        """Convert pair format from BTC/USDT to BTCUSDT"""
        return pair.replace('/', '')
    
    def get_timeframe_interval(self, timeframe):
        """Convert timeframe to Bybit interval format"""
        timeframe_map = {
            '5m': '5',
            '15m': '15', 
            '30m': '30',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")
        return timeframe_map[timeframe]
    
    def download_kline_data(self, symbol, interval, start_time, end_time, max_retries=3):
        """Download kline data from Bybit API with retry logic"""
        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval,
            'start': int(start_time.timestamp() * 1000),
            'end': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data['retCode'] == 0:
                    kline_data = data['result']['list']
                    # Validate data completeness
                    if len(kline_data) < 100:  # Expected minimum records
                        logger.warning(f"Partial data received for {symbol}: {len(kline_data)} records")
                    return kline_data
                else:
                    logger.error(f"API Error for {symbol}: {data['retMsg']}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {symbol} (attempt {attempt + 2}/{max_retries})")
                        time.sleep(0.5)  # Brief delay before retry
                    return []
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Longer delay for network errors
                return []
    
    def process_kline_data(self, kline_data):
        """Process and convert kline data to DataFrame"""
        if not kline_data:
            logger.warning("No kline data provided")
            return pd.DataFrame()
        
        df = pd.DataFrame(kline_data, columns=[
            'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert timestamp to datetime
        df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
        
        # Convert price and volume columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        coerced_count = 0
        for col in numeric_columns:
            original_count = len(df[col])
            df[col] = pd.to_numeric(df[col], errors='coerce')
            coerced_count += original_count - df[col].count()
        
        if coerced_count > 0:
            logger.warning(f"Coerced {coerced_count} non-numeric values to NaN")
            if df.isnull().all().any():
                logger.error("Some columns are entirely NaN after coercion")
                return pd.DataFrame()  # Return empty DataFrame for invalid data
        
        return df
    
    def save_data(self, df, symbol, timeframe, output_dir):
        """Save DataFrame to parquet file"""
        if df.empty:
            return
        
        # Create directory structure
        symbol_dir = os.path.join(output_dir, symbol)
        timeframe_dir = os.path.join(symbol_dir, timeframe)
        os.makedirs(timeframe_dir, exist_ok=True)
        
        # Group by year and month for partitioning
        df['year'] = df['start_time'].dt.year
        df['month'] = df['start_time'].dt.month
        
        for (year, month), group in df.groupby(['year', 'month']):
            filename = f"{symbol}_{timeframe}_{year}-{month:02d}.parquet"
            filepath = os.path.join(timeframe_dir, filename)
            group.drop(['year', 'month'], axis=1).to_parquet(filepath, index=False)
            logger.info(f"Saved {len(group)} records to {filepath}")
    
    def download_pair_data(self, pair, output_dir):
        """Download data for a single pair across all timeframes"""
        symbol = self.convert_pair_format(pair)
        logger.info(f"Downloading data for {pair} ({symbol})")
        
        for timeframe in self.timeframes:
            logger.info(f"Processing timeframe: {timeframe}")
            interval = self.get_timeframe_interval(timeframe)
            
            # Download data in chunks to handle large date ranges
            current_start = self.start_date
            all_data = []
            
            while current_start < self.end_date:
                # Calculate chunk end date (max 1000 records per request)
                if timeframe == '5m':
                    chunk_end = min(current_start + timedelta(days=3), self.end_date)
                elif timeframe == '15m':
                    chunk_end = min(current_start + timedelta(days=10), self.end_date)
                elif timeframe == '30m':
                    chunk_end = min(current_start + timedelta(days=20), self.end_date)
                elif timeframe == '1h':
                    chunk_end = min(current_start + timedelta(days=40), self.end_date)
                elif timeframe == '4h':
                    chunk_end = min(current_start + timedelta(days=160), self.end_date)
                else:  # 1d
                    chunk_end = min(current_start + timedelta(days=1000), self.end_date)
                
                kline_data = self.download_kline_data(symbol, interval, current_start, chunk_end)
                
                if kline_data:
                    all_data.extend(kline_data)
                    logger.info(f"Downloaded {len(kline_data)} records for {pair} {timeframe} from {current_start.date()} to {chunk_end.date()}")
                
                current_start = chunk_end
                time.sleep(0.05)  # Reduced rate limiting (still respectful)
            
            # Process and save all data
            if all_data:
                df = self.process_kline_data(all_data)
                if not df.empty:
                    self.save_data(df, symbol, timeframe, output_dir)
                    logger.info(f"Completed {pair} {timeframe}: {len(df)} total records")
                else:
                    logger.warning(f"No data processed for {pair} {timeframe}")
            else:
                logger.warning(f"No data downloaded for {pair} {timeframe}")
    
    def download_all_data(self, config_path, output_dir):
        """Download data for all pairs"""
        pairs = self.load_pairs(config_path)
        logger.info(f"Starting download for {len(pairs)} pairs")
        
        for i, pair in enumerate(pairs, 1):
            logger.info(f"Processing pair {i}/{len(pairs)}: {pair}")
            try:
                self.download_pair_data(pair, output_dir)
                logger.info(f"Completed {pair}")
            except Exception as e:
                logger.error(f"Error processing {pair}: {e}")
                continue
            
            # Rate limiting between pairs
            time.sleep(0.5)  # Reduced delay between pairs
        
        logger.info("Download completed!")

def main():
    downloader = BybitDataDownloader()
    config_path = "config/pairs.yaml"
    output_dir = "data/raw"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start download
    downloader.download_all_data(config_path, output_dir)

if __name__ == "__main__":
    main()
