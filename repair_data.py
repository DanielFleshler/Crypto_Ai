#!/usr/bin/env python3
"""
Data Repair Script for Crypto Trading Data
Fixes common data quality issues found during verification
"""

import pandas as pd
import os
import glob
import logging
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataRepairer:
    def __init__(self, data_dir="data/raw", backup_dir="data/backup"):
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        
    def create_backup(self):
        """Create backup of original data before repairs"""
        if os.path.exists(self.backup_dir):
            logger.info(f"Backup directory {self.backup_dir} already exists")
            return True
            
        try:
            shutil.copytree(self.data_dir, self.backup_dir)
            logger.info(f"Created backup at {self.backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def get_parquet_files(self, pair, timeframe):
        """Get all parquet files for a pair and timeframe"""
        pattern = os.path.join(self.data_dir, pair, timeframe, "*.parquet")
        return glob.glob(pattern)
    
    def repair_duplicate_timestamps(self, df):
        """Remove duplicate timestamps, keeping the last occurrence"""
        if 'start_time' not in df.columns:
            return df
            
        initial_count = len(df)
        df = df.drop_duplicates(subset=['start_time'], keep='last')
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate timestamps")
        
        return df
    
    def repair_timestamp_ordering(self, df):
        """Sort data by timestamp to fix ordering issues"""
        if 'start_time' not in df.columns:
            return df
            
        df = df.sort_values('start_time').reset_index(drop=True)
        logger.info("Fixed timestamp ordering")
        return df
    
    def repair_ohlc_relationships(self, df):
        """Fix invalid OHLC relationships"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return df
            
        initial_count = len(df)
        
        # Fix cases where high < low by swapping them
        invalid_high_low = df['high'] < df['low']
        if invalid_high_low.any():
            df.loc[invalid_high_low, ['high', 'low']] = df.loc[invalid_high_low, ['low', 'high']].values
            logger.info(f"Fixed {invalid_high_low.sum()} high/low swaps")
        
        # Fix cases where high < open or high < close
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid_high.any():
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
            logger.info(f"Fixed {invalid_high.sum()} high price issues")
        
        # Fix cases where low > open or low > close
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid_low.any():
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
            logger.info(f"Fixed {invalid_low.sum()} low price issues")
        
        return df
    
    def repair_negative_values(self, df):
        """Fix negative prices and volumes"""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    # Replace non-positive values with the previous valid value
                    df[col] = df[col].replace(0, pd.NA)
                    df[col] = df[col].fillna(method='ffill')
                    logger.info(f"Fixed {negative_count} non-positive values in {col}")
        
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                # Replace negative volumes with 0
                df.loc[df['volume'] < 0, 'volume'] = 0
                logger.info(f"Fixed {negative_volume} negative volumes")
        
        return df
    
    def repair_missing_values(self, df):
        """Handle missing values in the data"""
        missing_counts = df.isnull().sum()
        
        for col, count in missing_counts.items():
            if count > 0:
                if col == 'start_time':
                    # For timestamps, we can't interpolate, so we'll drop these rows
                    df = df.dropna(subset=['start_time'])
                    logger.info(f"Dropped {count} rows with missing timestamps")
                else:
                    # For numeric columns, use forward fill
                    df[col] = df[col].fillna(method='ffill')
                    logger.info(f"Forward filled {count} missing values in {col}")
        
        return df
    
    def repair_single_file(self, file_path):
        """Repair a single parquet file"""
        logger.info(f"Repairing {file_path}")
        
        try:
            # Load the data
            df = pd.read_parquet(file_path)
            initial_count = len(df)
            
            if len(df) == 0:
                logger.warning(f"File {file_path} is empty, skipping")
                return False
            
            # Apply all repair functions
            df = self.repair_duplicate_timestamps(df)
            df = self.repair_timestamp_ordering(df)
            df = self.repair_ohlc_relationships(df)
            df = self.repair_negative_values(df)
            df = self.repair_missing_values(df)
            
            # Save the repaired data
            df.to_parquet(file_path, index=False)
            
            final_count = len(df)
            logger.info(f"Repaired {file_path}: {initial_count} -> {final_count} records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to repair {file_path}: {e}")
            return False
    
    def repair_pair_timeframe(self, pair, timeframe):
        """Repair all files for a specific pair and timeframe"""
        logger.info(f"Repairing {pair} {timeframe}")
        
        parquet_files = self.get_parquet_files(pair, timeframe)
        if not parquet_files:
            logger.warning(f"No files found for {pair} {timeframe}")
            return True
        
        success_count = 0
        for file_path in parquet_files:
            if self.repair_single_file(file_path):
                success_count += 1
        
        logger.info(f"Repaired {success_count}/{len(parquet_files)} files for {pair} {timeframe}")
        return success_count == len(parquet_files)
    
    def repair_all_data(self):
        """Repair all data in the directory"""
        logger.info("Starting data repair process...")
        
        # Create backup first
        if not self.create_backup():
            logger.error("Failed to create backup. Aborting repair.")
            return False
        
        # Get all pairs
        pair_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d))]
        
        total_pairs = len(pair_dirs)
        successful_pairs = 0
        
        for i, pair in enumerate(pair_dirs, 1):
            logger.info(f"Processing pair {i}/{total_pairs}: {pair}")
            
            # Get timeframes for this pair
            pair_dir = os.path.join(self.data_dir, pair)
            timeframes = [d for d in os.listdir(pair_dir) 
                         if os.path.isdir(os.path.join(pair_dir, d))]
            
            pair_success = True
            for timeframe in timeframes:
                if not self.repair_pair_timeframe(pair, timeframe):
                    pair_success = False
            
            if pair_success:
                successful_pairs += 1
                logger.info(f"Successfully repaired {pair}")
            else:
                logger.error(f"Failed to repair {pair}")
        
        logger.info(f"Repair completed: {successful_pairs}/{total_pairs} pairs successful")
        return successful_pairs == total_pairs
    
    def generate_repair_report(self):
        """Generate a report of the repair process"""
        logger.info("Generating repair report...")
        
        report = {
            'repair_timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'backup_directory': self.backup_dir,
            'backup_exists': os.path.exists(self.backup_dir)
        }
        
        return report

def main():
    """Main function to run data repair"""
    repairer = DataRepairer()
    
    logger.info("=" * 60)
    logger.info("DATA REPAIR TOOL")
    logger.info("=" * 60)
    logger.info("This tool will fix common data quality issues:")
    logger.info("- Remove duplicate timestamps")
    logger.info("- Fix timestamp ordering")
    logger.info("- Repair invalid OHLC relationships")
    logger.info("- Fix negative prices and volumes")
    logger.info("- Handle missing values")
    logger.info("")
    
    # Ask for confirmation
    response = input("Do you want to proceed with data repair? (y/N): ")
    if response.lower() != 'y':
        logger.info("Repair cancelled by user")
        return
    
    success = repairer.repair_all_data()
    
    if success:
        logger.info("✅ Data repair completed successfully!")
        logger.info(f"Original data backed up to: {repairer.backup_dir}")
    else:
        logger.error("❌ Data repair encountered errors. Check logs for details.")
        logger.info(f"Original data backed up to: {repairer.backup_dir}")
    
    # Generate report
    report = repairer.generate_repair_report()
    logger.info("Repair Report:")
    for key, value in report.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
