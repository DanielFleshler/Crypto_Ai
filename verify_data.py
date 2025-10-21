#!/usr/bin/env python3
"""
Data Verification Script for Crypto Trading Data
Verifies the integrity, completeness, and quality of downloaded trading data
"""

import pandas as pd
import yaml
import os
import glob
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataVerifier:
    def __init__(self, data_dir="data/raw", config_path="config/pairs.yaml"):
        self.data_dir = data_dir
        self.config_path = config_path
        self.expected_pairs = []
        self.expected_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        self.expected_columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        self.verification_results = {}
        
    def load_expected_pairs(self):
        """Load expected trading pairs from config file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.expected_pairs = [pair.replace('/', '') for pair in config['pairs']]
            logger.info(f"Loaded {len(self.expected_pairs)} expected pairs from config")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def get_available_pairs(self):
        """Get list of available pairs in data directory"""
        pair_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d)) and d != '__pycache__']
        return sorted(pair_dirs)
    
    def get_available_timeframes(self, pair):
        """Get available timeframes for a specific pair"""
        pair_dir = os.path.join(self.data_dir, pair)
        if not os.path.exists(pair_dir):
            return []
        return [d for d in os.listdir(pair_dir) 
               if os.path.isdir(os.path.join(pair_dir, d))]
    
    def get_parquet_files(self, pair, timeframe):
        """Get all parquet files for a pair and timeframe"""
        pattern = os.path.join(self.data_dir, pair, timeframe, "*.parquet")
        return glob.glob(pattern)
    
    def verify_file_structure(self):
        """Verify the overall file structure"""
        logger.info("Verifying file structure...")
        issues = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            issues.append(f"Data directory {self.data_dir} does not exist")
            return issues
        
        # Get available pairs
        available_pairs = self.get_available_pairs()
        logger.info(f"Found {len(available_pairs)} pairs: {available_pairs}")
        
        # Check for missing expected pairs
        missing_pairs = set(self.expected_pairs) - set(available_pairs)
        if missing_pairs:
            issues.append(f"Missing expected pairs: {missing_pairs}")
        
        # Check for unexpected pairs
        unexpected_pairs = set(available_pairs) - set(self.expected_pairs)
        if unexpected_pairs:
            issues.append(f"Unexpected pairs found: {unexpected_pairs}")
        
        # Verify each pair's structure
        for pair in available_pairs:
            pair_issues = self.verify_pair_structure(pair)
            issues.extend(pair_issues)
        
        return issues
    
    def verify_pair_structure(self, pair):
        """Verify structure for a specific pair"""
        issues = []
        pair_dir = os.path.join(self.data_dir, pair)
        
        # Check available timeframes
        available_timeframes = self.get_available_timeframes(pair)
        missing_timeframes = set(self.expected_timeframes) - set(available_timeframes)
        if missing_timeframes:
            issues.append(f"Pair {pair}: Missing timeframes {missing_timeframes}")
        
        # Verify each timeframe
        for timeframe in available_timeframes:
            timeframe_issues = self.verify_timeframe_structure(pair, timeframe)
            issues.extend(timeframe_issues)
        
        return issues
    
    def verify_timeframe_structure(self, pair, timeframe):
        """Verify structure for a specific pair and timeframe"""
        issues = []
        parquet_files = self.get_parquet_files(pair, timeframe)
        
        if not parquet_files:
            issues.append(f"Pair {pair}, timeframe {timeframe}: No parquet files found")
            return issues
        
        # Check file naming convention
        for file_path in parquet_files:
            filename = os.path.basename(file_path)
            expected_pattern = f"{pair}_{timeframe}_"
            if not filename.startswith(expected_pattern):
                issues.append(f"Pair {pair}, timeframe {timeframe}: Unexpected filename {filename}")
        
        return issues
    
    def verify_data_quality(self, pair, timeframe, sample_size=5):
        """Verify data quality for a specific pair and timeframe"""
        logger.info(f"Verifying data quality for {pair} {timeframe}...")
        issues = []
        
        parquet_files = self.get_parquet_files(pair, timeframe)
        if not parquet_files:
            return issues
        
        # Sample a few files for detailed verification
        sample_files = parquet_files[:sample_size] if len(parquet_files) > sample_size else parquet_files
        
        for file_path in sample_files:
            try:
                df = pd.read_parquet(file_path)
                file_issues = self.verify_dataframe_quality(df, pair, timeframe, file_path)
                issues.extend(file_issues)
            except Exception as e:
                issues.append(f"Error reading {file_path}: {e}")
        
        return issues
    
    def verify_dataframe_quality(self, df, pair, timeframe, file_path):
        """Verify quality of a single DataFrame"""
        issues = []
        filename = os.path.basename(file_path)
        
        # Check columns
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            issues.append(f"{filename}: Missing columns {missing_columns}")
        
        unexpected_columns = set(df.columns) - set(self.expected_columns)
        if unexpected_columns:
            issues.append(f"{filename}: Unexpected columns {unexpected_columns}")
        
        # Check data types
        expected_dtypes = {
            'start_time': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'turnover': 'float64'
        }
        
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    issues.append(f"{filename}: Column {col} has wrong dtype {actual_dtype}, expected {expected_dtype}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                issues.append(f"{filename}: Column {col} has {count} missing values")
        
        # Check for negative prices/volumes
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    issues.append(f"{filename}: Column {col} has {negative_count} non-positive values")
        
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"{filename}: Volume column has {negative_volume} negative values")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = ((df['high'] < df['low']) | 
                           (df['high'] < df['open']) | 
                           (df['high'] < df['close']) |
                           (df['low'] > df['open']) | 
                           (df['low'] > df['close'])).sum()
            if invalid_ohlc > 0:
                issues.append(f"{filename}: {invalid_ohlc} rows have invalid OHLC relationships")
        
        # Check timestamp ordering
        if 'start_time' in df.columns and len(df) > 1:
            if not df['start_time'].is_monotonic_increasing:
                issues.append(f"{filename}: Timestamps are not in chronological order")
        
        # Check for duplicate timestamps
        if 'start_time' in df.columns:
            duplicate_timestamps = df['start_time'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append(f"{filename}: {duplicate_timestamps} duplicate timestamps found")
        
        return issues
    
    def verify_data_completeness(self):
        """Verify data completeness across all pairs and timeframes"""
        logger.info("Verifying data completeness...")
        issues = []
        
        for pair in self.get_available_pairs():
            for timeframe in self.get_available_timeframes(pair):
                completeness_issues = self.verify_timeframe_completeness(pair, timeframe)
                issues.extend(completeness_issues)
        
        return issues
    
    def verify_timeframe_completeness(self, pair, timeframe):
        """Verify completeness for a specific pair and timeframe"""
        issues = []
        parquet_files = self.get_parquet_files(pair, timeframe)
        
        if not parquet_files:
            return issues
        
        # Load all data for the timeframe
        all_data = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                all_data.append(df)
            except Exception as e:
                issues.append(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            return issues
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('start_time').reset_index(drop=True)
        
        # Check for gaps in data
        if len(combined_df) > 1:
            time_diff = combined_df['start_time'].diff().dropna()
            
            # Calculate expected interval based on timeframe
            interval_minutes = {
                '5m': 5, '15m': 15, '30m': 30, 
                '1h': 60, '4h': 240, '1d': 1440
            }
            expected_interval = pd.Timedelta(minutes=interval_minutes.get(timeframe, 60))
            
            # Find gaps larger than expected interval
            large_gaps = time_diff > expected_interval * 1.1  # 10% tolerance
            gap_count = large_gaps.sum()
            
            if gap_count > 0:
                issues.append(f"Pair {pair}, timeframe {timeframe}: {gap_count} gaps found in data")
                
                # Log some example gaps - fix indexing issue
                gap_indices = large_gaps[large_gaps].index
                if len(gap_indices) > 0:
                    gap_examples = combined_df.iloc[gap_indices[:3]]['start_time']
                    for timestamp in gap_examples:
                        issues.append(f"  Gap example at {timestamp}")
        
        return issues
    
    def generate_summary_report(self):
        """Generate a summary report of the verification"""
        logger.info("Generating summary report...")
        
        report = {
            'verification_timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'expected_pairs': len(self.expected_pairs),
            'available_pairs': len(self.get_available_pairs()),
            'expected_timeframes': len(self.expected_timeframes),
            'total_issues_found': 0,
            'issues_by_category': {}
        }
        
        # Count issues by category
        for category, issues in self.verification_results.items():
            report['issues_by_category'][category] = len(issues)
            report['total_issues_found'] += len(issues)
        
        return report
    
    def run_full_verification(self):
        """Run complete verification process"""
        logger.info("Starting full data verification...")
        
        # Load expected pairs
        if not self.load_expected_pairs():
            logger.error("Failed to load expected pairs. Aborting verification.")
            return False
        
        # Run all verification checks
        self.verification_results = {
            'file_structure': self.verify_file_structure(),
            'data_quality': [],
            'data_completeness': self.verify_data_completeness()
        }
        
        # Verify data quality for each pair and timeframe
        for pair in self.get_available_pairs():
            for timeframe in self.get_available_timeframes(pair):
                quality_issues = self.verify_data_quality(pair, timeframe)
                self.verification_results['data_quality'].extend(quality_issues)
        
        # Generate and display summary
        summary = self.generate_summary_report()
        
        logger.info("=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Verification completed at: {summary['verification_timestamp']}")
        logger.info(f"Data directory: {summary['data_directory']}")
        logger.info(f"Expected pairs: {summary['expected_pairs']}")
        logger.info(f"Available pairs: {summary['available_pairs']}")
        logger.info(f"Total issues found: {summary['total_issues_found']}")
        logger.info("")
        
        # Display issues by category
        for category, issues in self.verification_results.items():
            if issues:
                logger.info(f"{category.upper()} ISSUES ({len(issues)}):")
                for issue in issues[:10]:  # Show first 10 issues
                    logger.info(f"  - {issue}")
                if len(issues) > 10:
                    logger.info(f"  ... and {len(issues) - 10} more issues")
                logger.info("")
        
        if summary['total_issues_found'] == 0:
            logger.info("✅ All verifications passed! Data appears to be complete and valid.")
        else:
            logger.warning(f"⚠️  Found {summary['total_issues_found']} issues that need attention.")
        
        return summary['total_issues_found'] == 0

def main():
    """Main function to run data verification"""
    verifier = DataVerifier()
    success = verifier.run_full_verification()
    
    if success:
        logger.info("Data verification completed successfully!")
        exit(0)
    else:
        logger.error("Data verification found issues that need to be addressed.")
        exit(1)

if __name__ == "__main__":
    main()
