#!/usr/bin/env python3
"""
Detailed Data Verification Report Generator
Creates comprehensive reports with specific recommendations
"""

import pandas as pd
import os
import glob
import yaml
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, data_dir="data/raw", config_path="config/pairs.yaml", output_dir="reports"):
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_dir = output_dir
        self.expected_pairs = []
        self.expected_timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_expected_pairs(self):
        """Load expected trading pairs from config file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.expected_pairs = [pair.replace('/', '') for pair in config['pairs']]
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def get_data_summary(self):
        """Generate summary statistics for all data"""
        logger.info("Generating data summary...")
        
        summary = {
            'total_pairs': 0,
            'total_files': 0,
            'total_records': 0,
            'date_range': {},
            'file_sizes': {},
            'pairs_detail': {}
        }
        
        pair_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d))]
        
        summary['total_pairs'] = len(pair_dirs)
        
        for pair in pair_dirs:
            pair_info = {
                'timeframes': {},
                'total_files': 0,
                'total_records': 0,
                'date_range': {'start': None, 'end': None}
            }
            
            pair_dir = os.path.join(self.data_dir, pair)
            timeframes = [d for d in os.listdir(pair_dir) 
                         if os.path.isdir(os.path.join(pair_dir, d))]
            
            for timeframe in timeframes:
                timeframe_info = {
                    'files': 0,
                    'records': 0,
                    'file_size_mb': 0,
                    'date_range': {'start': None, 'end': None}
                }
                
                pattern = os.path.join(self.data_dir, pair, timeframe, "*.parquet")
                parquet_files = glob.glob(pattern)
                timeframe_info['files'] = len(parquet_files)
                
                for file_path in parquet_files:
                    try:
                        # Get file size
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        timeframe_info['file_size_mb'] += file_size
                        
                        # Load data to get record count and date range
                        df = pd.read_parquet(file_path)
                        records = len(df)
                        timeframe_info['records'] += records
                        
                        if 'start_time' in df.columns and len(df) > 0:
                            file_start = df['start_time'].min()
                            file_end = df['start_time'].max()
                            
                            if timeframe_info['date_range']['start'] is None or file_start < timeframe_info['date_range']['start']:
                                timeframe_info['date_range']['start'] = file_start
                            if timeframe_info['date_range']['end'] is None or file_end > timeframe_info['date_range']['end']:
                                timeframe_info['date_range']['end'] = file_end
                    
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
                
                pair_info['timeframes'][timeframe] = timeframe_info
                pair_info['total_files'] += timeframe_info['files']
                pair_info['total_records'] += timeframe_info['records']
                
                # Update pair date range
                if timeframe_info['date_range']['start']:
                    if pair_info['date_range']['start'] is None or timeframe_info['date_range']['start'] < pair_info['date_range']['start']:
                        pair_info['date_range']['start'] = timeframe_info['date_range']['start']
                if timeframe_info['date_range']['end']:
                    if pair_info['date_range']['end'] is None or timeframe_info['date_range']['end'] > pair_info['date_range']['end']:
                        pair_info['date_range']['end'] = timeframe_info['date_range']['end']
            
            summary['pairs_detail'][pair] = pair_info
            summary['total_files'] += pair_info['total_files']
            summary['total_records'] += pair_info['total_records']
        
        return summary
    
    def analyze_data_quality_issues(self):
        """Analyze specific data quality issues"""
        logger.info("Analyzing data quality issues...")
        
        issues = {
            'duplicate_timestamps': {},
            'ordering_issues': {},
            'ohlc_issues': {},
            'negative_values': {},
            'missing_values': {},
            'gaps_in_data': {}
        }
        
        pair_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for pair in pair_dirs:
            pair_dir = os.path.join(self.data_dir, pair)
            timeframes = [d for d in os.listdir(pair_dir) 
                         if os.path.isdir(os.path.join(pair_dir, d))]
            
            for timeframe in timeframes:
                pattern = os.path.join(self.data_dir, pair, timeframe, "*.parquet")
                parquet_files = glob.glob(pattern)
                
                # Sample a few files for detailed analysis
                sample_files = parquet_files[:3] if len(parquet_files) > 3 else parquet_files
                
                for file_path in sample_files:
                    try:
                        df = pd.read_parquet(file_path)
                        if len(df) == 0:
                            continue
                        
                        filename = os.path.basename(file_path)
                        key = f"{pair}_{timeframe}"
                        
                        # Check for duplicate timestamps
                        if 'start_time' in df.columns:
                            duplicates = df['start_time'].duplicated().sum()
                            if duplicates > 0:
                                if key not in issues['duplicate_timestamps']:
                                    issues['duplicate_timestamps'][key] = []
                                issues['duplicate_timestamps'][key].append({
                                    'file': filename,
                                    'count': duplicates
                                })
                            
                            # Check timestamp ordering
                            if not df['start_time'].is_monotonic_increasing:
                                if key not in issues['ordering_issues']:
                                    issues['ordering_issues'][key] = []
                                issues['ordering_issues'][key].append(filename)
                        
                        # Check OHLC relationships
                        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                            invalid_ohlc = ((df['high'] < df['low']) | 
                                           (df['high'] < df['open']) | 
                                           (df['high'] < df['close']) |
                                           (df['low'] > df['open']) | 
                                           (df['low'] > df['close'])).sum()
                            if invalid_ohlc > 0:
                                if key not in issues['ohlc_issues']:
                                    issues['ohlc_issues'][key] = []
                                issues['ohlc_issues'][key].append({
                                    'file': filename,
                                    'count': invalid_ohlc
                                })
                        
                        # Check for negative values
                        price_columns = ['open', 'high', 'low', 'close']
                        for col in price_columns:
                            if col in df.columns:
                                negative_count = (df[col] <= 0).sum()
                                if negative_count > 0:
                                    if key not in issues['negative_values']:
                                        issues['negative_values'][key] = {}
                                    if col not in issues['negative_values'][key]:
                                        issues['negative_values'][key][col] = []
                                    issues['negative_values'][key][col].append({
                                        'file': filename,
                                        'count': negative_count
                                    })
                        
                        # Check for missing values
                        missing_counts = df.isnull().sum()
                        for col, count in missing_counts.items():
                            if count > 0:
                                if key not in issues['missing_values']:
                                    issues['missing_values'][key] = {}
                                if col not in issues['missing_values'][key]:
                                    issues['missing_values'][key][col] = []
                                issues['missing_values'][key][col].append({
                                    'file': filename,
                                    'count': count
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error analyzing {file_path}: {e}")
        
        return issues
    
    def generate_recommendations(self, summary, issues):
        """Generate specific recommendations based on analysis"""
        recommendations = []
        
        # File structure recommendations
        if summary['total_pairs'] != len(self.expected_pairs):
            recommendations.append({
                'category': 'File Structure',
                'priority': 'High',
                'issue': f"Expected {len(self.expected_pairs)} pairs, found {summary['total_pairs']}",
                'recommendation': "Verify that all expected trading pairs are present in the data directory"
            })
        
        # Data quality recommendations
        total_duplicates = sum(len(files) for files in issues['duplicate_timestamps'].values())
        if total_duplicates > 0:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'issue': f"Found {total_duplicates} files with duplicate timestamps",
                'recommendation': "Run the repair_data.py script to remove duplicate timestamps"
            })
        
        total_ordering = sum(len(files) for files in issues['ordering_issues'].values())
        if total_ordering > 0:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'issue': f"Found {total_ordering} files with timestamp ordering issues",
                'recommendation': "Run the repair_data.py script to fix timestamp ordering"
            })
        
        total_ohlc = sum(len(files) for files in issues['ohlc_issues'].values())
        if total_ohlc > 0:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'Medium',
                'issue': f"Found {total_ohlc} files with invalid OHLC relationships",
                'recommendation': "Run the repair_data.py script to fix OHLC relationships"
            })
        
        # Performance recommendations
        large_files = []
        for pair, pair_info in summary['pairs_detail'].items():
            for timeframe, tf_info in pair_info['timeframes'].items():
                if tf_info['file_size_mb'] > 100:  # Files larger than 100MB
                    large_files.append(f"{pair}_{timeframe}: {tf_info['file_size_mb']:.1f}MB")
        
        if large_files:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Low',
                'issue': f"Found {len(large_files)} large files that may impact performance",
                'recommendation': "Consider splitting large files into smaller chunks for better performance"
            })
        
        return recommendations
    
    def generate_html_report(self, summary, issues, recommendations):
        """Generate an HTML report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Trading Data Verification Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .issue {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .recommendation {{ background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-left: 4px solid #17a2b8; }}
        .summary {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .priority-high {{ color: #dc3545; font-weight: bold; }}
        .priority-medium {{ color: #fd7e14; font-weight: bold; }}
        .priority-low {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Crypto Trading Data Verification Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Data Summary</h2>
        <div class="summary">
            <p><strong>Total Pairs:</strong> {summary['total_pairs']}</p>
            <p><strong>Total Files:</strong> {summary['total_files']}</p>
            <p><strong>Total Records:</strong> {summary['total_records']:,}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Issues Found</h2>
        <h3>Duplicate Timestamps</h3>
        {self._format_issues_html(issues['duplicate_timestamps'])}
        
        <h3>Timestamp Ordering Issues</h3>
        {self._format_issues_html(issues['ordering_issues'])}
        
        <h3>OHLC Relationship Issues</h3>
        {self._format_issues_html(issues['ohlc_issues'])}
        
        <h3>Negative Values</h3>
        {self._format_issues_html(issues['negative_values'])}
        
        <h3>Missing Values</h3>
        {self._format_issues_html(issues['missing_values'])}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {self._format_recommendations_html(recommendations)}
    </div>
</body>
</html>
        """
        
        report_path = os.path.join(self.output_dir, "verification_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
        return report_path
    
    def _format_issues_html(self, issues):
        """Format issues for HTML display"""
        if not issues:
            return "<p>No issues found</p>"
        
        html = "<ul>"
        for key, value in issues.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html
    
    def _format_recommendations_html(self, recommendations):
        """Format recommendations for HTML display"""
        if not recommendations:
            return "<p>No recommendations</p>"
        
        html = ""
        for rec in recommendations:
            priority_class = f"priority-{rec['priority'].lower()}"
            html += f"""
            <div class="recommendation">
                <h4 class="{priority_class}">{rec['category']} - {rec['priority']} Priority</h4>
                <p><strong>Issue:</strong> {rec['issue']}</p>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
            </div>
            """
        return html
    
    def generate_json_report(self, summary, issues, recommendations):
        """Generate a JSON report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'issues': issues,
            'recommendations': recommendations
        }
        
        report_path = os.path.join(self.output_dir, "verification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {report_path}")
        return report_path
    
    def generate_all_reports(self):
        """Generate all types of reports"""
        logger.info("Generating comprehensive reports...")
        
        # Load expected pairs
        if not self.load_expected_pairs():
            logger.error("Failed to load expected pairs")
            return False
        
        # Generate summary
        summary = self.get_data_summary()
        
        # Analyze issues
        issues = self.analyze_data_quality_issues()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(summary, issues)
        
        # Generate reports
        html_path = self.generate_html_report(summary, issues, recommendations)
        json_path = self.generate_json_report(summary, issues, recommendations)
        
        logger.info("=" * 60)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"HTML Report: {html_path}")
        logger.info(f"JSON Report: {json_path}")
        logger.info(f"Total Issues Found: {sum(len(v) if isinstance(v, dict) else 0 for v in issues.values())}")
        logger.info(f"Recommendations: {len(recommendations)}")
        
        return True

def main():
    """Main function to generate reports"""
    generator = ReportGenerator()
    success = generator.generate_all_reports()
    
    if success:
        logger.info("✅ Report generation completed successfully!")
    else:
        logger.error("❌ Report generation failed. Check logs for details.")

if __name__ == "__main__":
    main()
