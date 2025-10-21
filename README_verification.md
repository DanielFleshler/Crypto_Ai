# Data Quality Verification Toolkit

This toolkit provides comprehensive tools for verifying and repairing crypto trading data downloaded from Bybit.

## Tools Overview

### 1. `verify_data.py` - Data Verification
- **Purpose**: Comprehensive data quality verification
- **Features**:
  - File structure validation
  - Data quality checks (duplicates, ordering, OHLC relationships)
  - Completeness analysis (gaps in data)
  - Summary reporting

**Usage**:
```bash
python verify_data.py
```

### 2. `repair_data.py` - Data Repair
- **Purpose**: Fix common data quality issues
- **Features**:
  - Automatic backup creation before repairs
  - Duplicate timestamp removal
  - Timestamp ordering fixes
  - OHLC relationship corrections
  - Negative value handling
  - Missing value interpolation

**Usage**:
```bash
python repair_data.py
```

### 3. `generate_report.py` - Report Generation
- **Purpose**: Create detailed HTML and JSON reports
- **Features**:
  - Data summary statistics
  - Issue analysis and categorization
  - Specific recommendations
  - Visual HTML reports

**Usage**:
```bash
python generate_report.py
```

### 4. `data_quality_toolkit.py` - Interactive Toolkit
- **Purpose**: User-friendly interface for all tools
- **Features**:
  - Menu-driven interface
  - Full pipeline execution
  - Progress tracking

**Usage**:
```bash
python data_quality_toolkit.py
```

## Quick Start

### Option 1: Interactive Toolkit (Recommended)
```bash
python data_quality_toolkit.py
```
Follow the menu prompts to verify, repair, and generate reports.

### Option 2: Manual Pipeline
```bash
# 1. Verify data quality
python verify_data.py

# 2. Repair issues (creates backup automatically)
python repair_data.py

# 3. Generate reports
python generate_report.py
```

## What Gets Verified

### File Structure
- ✅ All expected trading pairs present
- ✅ All expected timeframes available
- ✅ Proper file naming conventions
- ✅ Directory structure integrity

### Data Quality
- ✅ Duplicate timestamps detection
- ✅ Timestamp ordering validation
- ✅ OHLC relationship verification
- ✅ Negative value detection
- ✅ Missing value identification
- ✅ Data type validation

### Data Completeness
- ✅ Gap detection in time series
- ✅ Date range coverage analysis
- ✅ Record count validation

## What Gets Repaired

### Automatic Fixes
- 🔧 Remove duplicate timestamps (keeps last occurrence)
- 🔧 Sort data by timestamp
- 🔧 Fix invalid OHLC relationships
- 🔧 Handle negative prices/volumes
- 🔧 Forward-fill missing values

### Safety Features
- 🔒 Automatic backup creation before repairs
- 🔒 Detailed logging of all changes
- 🔒 Rollback capability (restore from backup)

## Output Files

### Verification Results
- Console output with detailed issue reports
- Exit codes: 0 (success), 1 (issues found)

### Repair Results
- Backup created in `data/backup/` directory
- Original files modified in place
- Detailed repair logs

### Reports
- `reports/verification_report.html` - Visual HTML report
- `reports/verification_report.json` - Machine-readable JSON report

## Common Issues and Solutions

### Issue: Duplicate Timestamps
**Cause**: API returned duplicate records
**Solution**: Run `repair_data.py` to remove duplicates

### Issue: Out-of-Order Timestamps
**Cause**: Data downloaded in chunks and not properly sorted
**Solution**: Run `repair_data.py` to sort by timestamp

### Issue: Invalid OHLC Relationships
**Cause**: Data corruption or API inconsistencies
**Solution**: Run `repair_data.py` to fix relationships

### Issue: Gaps in Data
**Cause**: Market closures, API downtime, or download failures
**Solution**: Re-run download script for specific date ranges

## Best Practices

1. **Always verify before repair**: Run verification first to understand issues
2. **Use backups**: Repair script creates automatic backups
3. **Check reports**: Review HTML reports for detailed analysis
4. **Monitor logs**: Watch console output for detailed information
5. **Test on subset**: For large datasets, test on a small subset first

## Troubleshooting

### Permission Errors
```bash
chmod +x *.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Large Dataset Performance
- Use the interactive toolkit for better progress tracking
- Consider running verification on specific pairs first
- Monitor disk space during repair operations

## File Structure

```
Crypto_bot_trader/
├── data/
│   ├── raw/                    # Original data
│   └── backup/                 # Automatic backups
├── reports/                    # Generated reports
├── verify_data.py             # Verification tool
├── repair_data.py             # Repair tool
├── generate_report.py         # Report generator
├── data_quality_toolkit.py   # Interactive toolkit
└── README_verification.md     # This file
```

## Support

For issues or questions:
1. Check the console logs for detailed error messages
2. Review the generated HTML reports for analysis
3. Verify that all dependencies are installed
4. Ensure sufficient disk space for backups and repairs
