# 🚀 Quick Start Guide

## Running the Trading Bot - Simple Commands

### 1. Run Comprehensive Backtest
```bash
cd /Users/danielfleshler/Desktop/Code/Crypto_bot_trader
source venv/bin/activate
python realistic_backtest.py
```
**Output**: Detailed performance across 12 market periods, saved to JSON

### 2. Run Signal Diagnostics
```bash
python strategy_diagnostic.py
```
**Purpose**: Understand why signals are being generated or filtered

### 3. Run Analysis Suite
```bash
# Signal bias analysis
python analysis.py signal-bias --pair BTCUSDT --start 2023-01-01 --end 2023-12-31

# Risk analysis
python analysis.py risk-analysis --win-rate 0.35 --rr-ratio 2.0

# Market regime analysis  
python analysis.py market-regime --pair BTCUSDT --start 2024-01-01 --end 2024-10-18

# Comprehensive analysis
python analysis.py comprehensive --pair BTCUSDT
```

### 4. Run Unified Backtest (Multiple Modes)
```bash
# Quick validation
python backtest.py quick --pair BTCUSDT --start 2023-01-01 --end 2023-03-31

# Single period detailed
python backtest.py single --pair BTCUSDT --start 2023-01-01 --end 2023-12-31

# Multi-period comparison
python backtest.py multi --pair BTCUSDT

# Walk-forward analysis
python backtest.py walk-forward --pair BTCUSDT --start 2021-07-01 --end 2024-10-18

# Parameter optimization
python backtest.py optimize --pair BTCUSDT --start 2023-01-01 --end 2024-10-18
```

### 5. Performance Benchmark
```bash
python vectorized_backtest.py
```
**Purpose**: Benchmark backtesting speed (should be <10 seconds)

---

## 📝 Configuration

### Main Config File
```bash
nano config/trading_config.yaml
```

### Key Parameters to Adjust
```yaml
# For MORE signals (current setting)
entry_confirmation:
  min_confirmations: 1
  min_confirmation_score: 0.05

risk_management:
  minimum_rr_ratio: 1.5
  max_risk_per_trade: 0.02

sessions:
  filter_by_session: false

# For HIGHER QUALITY signals (if too many)
entry_confirmation:
  min_confirmations: 2
  min_confirmation_score: 0.3

risk_management:
  minimum_rr_ratio: 2.5

sessions:
  filter_by_session: true
```

---

## 🔍 Monitoring

### Check Cache Performance
```python
from trading_strategy.data_loader import DataLoader

loader = DataLoader(base_path='data/raw')
# ... use loader ...
stats = loader.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
```

### Clear Cache
```python
loader.clear_cache()
```

---

## 📊 Reading Results

### Backtest Results File
```bash
cat realistic_backtest_results_*.json | python -m json.tool | less
```

### Key Metrics to Watch
- **Win Rate**: > 35% is good
- **Profit Factor**: > 1.5 is profitable
- **Sharpe Ratio**: > 1.0 is excellent
- **Max Drawdown**: < 15% is ideal
- **Total Trades**: 50-100 per month is healthy

---

## 🐛 Troubleshooting

### No Trades Generated
1. Check signal diagnostics: `python strategy_diagnostic.py`
2. Lower confirmation requirements in config
3. Disable session filtering
4. Check data availability

### Slow Performance
1. Run benchmark: `python vectorized_backtest.py`
2. Check cache hit rate
3. Reduce lookback periods
4. Use fewer timeframes

### Data Loading Issues
1. Verify data files exist: `ls data/raw/BTCUSDT/1h/`
2. Check date ranges (data starts from 2021-07)
3. Try different pair or timeframe
4. Clear cache and retry

---

## 🎯 Next Actions

### Today
- [x] Review backtest results
- [x] Adjust configuration if needed
- [ ] Run new backtest
- [ ] Verify signal count improved

### This Week
- [ ] Set up paper trading account
- [ ] Connect to live data feed
- [ ] Monitor signal quality
- [ ] Collect 1 week of data

### This Month
- [ ] Validate strategy performance
- [ ] Optimize underperforming regimes
- [ ] Build monitoring dashboard
- [ ] Prepare for live trading

---

## 📞 Help

### Common Issues

**Issue**: "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

**Issue**: "Data not found"
- Data starts from 2021-07-01, adjust dates accordingly

**Issue**: "Too few signals"
- Lower `min_confirmations` to 1
- Reduce `min_confirmation_score` to 0.05
- Set `filter_by_session: false`

**Issue**: "Backtest too slow"
- Use vectorized backtest
- Enable caching
- Reduce data range

---

## 🎓 Learn More

- See `IMPLEMENTATION_SUMMARY.md` for full details
- See `strategy_improvement_plan.md` for optimization roadmap
- See `performance_optimization.md` for technical details

---

**Last Updated**: 2025-10-30
**Version**: 1.0.0
