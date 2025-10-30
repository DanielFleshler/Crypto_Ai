# Crypto Trading Bot - Elliott Wave & ICT Strategy

A sophisticated cryptocurrency trading bot implementing Elliott Wave Theory and Inner Circle Trader (ICT) concepts for automated trading.

## 🎯 Overview

This trading system combines:
- **Elliott Wave Analysis**: Identifies market cycles and wave patterns
- **ICT Concepts**: Fair Value Gaps (FVG), Order Blocks, Liquidity Sweeps
- **Market Structure**: Break of Structure (BOS), Change of Character (ChoCh)
- **Kill Zones**: Session-based entry timing (London, New York)
- **Multi-Timeframe Analysis**: HTF bias with LTF precision entries

## 📁 Project Structure

```
crypto_bot_trader/
├── trading_strategy/          # Core trading logic
│   ├── trading_strategy.py    # Main strategy orchestrator
│   ├── elliott_wave.py        # Elliott Wave detection
│   ├── ict_concepts.py        # ICT concept detection
│   ├── ict_entries.py         # Entry signal generation
│   ├── market_structure.py    # Market structure analysis
│   ├── ltf_precision_entry.py # Lower timeframe entries
│   ├── kill_zones.py          # Session detection
│   ├── data_loader.py         # Data loading and caching
│   ├── data_structures.py     # Core data models
│   └── config_loader.py       # Configuration management
│
├── config/                    # Configuration files
│   ├── trading_config.yaml    # Strategy parameters
│   ├── pairs.yaml             # Trading pairs
│   └── timeframes.yaml        # Timeframe settings
│
├── scripts/                   # Utility scripts
│   ├── backtest/              # Backtesting scripts
│   ├── analysis/              # Analysis tools
│   └── diagnostics/           # Diagnostic utilities
│
├── tests/                     # Test suite
│   ├── test_trading_strategy.py
│   ├── test_backtester.py
│   ├── test_ict_concepts.py
│   └── ...
│
├── data/                      # Market data
│   └── raw/                   # Raw OHLCV data
│       ├── BTCUSDT/
│       ├── ETHUSDT/
│       └── ...
│
├── docs/                      # Documentation
│   ├── strategy_improvement_plan.md
│   ├── performance_optimization.md
│   └── ...
│
├── results/                   # Results and reports
│   └── backtests/             # Backtest results
│
├── backtester.py              # Core backtest engine
├── backtest.py                # Unified backtest interface
├── setup.py                   # Package setup
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto_bot_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running a Backtest

```bash
# Quick validation backtest
python backtest.py quick --pair BTCUSDT --start 2023-01-01 --end 2023-03-31

# Single period backtest
python backtest.py single --pair BTCUSDT --start 2023-01-01 --end 2023-12-31

# Multi-period comprehensive backtest
python backtest.py multi --pair BTCUSDT

# Walk-forward analysis
python backtest.py walk-forward --pair BTCUSDT --start 2021-01-01 --end 2024-10-18

# Parameter optimization
python backtest.py optimize --pair BTCUSDT --start 2023-01-01 --end 2023-12-31
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trading_strategy.py

# Run with coverage
pytest --cov=trading_strategy --cov-report=html

# Run tests in parallel
pytest -n auto
```

## 📊 Strategy Components

### 1. Elliott Wave Analysis
- Identifies 5-wave impulse patterns (12345)
- Detects 3-wave corrective patterns (ABC)
- Validates wave relationships using Fibonacci ratios
- Ranks wave quality and confidence

### 2. ICT Concepts
- **Fair Value Gaps (FVG)**: Price imbalances that attract price
- **Order Blocks**: Institutional accumulation/distribution zones
- **Liquidity Sweeps**: Stop-loss raids before reversals
- **Breaker Blocks**: Failed support/resistance flips

### 3. Market Structure
- **Break of Structure (BOS)**: Continuation signals
- **Change of Character (ChoCh)**: Reversal signals
- Swing high/low detection
- Trend identification

### 4. Entry System
- **HTF Bias**: Higher timeframe trend direction (4h)
- **MTF Confluence**: Mid timeframe confirmation (1h)
- **LTF Precision**: Lower timeframe entry (15m)
- Multi-confirmation requirement

### 5. Risk Management
- Fixed percentage risk per trade (1-2%)
- Volatility-adjusted position sizing
- Maximum daily risk limits (5%)
- Maximum concurrent positions (3)
- Correlation-based position limits
- Stop-loss cooldown periods

## ⚙️ Configuration

### Main Configuration (`config/trading_config.yaml`)

```yaml
# Risk Management
risk_management:
  max_risk_per_trade: 0.02           # 2% risk per trade
  max_daily_risk: 0.05               # 5% maximum daily risk
  max_concurrent_positions: 3        # Maximum open positions
  minimum_rr_ratio: 2.0              # Minimum risk-reward ratio

# Entry Confirmation
entry_confirmation:
  minimum_confirmations: 2           # Required confirmations
  minimum_score: 0.5                 # Minimum signal quality

# HTF Bias
htf_bias:
  required_strength: 0.5             # Bias strength threshold
  allow_neutral: false               # Trade in neutral bias

# Session Filtering
session_filtering:
  filter_by_session: true            # Enable session filtering
  enabled_sessions:
    - london
    - new_york
```

## 📈 Performance Metrics

The backtester calculates comprehensive performance metrics:

- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Expectancy**: Expected value per trade
- **Recovery Factor**: Net profit / max drawdown
- **Calmar Ratio**: Return / max drawdown

## 🔧 Development

### Code Quality

```bash
# Format code
black trading_strategy/ tests/

# Sort imports
isort trading_strategy/ tests/

# Lint code
flake8 trading_strategy/ tests/

# Type checking
mypy trading_strategy/
```

### Adding a New Strategy Component

1. Create module in `trading_strategy/`
2. Add corresponding tests in `tests/`
3. Update configuration in `config/trading_config.yaml`
4. Document in appropriate markdown file

### Running Analysis Scripts

```bash
# Diagnose signal generation issues
python scripts/diagnostics/diagnose_signal_bias.py

# Analyze strategy performance
python scripts/diagnostics/strategy_diagnostic.py

# Risk analysis
python scripts/analysis/risk_analysis.py

# Comprehensive analysis
python scripts/analysis/analysis.py --mode diagnose --pair BTCUSDT
```

## 📚 Documentation

- **[Strategy Improvement Plan](docs/strategy_improvement_plan.md)**: Current performance and improvement roadmap
- **[Performance Optimization](docs/performance_optimization.md)**: Performance tuning guidelines
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)**: Implementation details
- **[Quick Start Guide](docs/QUICK_START.md)**: Getting started tutorial

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Strategy Tests**: End-to-end strategy testing
- **Edge Case Tests**: Boundary condition testing
- **Performance Tests**: Optimization validation

Run specific test suites:
```bash
pytest tests/test_ict_concepts.py       # ICT concept tests
pytest tests/test_elliott_wave.py       # Elliott Wave tests
pytest tests/test_market_structure.py   # Market structure tests
pytest tests/test_risk_management.py    # Risk management tests
pytest tests/test_backtester.py         # Backtester tests
```

## 🎯 Current Performance

Based on recent comprehensive backtesting:

| Metric | Value | Target |
|--------|-------|--------|
| Win Rate | 2.27% | >40% |
| Profit Factor | 0.23 | >1.5 |
| Sharpe Ratio | -24.29 | >1.0 |
| Max Drawdown | TBD | <15% |
| Total Trades | 19 (12 periods) | 100-200+ |

**Status**: Under active development and optimization. See [Strategy Improvement Plan](docs/strategy_improvement_plan.md) for details.

## 🔄 Recent Improvements

- ✅ Reorganized project structure for better maintainability
- ✅ Consolidated redundant scripts
- ✅ Enhanced documentation and code comments
- ✅ Improved test coverage
- ✅ Added comprehensive backtesting suite
- ✅ Implemented advanced risk management

## 🚧 Roadmap

### Phase 1: Signal Generation (Current)
- [ ] Loosen entry confirmation requirements
- [ ] Adjust HTF bias filtering
- [ ] Implement counter-trend FVG trading
- [ ] Reduce minimum R:R ratio

### Phase 2: Entry Diversification
- [ ] Add liquidity sweep reversals
- [ ] Implement order block rejections
- [ ] Add session breakout entries
- [ ] FVG + momentum combination

### Phase 3: Advanced Features
- [ ] Dynamic position sizing based on signal quality
- [ ] Machine learning signal ranking
- [ ] Multi-pair portfolio optimization
- [ ] Real-time execution system

## 📝 License

[Specify your license here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## ⚠️ Disclaimer

This is educational software for backtesting purposes only. Trading cryptocurrencies involves substantial risk of loss. This software is provided "as-is" without any warranty. Use at your own risk.

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review test files for usage examples

---

**Built with ❤️ for algorithmic trading excellence**

