# Crypto Bot Trader

A sophisticated cryptocurrency trading bot that combines Elliott Wave Theory with ICT (Inner Circle Trading) concepts to identify high-probability trading opportunities.

## Overview

This trading system uses advanced technical analysis methods including:

- **Elliott Wave Analysis**: Pattern recognition and wave counting
- **ICT Concepts**: Order blocks, fair value gaps, liquidity zones, and kill zones
- **Multi-Timeframe Analysis**: Higher timeframe bias with lower timeframe precision entries
- **Regime-Adaptive Scoring**: Dynamic adjustment based on market conditions
- **Market Structure Detection**: Break of structure (BOS) and change of character (CHoCH) identification

## Features

- Multi-timeframe signal generation (HTF bias + LTF entries)
- Advanced risk management with dynamic position sizing
- Comprehensive backtesting engine with walk-forward analysis
- Production validation suite for pre-deployment testing
- Fibonacci-based stop loss and take profit levels
- Kill zone awareness (London, New York sessions)
- Wave ranking system for trade prioritization
- Multi-confirmation entry system

## Project Structure

```
.
├── backtest.py                      # Unified backtesting interface
├── backtester.py                    # Core backtesting engine
├── PRODUCTION_VALIDATION_SUITE.py   # Pre-production validation tests
├── config/                          # Configuration files
│   ├── pairs.yaml                   # Trading pairs configuration
│   ├── timeframes.yaml              # Timeframe settings
│   └── trading_config.yaml          # Main trading strategy config
├── trading_strategy/                # Core trading strategy modules
│   ├── config_loader.py             # Configuration management
│   ├── data_loader.py               # Data loading and caching
│   ├── data_structures.py           # Signal and trade data structures
│   ├── elliott_wave.py              # Elliott Wave detection
│   ├── ict_concepts.py              # ICT concepts implementation
│   ├── ict_entries.py               # ICT-based entry signals
│   ├── kill_zones.py                # Trading session detection
│   ├── ltf_precision_entry.py       # Lower timeframe entry logic
│   ├── market_structure.py          # Market structure analysis
│   ├── regime_adaptive_scoring.py   # Regime-based signal scoring
│   └── trading_strategy.py          # Main strategy orchestration
├── data/                            # Market data directory
├── results/                         # Backtest results output
└── requirements.txt                 # Python dependencies
```

## Configuration

The strategy is configured through YAML files in the `config/` directory:

### trading_config.yaml

Contains strategy parameters including:

- Elliott Wave settings
- ICT concept parameters
- Risk management rules
- Entry confirmation requirements
- Wave ranking criteria

### timeframes.yaml

Defines the multi-timeframe structure:

- Higher timeframe (HTF) for bias
- Medium timeframe (MTF) for structure
- Lower timeframe (LTF) for entries

### pairs.yaml

Lists the trading pairs to analyze

## Strategy Logic

### Signal Generation Process

1. **HTF Bias Determination**: Analyze higher timeframe for overall market direction
2. **Market Structure**: Identify key support/resistance levels and structure breaks
3. **Elliott Wave Analysis**: Detect wave patterns and count wave positions
4. **ICT Concepts**: Identify order blocks, fair value gaps, and liquidity zones
5. **Kill Zone Filter**: Check if current time is within optimal trading sessions
6. **LTF Entry Confirmation**: Look for precise entry triggers on lower timeframe
7. **Multi-Confirmation**: Require multiple confirmations before generating signal
8. **Wave Ranking**: Score and prioritize signals based on quality

### Risk Management

- Dynamic position sizing based on account balance
- Maximum risk per trade (configurable, typically 1-2%)
- Maximum daily risk limits
- Maximum concurrent positions
- Fibonacci-based stop loss placement
- Multiple take profit targets with partial exits

## Backtesting

The backtesting engine provides:

- Realistic execution modeling
- Slippage and commission simulation
- Multiple position management
- Partial profit taking
- Comprehensive performance metrics
- Trade-by-trade analysis
- Equity curve generation
- Drawdown analysis

### Backtest Metrics

- Total return and annualized return
- Win rate and profit factor
- Average win/loss ratio
- Maximum drawdown
- Sharpe ratio
- Recovery factor
- Trade statistics (count, duration, etc.)
