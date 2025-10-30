# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced cryptocurrency trading bot that generates trading signals by combining **Elliott Wave Theory** with **ICT (Inner Circle Trader) concepts**. The system uses multi-timeframe analysis to identify high-probability trade entries with institutional-level liquidity analysis.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_elliott_wave.py

# Run tests with specific markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m fast          # Fast tests only
pytest -m "not slow"    # Exclude slow tests

# Run single test function
pytest tests/test_elliott_wave.py::TestElliottWave::test_wave_1_detection

# Verbose output with full traceback
pytest -vv --tb=long
```

### Python Version
- Project uses Python 3.13.3
- Use `python3` command (not `python`)

## Architecture Overview

### Signal Generation Pipeline

The bot operates in a hierarchical multi-timeframe approach:

```
HTF (1h) → Establishes market bias (BULLISH/BEARISH/NEUTRAL)
    ↓
MTF (15m) → Generates trading signals (Elliott Wave + ICT patterns)
    ↓
LTF (5m) → Refines entry timing and tightens stop-loss
```

**Key principle**: Signals are ONLY generated when aligned with HTF bias. If HTF is BULLISH, only BUY signals are valid. If BEARISH, only SELL signals.

### Core Components

#### 1. TradingStrategy (`trading_strategy/trading_strategy.py`)
The orchestrator that:
- Manages HTF bias detection and dynamic updates
- Coordinates MTF analysis (Elliott Wave + ICT concepts)
- Applies multi-confirmation filtering
- Tracks bias history and invalidates misaligned signals on bias flips

#### 2. Elliott Wave Detection (`trading_strategy/elliott_wave.py`)
Implements strict Elliott Wave rules:
- **Wave 1**: Requires BOS/structure confirmation
- **Wave 2**: 23.6%-78.6% retracement, CANNOT exceed Wave 1 start (invalidation rule)
- **Wave 3**: Must break Wave 1 extreme, cannot be shortest wave
- **Wave 4**: 23.6%-50% retracement, CANNOT enter Wave 1 territory
- **Wave 5**: Extension with momentum divergence detection
- **ABC Corrections**: Detected after completed 5-wave impulse

**Critical**: Bidirectional symmetry for Fibonacci calculations - bullish and bearish waves use symmetric reflection logic.

#### 3. ICT Concepts (`trading_strategy/ict_concepts.py`)
Detects institutional trading patterns:
- **FVG (Fair Value Gaps)**: 3-candle price imbalances
- **Order Blocks**: Last opposite candle before strong moves
- **Breaker Blocks**: Failed order blocks that reverse
- **OTE Zones**: 62%-79% Fibonacci retracements (filtered by HTF bias)
- **Liquidity Grabs**: Sweeps of swing highs/lows followed by reversals

**Status tracking**: FVGs track fill status, Order Blocks track freshness.

#### 4. Market Structure (`trading_strategy/market_structure.py`)
Foundational structure detection:
- **Swing Points**: Configurable lookback periods (default: 5 left, 5 right)
- **BOS (Break of Structure)**: Trend continuation confirmation
- **CHoCH (Change of Character)**: Potential reversal signal
- **Bias Determination**: BULLISH (higher highs/lows), BEARISH (lower highs/lows), NEUTRAL (mixed)

#### 5. Kill Zones (`trading_strategy/kill_zones.py`)
Session-aware trading with DST handling:
- **Asia** (0-8 UTC): Low volatility, liquidity building, confidence weight 0.4
- **London** (8-13 UTC): High volatility, breakouts (watch for fakeouts), weight 0.8
- **NY** (13-21 UTC): High volatility, strongest moves, weight 0.9
- **London-NY Overlap** (13-16 UTC): Maximum liquidity/volatility, weight 1.0

**Key feature**: Dynamic position sizing and stop-loss adjustment based on session volatility profiles.

#### 6. ICT Entry Types (`trading_strategy/ict_entries.py`)
Five distinct entry strategies (all require HTF alignment):
1. **Liquidity Grab + CHoCH**: Sweep + reversal confirmation
2. **FVG Entry**: Retracement into unfilled FVG zones
3. **Order Block Entry**: Fresh OB with pattern confluence
4. **OTE Entry**: 62-79% retracement with OB/FVG confluence
5. **Breaker Block Entry**: Failed OB reversal

**All entries use structure-based stops/TPs** (not fixed percentages). Minimum 3:1 risk-reward required.

#### 7. LTF Precision Entry (`trading_strategy/ltf_precision_entry.py`)
Micro-level refinement:
- Detects micro FVGs, OBs, OTE zones, CHoCH on 1m/5m timeframes
- Gates MTF signals - only passes signals with LTF confirmation
- Calculates tighter stop-loss using LTF structure + ATR
- Improves average risk-reward ratios

#### 8. Backtester (`backtester.py`)
Position and risk management:
- **Position sizing**: Volatility-adjusted (not fixed distance)
- **Partial exits**: Multi-TP system (TP1, TP2, TP3)
- **Risk limits**: 2% per trade, 6% daily, 15% max drawdown
- **Correlation checks**: Avoids over-concentrated exposure
- **Stop-loss cooldowns**: Prevents revenge trading

### Configuration System

All parameters externalized in `config/`:
- **trading_config.yaml**: Elliott Wave rules, ICT parameters, risk management, session times
- **pairs.yaml**: Trading pairs (BTC/USDT, ETH/USDT, SOL/USDT, etc.)
- **timeframes.yaml**: HTF/MTF/LTF definitions

**ConfigLoader** (`trading_strategy/config_loader.py`) provides strongly-typed access via dataclasses.

### Data Structures (`trading_strategy/data_structures.py`)

Core dataclasses with validation:
- **Signal**: Trading signal with entry, SL, multiple TPs, confidence, metadata
- **ICTConcept**: Base class for FVGs, OBs, OTE zones, etc.
- **ElliottWave**: Wave with price levels, Fibonacci zones, status
- **MarketStructure**: BOS/CHoCH with confirmation details
- **Position**: Trade tracking with partial exits
- **LiquidityLevel**: Swing high/low tracking for liquidity analysis

## Key Implementation Details

### HTF Bias Management

The system maintains strict HTF bias discipline:

```python
# In TradingStrategy.generate_signals()
htf_bias = htf_analysis.get('bias', 'NEUTRAL')

# Only generate signals aligned with bias
if htf_bias == 'BULLISH':
    # Only BUY signals
elif htf_bias == 'BEARISH':
    # Only SELL signals
else:
    # No signals in NEUTRAL bias
```

**Bias flips**: When HTF bias changes (BULLISH → BEARISH or vice versa), existing signals are invalidated via `invalidate_signals_on_bias_flip()`.

### Multi-Confirmation System

Signals require multiple confluences from `entry_confirmation_config`:
```yaml
entry_confirmation:
  required_confirmations: 2
  confirmation_weights:
    fvg: 0.3
    order_block: 0.4
    ote: 0.3
    liquidity_grab: 0.5
    choch: 0.4
    bos: 0.3
```

Signals passing threshold are kept; others filtered out.

### Structure-Based Stop-Loss Logic

Stops are NOT fixed percentages. Logic:
```python
def _calculate_structure_based_stop_loss(entry_price, signal_type, swing_points, fvgs, order_blocks, structures):
    if signal_type == 'BUY':
        # Find nearest support: swing lows, OB bottoms, FVG bottoms
        # Place SL below strongest support with small buffer
    else:  # SELL
        # Find nearest resistance: swing highs, OB tops, FVG tops
        # Place SL above strongest resistance with small buffer
```

### Bug Fixes Implemented

The codebase contains fixes for identified bugs:
- **BUG-EW-001**: Wave 2 invalidation enforcement
- **BUG-EW-002**: Wave 2 bearish Fibonacci comparison (was inverted)
- **BUG-EW-003**: Wave 3 validation with strict break rules
- **BUG-EW-004**: Wave 3 cannot be shortest rule
- **BUG-EW-005**: Wave 1 BOS confirmation requirement
- **BUG-KZ-001**: DST-aware timezone handling
- **BUG-KZ-002**: London/NY overlap logic consistency
- **BUG-RISK-001**: Volatility-adjusted position sizing
- **BUG-RISK-002**: Multi-TP partial exit system
- **BUG-TS-001**: Fibonacci-based stop loss (was fixed 2%)
- **BUG-TS-002**: Fibonacci-based take profits (was multipliers)
- **BUG-TS-003**: Bidirectional reversal signals (was hardcoded SELL)
- **BUG-MTF-001**: HTF bias integration

When modifying Elliott Wave or ICT logic, ensure these bug fixes remain intact.

## Testing Strategy

Test files mirror module structure:
- `test_elliott_wave.py`: Wave detection and validation rules
- `test_ict_concepts.py`: FVG, OB, OTE, liquidity grab detection
- `test_market_structure.py`: Swing points, BOS, CHoCH
- `test_backtester.py`: Position management, risk controls
- `test_fibonacci.py`: Fibonacci calculation symmetry
- `test_bidirectional_symmetry_comprehensive.py`: Bullish/bearish pattern parity
- `test_edge_cases_comprehensive.py`: Edge case handling
- `test_dynamic_htf_bias.py`: HTF bias updates and signal invalidation

Tests use pytest markers (see `pytest.ini`): `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## Data Requirements

Market data expected in CSV format with columns:
- `timestamp` (datetime index)
- `open`, `high`, `low`, `close` (float, OHLC prices)
- `volume` (float)

Load via `DataLoader.load_data(pair, timeframe, start_date, end_date)`

## Common Pitfalls

1. **Don't generate signals without HTF bias**: Always check `htf_analysis['bias']` first
2. **Elliott Wave invalidation rules are strict**: Wave 2 and Wave 4 have hard boundaries
3. **ICT concepts require freshness tracking**: Order Blocks can become "unfresh" when retested
4. **Session-based filtering is critical**: Avoid Asia session entries (low quality)
5. **Risk-reward minimum is 3:1**: Signals below this threshold are rejected
6. **Fibonacci calculations are bidirectional**: Bullish and bearish use symmetric reflection
7. **Stop-loss must be structure-based**: Never use fixed percentage stops
8. **LTF refinement is optional but recommended**: It significantly improves entry quality

## Module Import Patterns

```python
# Standard imports for strategy work
from trading_strategy.trading_strategy import TradingStrategy
from trading_strategy.data_loader import DataLoader
from trading_strategy.config_loader import ConfigLoader
from trading_strategy.data_structures import Signal, ICTConcept, ElliottWave

# For backtesting
from backtester import BacktestEngine

# For specific detectors
from trading_strategy.elliott_wave import ElliottWaveDetector
from trading_strategy.ict_concepts import ICTConceptsDetector
from trading_strategy.market_structure import MarketStructureDetector
```

## Extending the System

When adding new features:
1. **New ICT concept**: Add to `ICTConceptsDetector`, create dataclass in `data_structures.py`
2. **New entry type**: Add to `ICTEntries`, implement structure-based SL/TP calculation
3. **New confirmation type**: Update `entry_confirmation_config` in YAML and multi-confirmation filtering
4. **New timeframe**: Update `timeframes.yaml`, ensure HTF/MTF/LTF hierarchy maintained
5. **New risk rule**: Add to `BacktestEngine`, update `risk_management_config` in YAML

Always maintain the principle: **HTF bias filters → MTF signals generate → LTF refines**.
