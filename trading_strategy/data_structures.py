from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    signal_type: str  # 'BUY' or 'SELL'
    entry_type: str  # Type of entry signal
    price: float
    stop_loss: float
    take_profits: List[float]
    risk_reward: float
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ElliottWave:
    """Elliott Wave data structure"""
    wave_number: int
    wave_type: str  # 'IMPULSE' or 'CORRECTIVE'
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    status: str  # 'current', 'completed', etc.
    fibonacci_levels: Dict[str, float]
    
    def __post_init__(self):
        if self.fibonacci_levels is None:
            self.fibonacci_levels = {}

@dataclass
class ICTConcept:
    """ICT (Inner Circle Trader) concept data structure"""
    timestamp: datetime
    concept_type: str  # 'FVG_BULLISH', 'FVG_BEARISH', 'OB_BULLISH', 'OB_BEARISH', etc.
    start_price: float
    end_price: float
    status: str  # 'current', 'broken', etc.
    strength: float  # 0.0 to 1.0
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.5

@dataclass
class MarketStructure:
    """Market structure data structure"""
    timestamp: datetime
    structure_type: str  # 'BOS', 'CHoCH', etc.
    price: float
    timeframe: str
    strength: float
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.5

@dataclass
class SwingPoint:
    """Swing point data structure"""
    timestamp: datetime
    point_type: str  # 'HIGH' or 'LOW'
    price: float
    strength: int  # Number of candles confirming the swing
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 1

@dataclass
class OrderBlock:
    """Order block data structure"""
    timestamp: datetime
    block_type: str  # 'BULLISH' or 'BEARISH'
    start_price: float
    end_price: float
    volume: float
    strength: float
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.5

@dataclass
class FairValueGap:
    """Fair Value Gap data structure"""
    timestamp: datetime
    gap_type: str  # 'BULLISH' or 'BEARISH'
    start_price: float
    end_price: float
    gap_size: float
    strength: float
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.5

@dataclass
class KillZone:
    """Kill zone data structure"""
    timestamp: datetime
    zone_type: str  # 'ASIA', 'LONDON', 'NEW_YORK', 'LONDON_NY_OVERLAP'
    is_active: bool
    strength: float
    
    def __post_init__(self):
        if self.strength is None:
            self.strength = 0.5

@dataclass
class TradingSession:
    """Trading session data structure"""
    session_name: str
    start_time: datetime
    end_time: datetime
    timezone: str
    is_active: bool
    
    def __post_init__(self):
        if self.is_active is None:
            self.is_active = False

@dataclass
class BacktestResult:
    """Backtest result data structure"""
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: List[float]
    trade_journal: pd.DataFrame
    
    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trade_journal is None:
            self.trade_journal = pd.DataFrame()

@dataclass
class TradingPair:
    """Trading pair data structure"""
    symbol: str
    base_currency: str
    quote_currency: str
    is_active: bool
    min_trade_size: float
    tick_size: float
    
    def __post_init__(self):
        if self.is_active is None:
            self.is_active = True
        if self.min_trade_size is None:
            self.min_trade_size = 0.0
        if self.tick_size is None:
            self.tick_size = 0.01

@dataclass
class RiskManagement:
    """Risk management data structure"""
    max_risk_per_trade: float
    max_daily_risk: float
    max_positions: int
    stop_loss_pips: float
    take_profit_pips: float
    
    def __post_init__(self):
        if self.max_risk_per_trade is None:
            self.max_risk_per_trade = 0.01
        if self.max_daily_risk is None:
            self.max_daily_risk = 0.05
        if self.max_positions is None:
            self.max_positions = 5
        if self.stop_loss_pips is None:
            self.stop_loss_pips = 20.0
        if self.take_profit_pips is None:
            self.take_profit_pips = 40.0
