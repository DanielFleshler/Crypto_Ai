from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Trading signal data structure with validation."""
    timestamp: datetime
    signal_type: str  # 'BUY' or 'SELL'
    entry_type: str  # Type of entry signal
    price: float
    stop_loss: float
    take_profits: List[float]
    risk_reward: float
    confidence: float
    timeframe: Optional[str] = None  # Timeframe for the signal
    source: Optional[str] = None  # Source of the signal (e.g., 'elliott_wave', 'ict_concept')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data after initialization."""
        self._validate()

    def _validate(self):
        """Validate signal parameters."""
        if self.signal_type not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid signal_type: {self.signal_type}. Must be 'BUY' or 'SELL'")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

        if self.stop_loss <= 0:
            raise ValueError(f"Invalid stop_loss: {self.stop_loss}. Must be positive")

        if not self.take_profits:
            raise ValueError("take_profits cannot be empty")

        for tp in self.take_profits:
            if tp <= 0:
                raise ValueError(f"Invalid take_profit: {tp}. Must be positive")

        if self.risk_reward <= 0:
            raise ValueError(f"Invalid risk_reward: {self.risk_reward}. Must be positive")

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be between 0 and 1")

    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.signal_type == 'BUY'

    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.signal_type == 'SELL'

    def get_risk_amount(self, account_balance: float = None, risk_percent: float = None) -> float:
        """Calculate risk amount (absolute difference between entry and stop)."""
        if account_balance is not None and risk_percent is not None:
            return account_balance * risk_percent
        return abs(self.price - self.stop_loss)

    def get_reward_amount(self, account_balance: float = None, risk_percent: float = None, tp_index: int = 0) -> float:
        """Calculate reward amount for a specific take profit level."""
        if account_balance is not None and risk_percent is not None:
            risk_amount = account_balance * risk_percent
            return risk_amount * self.risk_reward

        if tp_index >= len(self.take_profits):
            raise IndexError(f"Take profit index {tp_index} out of range")

        tp = self.take_profits[tp_index]
        if self.is_bullish():
            return tp - self.price
        else:
            return self.price - tp

@dataclass
class LiquidityLevel:
    """Liquidity level data structure for tracking previous highs/lows."""
    timestamp: datetime
    level_type: str  # 'HIGH' or 'LOW'
    price: float
    strength: float
    is_swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    reversal_confirmed: bool = False

    def __post_init__(self):
        """Validate liquidity level data."""
        if self.level_type not in ['HIGH', 'LOW']:
            raise ValueError(f"Invalid level_type: {self.level_type}. Must be 'HIGH' or 'LOW'")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

    def mark_swept(self, sweep_timestamp: datetime):
        """Mark this liquidity level as swept."""
        self.is_swept = True
        self.sweep_timestamp = sweep_timestamp

    def confirm_reversal(self):
        """Confirm reversal after liquidity sweep."""
        if not self.is_swept:
            raise ValueError("Cannot confirm reversal without sweep")
        self.reversal_confirmed = True


@dataclass
class Confirmation:
    """Confirmation data structure for multi-confirmation system."""
    confirmation_type: str  # 'FVG', 'OB', 'OTE', 'BOS', 'CHoCH', 'LIQUIDITY_GRAB'
    timestamp: datetime
    price: float
    strength: float
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confirmation data."""
        valid_types = ['FVG', 'OB', 'OTE', 'BOS', 'CHoCH', 'LIQUIDITY_GRAB']
        if self.confirmation_type not in valid_types:
            raise ValueError(f"Invalid confirmation_type: {self.confirmation_type}. Must be one of {valid_types}")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

    def deactivate(self):
        """Deactivate this confirmation."""
        self.is_active = False


@dataclass
class ElliottWave:
    """Elliott Wave data structure with validation."""
    wave_number: int
    wave_type: str  # 'IMPULSE' or 'CORRECTIVE'
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    status: str  # 'current', 'completed', etc.
    fibonacci_levels: Dict[str, float] = field(default_factory=dict)
    degree: str = 'MINOR'  # 'MINOR', 'INTERMEDIATE', 'PRIMARY'
    is_extended: bool = False
    momentum_divergence: Optional[bool] = None

    def __post_init__(self):
        """Validate Elliott Wave data."""
        if self.wave_number not in [1, 2, 3, 4, 5]:
            raise ValueError(f"Invalid wave_number: {self.wave_number}. Must be 1-5")

        if self.wave_type not in ['IMPULSE', 'CORRECTIVE']:
            raise ValueError(f"Invalid wave_type: {self.wave_type}. Must be 'IMPULSE' or 'CORRECTIVE'")

        if self.start_price <= 0 or self.end_price <= 0:
            raise ValueError("Prices must be positive")

        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")

        if self.status not in ['current', 'completed', 'invalidated']:
            raise ValueError(f"Invalid status: {self.status}")

        if self.degree not in ['MINOR', 'INTERMEDIATE', 'PRIMARY']:
            raise ValueError(f"Invalid degree: {self.degree}")

    def is_bullish(self) -> bool:
        """Check if wave is bullish."""
        return self.end_price > self.start_price

    def is_bearish(self) -> bool:
        """Check if wave is bearish."""
        return self.end_price < self.start_price

    def get_length(self) -> float:
        """Get wave length (absolute price difference)."""
        return abs(self.end_price - self.start_price)

    def get_percentage_move(self) -> float:
        """Get percentage move of the wave."""
        return abs(self.end_price - self.start_price) / self.start_price * 100

    def is_validated(self) -> bool:
        """Check if wave is validated according to Elliott Wave rules."""
        if self.wave_number == 2:
            # Wave 2 cannot exceed start of Wave 1
            return True  # This will be validated in context
        elif self.wave_number == 3:
            # Wave 3 must break Wave 1 high/low
            return True  # This will be validated in context
        elif self.wave_number == 4:
            # Wave 4 cannot enter Wave 1 territory
            return True  # This will be validated in context
        return True

@dataclass
class ICTConcept:
    """ICT (Inner Circle Trader) concept data structure with validation."""
    timestamp: datetime
    concept_type: str  # 'FVG_BULLISH', 'FVG_BEARISH', 'OB_BULLISH', 'OB_BEARISH', etc.
    start_price: float
    end_price: float
    status: str  # 'current', 'broken', 'filled', etc.
    strength: float  # 0.0 to 1.0
    is_fresh: bool = True  # Fresh vs tested vs broken
    is_filled: bool = False  # Whether the concept has been filled
    is_broken: bool = False  # Whether the concept has been broken
    fill_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ICT concept data."""
        valid_types = [
            'FVG_BULLISH', 'FVG_BEARISH', 'IFVG_BULLISH', 'IFVG_BEARISH',
            'OB_BULLISH', 'OB_BEARISH', 'BB_BULLISH', 'BB_BEARISH',
            'OTE_BULLISH', 'OTE_BEARISH', 'LIQUIDITY_GRAB_BULLISH', 'LIQUIDITY_GRAB_BEARISH'
        ]
        if self.concept_type not in valid_types:
            raise ValueError(f"Invalid concept_type: {self.concept_type}. Must be one of {valid_types}")

        if self.start_price <= 0 or self.end_price <= 0:
            raise ValueError("Prices must be positive")

        if self.start_price >= self.end_price:
            raise ValueError("start_price must be less than end_price")

        if self.status not in ['current', 'broken', 'filled', 'invalidated']:
            raise ValueError(f"Invalid status: {self.status}")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

    def is_bullish(self) -> bool:
        """Check if concept is bullish."""
        return 'BULLISH' in self.concept_type

    def is_bearish(self) -> bool:
        """Check if concept is bearish."""
        return 'BEARISH' in self.concept_type

    def is_fvg(self) -> bool:
        """Check if this is a Fair Value Gap."""
        return 'FVG' in self.concept_type

    def is_ifvg(self) -> bool:
        """Check if this is an Inverse Fair Value Gap."""
        return 'IFVG' in self.concept_type

    def is_order_block(self) -> bool:
        """Check if this is an Order Block."""
        return 'OB' in self.concept_type

    def is_breaker_block(self) -> bool:
        """Check if this is a Breaker Block."""
        return 'BB' in self.concept_type

    def is_ote(self) -> bool:
        """Check if this is an Optimal Trade Entry."""
        return 'OTE' in self.concept_type

    def is_liquidity_grab(self) -> bool:
        """Check if this is a Liquidity Grab."""
        return 'LIQUIDITY_GRAB' in self.concept_type

    def mark_filled(self, fill_timestamp: datetime):
        """Mark this concept as filled."""
        self.status = 'filled'
        self.is_filled = True
        self.fill_timestamp = fill_timestamp
        self.is_fresh = False

    def mark_broken(self):
        """Mark this concept as broken."""
        self.status = 'broken'
        self.is_broken = True
        self.is_fresh = False

    def mark_tested(self):
        """Mark this concept as tested (retested but not broken)."""
        self.is_fresh = False
        # Increment test count if it exists in metadata
        if 'test_count' not in self.metadata:
            self.metadata['test_count'] = 0
        self.metadata['test_count'] += 1

    def get_zone_size(self) -> float:
        """Get the size of the zone."""
        return self.end_price - self.start_price

    def is_price_in_zone(self, price: float) -> bool:
        """Check if a price is within this zone."""
        return self.start_price <= price <= self.end_price

@dataclass
class MarketStructure:
    """Market structure data structure with validation."""
    timestamp: datetime
    structure_type: str  # 'BOS', 'CHoCH', etc.
    price: float
    timeframe: str
    strength: float
    trend_direction: str = 'NEUTRAL'  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    volume_at_break: Optional[float] = None
    impulse_strength: Optional[float] = None
    confirmation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate market structure data."""
        valid_types = ['BOS', 'CHoCH', 'HH', 'HL', 'LH', 'LL']
        if self.structure_type not in valid_types:
            raise ValueError(f"Invalid structure_type: {self.structure_type}. Must be one of {valid_types}")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

        if self.trend_direction not in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            raise ValueError(f"Invalid trend_direction: {self.trend_direction}")

    def is_bullish_structure(self) -> bool:
        """Check if this is a bullish structure."""
        return self.trend_direction == 'BULLISH'

    def is_bearish_structure(self) -> bool:
        """Check if this is a bearish structure."""
        return self.trend_direction == 'BEARISH'

    def is_break_of_structure(self) -> bool:
        """Check if this is a Break of Structure."""
        return self.structure_type == 'BOS'

    def is_change_of_character(self) -> bool:
        """Check if this is a Change of Character."""
        return self.structure_type == 'CHoCH'

    def add_confirmation(self):
        """Add a confirmation to this structure."""
        self.confirmation_count += 1

    def get_strength_score(self) -> float:
        """Calculate overall strength score."""
        base_strength = self.strength

        # Add volume weight if available
        if self.volume_at_break:
            volume_factor = min(self.volume_at_break / 1000000, 2.0)  # Cap at 2x
            base_strength *= (1 + volume_factor * 0.1)

        # Add impulse strength if available
        if self.impulse_strength:
            base_strength *= (1 + self.impulse_strength * 0.2)

        # Add confirmation bonus
        confirmation_bonus = min(self.confirmation_count * 0.1, 0.5)
        base_strength += confirmation_bonus

        return min(base_strength, 1.0)  # Cap at 1.0

@dataclass
class SwingPoint:
    """Swing point data structure with validation."""
    timestamp: datetime
    point_type: str  # 'HIGH' or 'LOW'
    price: float
    strength: int  # Number of candles confirming the swing
    volume: Optional[float] = None
    is_validated: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate swing point data."""
        if self.point_type not in ['HIGH', 'LOW']:
            raise ValueError(f"Invalid point_type: {self.point_type}. Must be 'HIGH' or 'LOW'")

        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}. Must be positive")

        if self.strength < 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be >= 1")

    def is_high(self) -> bool:
        """Check if this is a swing high."""
        return self.point_type == 'HIGH'

    def is_low(self) -> bool:
        """Check if this is a swing low."""
        return self.point_type == 'LOW'

    def get_strength_score(self) -> float:
        """Get normalized strength score (0-1)."""
        return min(self.strength / 10.0, 1.0)  # Cap at 1.0 for 10+ candles

@dataclass
class OrderBlock:
    """Order block data structure with validation."""
    timestamp: datetime
    block_type: str  # 'BULLISH' or 'BEARISH'
    start_price: float
    end_price: float
    volume: float
    strength: float
    is_fresh: bool = True
    test_count: int = 0
    is_broken: bool = False
    break_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate order block data."""
        if self.block_type not in ['BULLISH', 'BEARISH']:
            raise ValueError(f"Invalid block_type: {self.block_type}. Must be 'BULLISH' or 'BEARISH'")

        if self.start_price <= 0 or self.end_price <= 0:
            raise ValueError("Prices must be positive")

        if self.start_price >= self.end_price:
            raise ValueError("start_price must be less than end_price")

        if self.volume < 0:
            raise ValueError("Volume must be non-negative")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

    def is_bullish(self) -> bool:
        """Check if this is a bullish order block."""
        return self.block_type == 'BULLISH'

    def is_bearish(self) -> bool:
        """Check if this is a bearish order block."""
        return self.block_type == 'BEARISH'

    def mark_tested(self):
        """Mark this order block as tested."""
        self.test_count += 1
        self.is_fresh = False

    def mark_broken(self, break_timestamp: datetime):
        """Mark this order block as broken."""
        self.is_broken = True
        self.break_timestamp = break_timestamp
        self.is_fresh = False

    def is_price_in_block(self, price: float) -> bool:
        """Check if a price is within this order block."""
        return self.start_price <= price <= self.end_price

    def get_zone_size(self) -> float:
        """Get the size of the order block zone."""
        return self.end_price - self.start_price

@dataclass
class FairValueGap:
    """Fair Value Gap data structure with validation."""
    timestamp: datetime
    gap_type: str  # 'BULLISH' or 'BEARISH'
    start_price: float
    end_price: float
    gap_size: float
    strength: float
    is_filled: bool = False
    fill_timestamp: Optional[datetime] = None
    is_ifvg: bool = False  # Inverse Fair Value Gap
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate Fair Value Gap data."""
        if self.gap_type not in ['BULLISH', 'BEARISH']:
            raise ValueError(f"Invalid gap_type: {self.gap_type}. Must be 'BULLISH' or 'BEARISH'")

        if self.start_price <= 0 or self.end_price <= 0:
            raise ValueError("Prices must be positive")

        if self.start_price >= self.end_price:
            raise ValueError("start_price must be less than end_price")

        if self.gap_size <= 0:
            raise ValueError("Gap size must be positive")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

    def is_bullish(self) -> bool:
        """Check if this is a bullish FVG."""
        return self.gap_type == 'BULLISH'

    def is_bearish(self) -> bool:
        """Check if this is a bearish FVG."""
        return self.gap_type == 'BEARISH'

    def mark_filled(self, fill_timestamp: datetime):
        """Mark this FVG as filled."""
        self.is_filled = True
        self.fill_timestamp = fill_timestamp

    def is_price_in_gap(self, price: float) -> bool:
        """Check if a price is within this FVG."""
        return self.start_price <= price <= self.end_price

    def get_fill_percentage(self, current_price: float) -> float:
        """Get the percentage of the gap that has been filled."""
        if self.is_bullish():
            if current_price <= self.start_price:
                return 0.0
            elif current_price >= self.end_price:
                return 1.0
            else:
                return (current_price - self.start_price) / (self.end_price - self.start_price)
        else:
            if current_price >= self.start_price:
                return 0.0
            elif current_price <= self.end_price:
                return 1.0
            else:
                return (self.start_price - current_price) / (self.start_price - self.end_price)

@dataclass
class KillZone:
    """Kill zone data structure with validation."""
    zone_type: str  # 'asia', 'london', 'ny', 'london_ny'
    is_active: bool
    strength: float
    timestamp: Optional[datetime] = None
    session_bias: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    liquidity_building: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate kill zone data."""
        valid_types = ['asia', 'london', 'ny', 'london_ny', 'off_hours']
        if self.zone_type not in valid_types:
            raise ValueError(f"Invalid zone_type: {self.zone_type}. Must be one of {valid_types}")

        if not 0 <= self.strength <= 1:
            raise ValueError(f"Invalid strength: {self.strength}. Must be between 0 and 1")

        # Validate session bias (keep original case)
        if self.session_bias not in ['bullish', 'bearish', 'neutral']:
            raise ValueError(f"Invalid session_bias: {self.session_bias}")

    def is_asia(self) -> bool:
        """Check if this is Asia session."""
        return self.zone_type == 'asia'

    def is_london(self) -> bool:
        """Check if this is London session."""
        return self.zone_type == 'london'

    def is_ny(self) -> bool:
        """Check if this is New York session."""
        return self.zone_type == 'ny'

    def is_overlap(self) -> bool:
        """Check if this is London/NY overlap."""
        return self.zone_type == 'london_ny'

    def is_high_priority(self) -> bool:
        """Check if this is a high priority session for entries."""
        return self.is_overlap() or self.is_ny() or self.is_london()

    def is_low_priority(self) -> bool:
        """Check if this is a low priority session (avoid entries)."""
        return self.is_asia()

@dataclass
class TradingSession:
    """Trading session data structure with validation."""
    session_name: str
    start_time: datetime
    end_time: datetime
    timezone: str
    is_active: bool
    session_bias: str = 'neutral'  # 'bullish', 'bearish', 'neutral'
    liquidity_characteristics: str = 'NORMAL'  # 'NORMAL', 'BUILDING', 'BREAKING'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trading session data."""
        # Convert string times to datetime objects if needed
        if isinstance(self.start_time, str):
            self.start_time = self._parse_time_string(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = self._parse_time_string(self.end_time)

        if self.start_time >= self.end_time:
            raise ValueError("start_time must be before end_time")

        # Normalize session bias to uppercase
        self.session_bias = self.session_bias.upper()
        if self.session_bias not in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            raise ValueError(f"Invalid session_bias: {self.session_bias}")

        # Handle liquidity_characteristics - can be string or dict
        if isinstance(self.liquidity_characteristics, dict):
            if not self.liquidity_characteristics:  # Empty dict is invalid
                raise ValueError("liquidity_characteristics cannot be empty")
            # Extract volatility level from dict if present
            volatility = self.liquidity_characteristics.get('volatility', 'NORMAL')
            if volatility == 'high':
                self.liquidity_characteristics = 'BREAKING'
            elif volatility == 'low':
                self.liquidity_characteristics = 'BUILDING'
            else:
                self.liquidity_characteristics = 'NORMAL'
        else:
            # Normalize liquidity characteristics to uppercase
            self.liquidity_characteristics = self.liquidity_characteristics.upper()

        if self.liquidity_characteristics not in ['NORMAL', 'BUILDING', 'BREAKING']:
            raise ValueError(f"Invalid liquidity_characteristics: {self.liquidity_characteristics}")

    def _parse_time_string(self, time_str: str) -> datetime:
        """Parse time string in HH:MM format to datetime."""
        from datetime import time
        try:
            time_obj = datetime.strptime(time_str, '%H:%M').time()
            # Create a datetime object for a reference date (2023-01-01)
            reference_date = datetime(2023, 1, 1).date()
            return datetime.combine(reference_date, time_obj)
        except ValueError:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM format")

    def is_current_session(self, current_time: datetime) -> bool:
        """Check if this session is currently active."""
        # Handle timezone-aware comparison
        if current_time.tzinfo is not None and self.start_time.tzinfo is None:
            # Make start_time timezone-aware
            start_time = self.start_time.replace(tzinfo=current_time.tzinfo)
            end_time = self.end_time.replace(tzinfo=current_time.tzinfo)
        elif current_time.tzinfo is None and self.start_time.tzinfo is not None:
            # Make current_time timezone-aware
            current_time = current_time.replace(tzinfo=self.start_time.tzinfo)
            start_time = self.start_time
            end_time = self.end_time
        else:
            start_time = self.start_time
            end_time = self.end_time

        return start_time <= current_time <= end_time and self.is_active

    def get_session_duration_hours(self) -> float:
        """Get session duration in hours."""
        return (self.end_time - self.start_time).total_seconds() / 3600

@dataclass
class BacktestResult:
    """Backtest result data structure with validation."""
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
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate backtest result data."""
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")

        if self.total_trades < 0:
            raise ValueError("Total trades cannot be negative")

        if self.winning_trades < 0 or self.losing_trades < 0:
            raise ValueError("Winning/losing trades cannot be negative")

        if not 0 <= self.win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")

        if self.profit_factor < 0:
            raise ValueError("Profit factor cannot be negative")

        if not 0 <= self.max_drawdown <= 1:
            raise ValueError("Max drawdown must be between 0 and 1")

        if self.equity_curve is None:
            self.equity_curve = []
        if self.trade_journal is None:
            self.trade_journal = pd.DataFrame()

    def get_total_return(self) -> float:
        """Calculate total return percentage."""
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    def get_net_profit(self) -> float:
        """Calculate net profit."""
        return self.final_balance - self.initial_balance

    def get_average_win(self) -> float:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return 0.0
        winning_trades = self.trade_journal[self.trade_journal['pnl'] > 0]
        return winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0

    def get_average_loss(self) -> float:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return 0.0
        losing_trades = self.trade_journal[self.trade_journal['pnl'] < 0]
        return losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0

@dataclass
class TradingPair:
    """Trading pair data structure with validation."""
    symbol: str
    base_currency: str
    quote_currency: str
    is_active: bool
    min_trade_size: float
    tick_size: float
    lot_size: float = 0.01
    max_trade_size: float = 1000000.0
    price_precision: int = 8
    volume_precision: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trading pair data."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

        if not self.base_currency or not self.quote_currency:
            raise ValueError("Base and quote currencies must be specified")

        if self.min_trade_size < 0:
            raise ValueError("Minimum trade size cannot be negative")

        if self.tick_size <= 0:
            raise ValueError("Tick size must be positive")

        if self.lot_size <= 0:
            raise ValueError("Lot size must be positive")

        if self.max_trade_size <= self.min_trade_size:
            raise ValueError("Maximum trade size must be greater than minimum")

    def get_pair_name(self) -> str:
        """Get formatted pair name."""
        return f"{self.base_currency}/{self.quote_currency}"

    def is_valid_trade_size(self, size: float) -> bool:
        """Check if trade size is valid for this pair."""
        return self.min_trade_size <= size <= self.max_trade_size

    def round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size

    def round_volume(self, volume: float) -> float:
        """Round volume to lot size."""
        return round(volume / self.lot_size) * self.lot_size

@dataclass
class RiskManagement:
    """Risk management data structure with validation."""
    max_risk_per_trade: float
    max_daily_risk: float
    max_positions: int
    stop_loss_pips: float
    take_profit_pips: float
    max_drawdown_percent: float = 0.15
    drawdown_recovery_percent: float = 0.05
    volatility_factor: bool = True
    correlation_limit: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate risk management data."""
        if not 0 < self.max_risk_per_trade <= 1:
            raise ValueError("Max risk per trade must be between 0 and 1")

        if not 0 < self.max_daily_risk <= 1:
            raise ValueError("Max daily risk must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")

        if self.stop_loss_pips <= 0:
            raise ValueError("Stop loss pips must be positive")

        if self.take_profit_pips <= 0:
            raise ValueError("Take profit pips must be positive")

        if not 0 < self.max_drawdown_percent <= 1:
            raise ValueError("Max drawdown percent must be between 0 and 1")

        if not 0 < self.drawdown_recovery_percent <= 1:
            raise ValueError("Drawdown recovery percent must be between 0 and 1")

        if not 0 < self.correlation_limit <= 1:
            raise ValueError("Correlation limit must be between 0 and 1")

    def is_risk_limit_exceeded(self, current_risk: float) -> bool:
        """Check if current risk exceeds limits."""
        return current_risk > self.max_risk_per_trade

    def is_daily_risk_exceeded(self, daily_risk: float) -> bool:
        """Check if daily risk exceeds limits."""
        return daily_risk > self.max_daily_risk

    def is_drawdown_exceeded(self, current_drawdown: float) -> bool:
        """Check if current drawdown exceeds limits."""
        return current_drawdown > self.max_drawdown_percent

    def can_recover_from_drawdown(self, current_drawdown: float) -> bool:
        """Check if can recover from current drawdown."""
        return current_drawdown < self.drawdown_recovery_percent


@dataclass
class PositionState:
    """Position state data structure for tracking partial exits."""
    entry_qty: float
    remaining_qty: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    stop_at_be: bool = False
    stop_at_tp1: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profits: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate position state data."""
        if self.entry_qty <= 0:
            raise ValueError("Entry quantity must be positive")

        if self.remaining_qty < 0:
            raise ValueError("Remaining quantity cannot be negative")

        if self.remaining_qty > self.entry_qty:
            raise ValueError("Remaining quantity cannot exceed entry quantity")

    def close_position_percentage(self, percentage: float) -> float:
        """Close a percentage of the position."""
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")

        close_qty = self.entry_qty * percentage
        self.remaining_qty = max(0, self.remaining_qty - close_qty)
        return close_qty

    def is_position_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.remaining_qty <= 0

    def get_closed_percentage(self) -> float:
        """Get percentage of position that has been closed."""
        return (self.entry_qty - self.remaining_qty) / self.entry_qty


@dataclass
class WaveSequence:
    """Elliott Wave sequence data structure."""
    waves: List[ElliottWave]
    sequence_type: str  # 'IMPULSE', 'CORRECTIVE', 'COMPLETE'
    is_valid: bool = True
    invalidation_level: Optional[float] = None
    target_levels: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate wave sequence data."""
        if not self.waves:
            raise ValueError("Wave sequence cannot be empty")

        if self.sequence_type not in ['IMPULSE', 'CORRECTIVE', 'COMPLETE']:
            raise ValueError(f"Invalid sequence_type: {self.sequence_type}")

        # Validate wave order
        for i, wave in enumerate(self.waves):
            if wave.wave_number != i + 1:
                raise ValueError(f"Wave {i+1} has incorrect wave number: {wave.wave_number}")

    def get_sequence_length(self) -> int:
        """Get the number of waves in the sequence."""
        return len(self.waves)

    def is_complete_impulse(self) -> bool:
        """Check if this is a complete 5-wave impulse sequence."""
        return (self.sequence_type == 'IMPULSE' and
                len(self.waves) == 5 and
                all(w.wave_type == 'IMPULSE' for w in self.waves[::2]) and
                all(w.wave_type == 'CORRECTIVE' for w in self.waves[1::2]))

    def is_complete_correction(self) -> bool:
        """Check if this is a complete ABC correction."""
        return (self.sequence_type == 'CORRECTIVE' and
                len(self.waves) == 3 and
                self.waves[0].wave_type == 'CORRECTIVE' and
                self.waves[1].wave_type == 'IMPULSE' and
                self.waves[2].wave_type == 'CORRECTIVE')

    def get_sequence_direction(self) -> str:
        """Get the overall direction of the sequence."""
        if not self.waves:
            return 'NEUTRAL'

        first_wave = self.waves[0]
        if first_wave.is_bullish():
            return 'BULLISH'
        elif first_wave.is_bearish():
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def add_wave(self, wave: ElliottWave):
        """Add a wave to the sequence."""
        if wave.wave_number != len(self.waves) + 1:
            raise ValueError(f"Cannot add wave {wave.wave_number} to sequence of length {len(self.waves)}")

        self.waves.append(wave)

    def invalidate_sequence(self, invalidation_level: float):
        """Invalidate the sequence at the given level."""
        self.is_valid = False
        self.invalidation_level = invalidation_level


@dataclass
class Position:
    """Position data structure for tracking active trades."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profits: List[float]
    entry_time: datetime
    timestamp: Optional[datetime] = None  # Added timestamp field
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_active: bool = True
    partial_exits: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate position data."""
        if self.side not in ['LONG', 'SHORT']:
            raise ValueError(f"Invalid side: {self.side}. Must be 'LONG' or 'SHORT'")

        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}. Must be positive")

        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}. Must be positive")

        if self.stop_loss <= 0:
            raise ValueError(f"Invalid stop_loss: {self.stop_loss}. Must be positive")

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == 'LONG'

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == 'SHORT'

    @property
    def signal_type(self) -> str:
        """Get signal type for backward compatibility."""
        return 'BUY' if self.side == 'LONG' else 'SELL'

    def update_current_price(self, current_price: float):
        """Update current price and calculate unrealized PnL."""
        self.current_price = current_price

        if self.is_long():
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def close_position(self, exit_price: float, exit_time: datetime, reason: str = "MANUAL"):
        """Close the position."""
        if self.is_long():
            pnl = (exit_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - exit_price) * self.quantity

        self.realized_pnl += pnl
        self.is_active = False

        # Record the exit
        self.partial_exits.append({
            'timestamp': exit_time,
            'exit_price': exit_price,
            'quantity': self.quantity,
            'pnl': pnl,
            'reason': reason
        })

    def partial_close(self, exit_price: float, exit_quantity: float, exit_time: datetime, reason: str = "TAKE_PROFIT"):
        """Partially close the position."""
        if exit_quantity >= self.quantity:
            self.close_position(exit_price, exit_time, reason)
            return

        if self.is_long():
            pnl = (exit_price - self.entry_price) * exit_quantity
        else:
            pnl = (self.entry_price - exit_price) * exit_quantity

        self.realized_pnl += pnl
        self.quantity -= exit_quantity

        # Record the partial exit
        self.partial_exits.append({
            'timestamp': exit_time,
            'exit_price': exit_price,
            'quantity': exit_quantity,
            'pnl': pnl,
            'reason': reason
        })

    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def get_risk_amount(self) -> float:
        """Get risk amount (absolute difference between entry and stop)."""
        return abs(self.entry_price - self.stop_loss) * self.quantity

    def get_reward_amount(self, tp_index: int = 0) -> float:
        """Calculate reward amount for a specific take profit level."""
        if tp_index >= len(self.take_profits):
            raise IndexError(f"Take profit index {tp_index} out of range")

        tp = self.take_profits[tp_index]
        if self.is_long():
            return (tp - self.entry_price) * self.quantity
        else:
            return (self.entry_price - tp) * self.quantity
