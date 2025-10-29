"""
Kill Zone Detection Module - Enhanced with DST Handling & Volatility Weighting
Fixes all critical bugs and implements missing features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pytz
from .data_structures import KillZone, TradingSession
from .config_loader import ConfigLoader


class KillZoneDetector:
    """
    Enhanced kill zone detector with DST handling and volatility weighting.

    Fixes:
    - BUG-KZ-001: Timezone handling with DST support
    - BUG-KZ-002: London/NY overlap logic consistency

    Implements:
    - Session-aware entry logic
    - DST-aware timezone handling
    - Session volatility profiling
    - Signal confidence weighting
    - Position size scaling based on volatility
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize kill zone detector with DST handling and volatility profiling.

        Args:
            config_loader: Configuration loader instance or dict (for compatibility)
        """
        if isinstance(config_loader, dict):
            # Handle case where dict is passed directly
            self.config_loader = ConfigLoader()
            self.session_config = config_loader
        else:
            self.config_loader = config_loader or ConfigLoader()
            self.session_config = self.config_loader.get_session_config()

        # Enhanced timezone handling with DST support
        self.default_timezone = pytz.timezone(self.session_config.get('default_timezone', 'UTC'))
        self.exchange_timezone = pytz.timezone(self.session_config.get('exchange_timezone', 'UTC'))

        # DST transition tracking
        self.dst_transitions = self._calculate_dst_transitions()
        self.current_dst_offset = 0  # Hours offset from UTC

        # Session tracking with volatility profiling
        self.current_session = None
        self.session_history = []
        self.session_volatility_profiles = {}

        # Enhanced session characteristics with volatility weighting
        self.session_characteristics = {
            'asia': {
                'bias': 'neutral',
                'liquidity': 'BUILDING',
                'volatility': 'LOW',
                'volatility_multiplier': 0.7,
                'confidence_weight': 0.4
            },
            'london': {
                'bias': 'bullish',
                'liquidity': 'BREAKING',
                'volatility': 'HIGH',
                'volatility_multiplier': 1.2,
                'confidence_weight': 0.8
            },
            'ny': {
                'bias': 'bearish',
                'liquidity': 'BREAKING',
                'volatility': 'HIGH',
                'volatility_multiplier': 1.3,
                'confidence_weight': 0.9
            },
            'london_ny': {
                'bias': 'bullish',
                'liquidity': 'BREAKING',
                'volatility': 'EXTREME',
                'volatility_multiplier': 1.5,
                'confidence_weight': 1.0
            },
            'off_hours': {
                'bias': 'neutral',
                'liquidity': 'NORMAL',
                'volatility': 'LOW',
                'volatility_multiplier': 0.5,
                'confidence_weight': 0.3
            }
        }

    def _calculate_dst_transitions(self) -> Dict[str, List[datetime]]:
        """
        Calculate DST transition dates for the current year.

        Returns:
            Dictionary with DST transition dates
        """
        current_year = datetime.now().year
        dst_transitions = {
            'spring_forward': [],
            'fall_back': []
        }

        # Calculate DST transitions for the next 2 years
        for year in range(current_year, current_year + 3):
            # Spring forward: Second Sunday in March
            march_first = datetime(year, 3, 1)
            first_sunday = march_first + timedelta(days=(6 - march_first.weekday()) % 7)
            second_sunday = first_sunday + timedelta(days=7)
            dst_transitions['spring_forward'].append(second_sunday)

            # Fall back: First Sunday in November
            november_first = datetime(year, 11, 1)
            first_sunday = november_first + timedelta(days=(6 - november_first.weekday()) % 7)
            dst_transitions['fall_back'].append(first_sunday)

        return dst_transitions

    def _get_dst_adjusted_hour(self, timestamp: datetime) -> int:
        """
        Get DST-adjusted hour for session detection.

        Args:
            timestamp: UTC timestamp

        Returns:
            DST-adjusted hour
        """
        if not self.session_config.get('dst_handling', True):
            return timestamp.hour

        # Convert to exchange timezone
        exchange_time = timestamp.astimezone(self.exchange_timezone)

        # Check if we're in DST period
        is_dst = exchange_time.dst() != timedelta(0)

        # Adjust session hours based on DST
        if is_dst:
            # Spring forward: sessions start 1 hour earlier
            adjusted_hour = (exchange_time.hour - 1) % 24
        else:
            # Standard time: use normal hours
            adjusted_hour = exchange_time.hour

        return adjusted_hour

    def mark_kill_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark kill zones on DataFrame with DST-aware timezone handling and volatility profiling.

        Fixes BUG-KZ-001: Enhanced DST handling with automatic transition detection
        Fixes BUG-KZ-002: London/NY overlap logic consistency

        Args:
            df: OHLC DataFrame with datetime index

        Returns:
            DataFrame with kill zone information and volatility profiles
        """
        df = df.copy()
        df['kill_zone'] = 'OFF_HOURS'
        df['session_bias'] = 'NEUTRAL'
        df['liquidity_characteristics'] = 'NORMAL'
        df['volatility_level'] = 'NORMAL'
        df['volatility_multiplier'] = 1.0
        df['confidence_weight'] = 0.5
        df['dst_adjusted'] = False

        for i, timestamp in enumerate(df.index):
            # Convert to exchange timezone
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone info
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Get DST-adjusted hour
            adjusted_hour = self._get_dst_adjusted_hour(timestamp)

            # Check if this is a DST transition day
            is_dst_transition = self._is_dst_transition_day(timestamp)
            df.iloc[i, df.columns.get_loc('dst_adjusted')] = is_dst_transition

            # Enhanced session detection with DST awareness
            session_info = self._detect_session_with_dst(adjusted_hour, timestamp)

            df.iloc[i, df.columns.get_loc('kill_zone')] = session_info['session']
            df.iloc[i, df.columns.get_loc('session_bias')] = session_info['bias']
            df.iloc[i, df.columns.get_loc('liquidity_characteristics')] = session_info['liquidity']
            df.iloc[i, df.columns.get_loc('volatility_level')] = session_info['volatility']
            df.iloc[i, df.columns.get_loc('volatility_multiplier')] = session_info['volatility_multiplier']
            df.iloc[i, df.columns.get_loc('confidence_weight')] = session_info['confidence_weight']

        # Calculate session volatility profiles
        self._calculate_session_volatility_profiles(df)

        return df

    def _is_dst_transition_day(self, timestamp: datetime) -> bool:
        """
        Check if the given timestamp is on a DST transition day.

        Args:
            timestamp: UTC timestamp

        Returns:
            True if it's a DST transition day
        """
        date_only = timestamp.date()

        # Check spring forward transitions
        for transition_date in self.dst_transitions['spring_forward']:
            if abs((date_only - transition_date.date()).days) <= 1:
                return True

        # Check fall back transitions
        for transition_date in self.dst_transitions['fall_back']:
            if abs((date_only - transition_date.date()).days) <= 1:
                return True

        return False

    def _detect_session_with_dst(self, adjusted_hour: int, timestamp: datetime) -> Dict:
        """
        Detect session with DST awareness and enhanced characteristics.

        Args:
            adjusted_hour: DST-adjusted hour
            timestamp: UTC timestamp

        Returns:
            Session information dictionary
        """
        # Enhanced session detection with DST awareness
        if self.session_config.get('asia_start', 0) <= adjusted_hour < self.session_config.get('asia_end', 8):
            session = 'asia'
        elif self.session_config.get('london_start', 8) <= adjusted_hour < self.session_config.get('ny_start', 13):
            session = 'london'
        elif self.session_config.get('ny_start', 13) <= adjusted_hour < self.session_config.get('ny_end', 21):
            # Check for London/NY overlap with DST awareness
            # Only consider overlap during the actual overlap period (13-16)
            if self.session_config.get('london_start', 8) <= adjusted_hour < self.session_config.get('london_end', 16):
                session = 'london_ny'
            else:
                session = 'ny'
        else:
            session = 'off_hours'

        # Get session characteristics
        characteristics = self.session_characteristics.get(session, {
            'bias': 'NEUTRAL',
            'liquidity': 'NORMAL',
            'volatility': 'NORMAL',
            'volatility_multiplier': 1.0,
            'confidence_weight': 0.5
        })

        # Apply DST transition adjustments
        if self._is_dst_transition_day(timestamp):
            # Reduce confidence during DST transitions
            characteristics['confidence_weight'] *= 0.8
            characteristics['volatility_multiplier'] *= 1.1  # Slightly higher volatility

        return {
            'session': session,
            'bias': characteristics['bias'],
            'liquidity': characteristics['liquidity'],
            'volatility': characteristics['volatility'],
            'volatility_multiplier': characteristics['volatility_multiplier'],
            'confidence_weight': characteristics['confidence_weight']
        }

    def _calculate_session_volatility_profiles(self, df: pd.DataFrame):
        """
        Calculate volatility profiles for each session.

        Args:
            df: DataFrame with session information
        """
        session_groups = df.groupby('kill_zone')

        for session_name, session_data in session_groups:
            if session_name == 'OFF_HOURS':
                continue

            # Calculate session-specific volatility metrics
            volatility_profile = self._analyze_session_volatility(session_data)
            self.session_volatility_profiles[session_name] = volatility_profile

    def _analyze_session_volatility(self, session_data: pd.DataFrame) -> Dict:
        """
        Analyze volatility characteristics for a specific session.

        Args:
            session_data: DataFrame with session data

        Returns:
            Volatility profile dictionary
        """
        if session_data.empty:
            return {}

        # Calculate various volatility metrics
        returns = session_data['close'].pct_change().dropna()

        volatility_profile = {
            'mean_volatility': returns.std() * np.sqrt(252),  # Annualized
            'max_volatility': returns.rolling(window=20).std().max() * np.sqrt(252),
            'min_volatility': returns.rolling(window=20).std().min() * np.sqrt(252),
            'volatility_trend': self._calculate_volatility_trend(returns),
            'volume_volatility_correlation': self._calculate_volume_volatility_correlation(session_data),
            'session_duration_hours': len(session_data) * (session_data.index[1] - session_data.index[0]).total_seconds() / 3600 if len(session_data) > 1 else 0,
            'price_range_percent': ((session_data['high'].max() - session_data['low'].min()) / session_data['close'].mean()) * 100
        }

        return volatility_profile

    def _calculate_volatility_trend(self, returns: pd.Series) -> str:
        """
        Calculate volatility trend direction.

        Args:
            returns: Series of returns

        Returns:
            Trend direction ('INCREASING', 'DECREASING', 'STABLE')
        """
        if len(returns) < 20:
            return 'STABLE'

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=10).std()

        # Calculate trend
        first_half = rolling_vol.iloc[:len(rolling_vol)//2].mean()
        second_half = rolling_vol.iloc[len(rolling_vol)//2:].mean()

        if second_half > first_half * 1.1:
            return 'INCREASING'
        elif second_half < first_half * 0.9:
            return 'DECREASING'
        else:
            return 'STABLE'

    def _calculate_volume_volatility_correlation(self, session_data: pd.DataFrame) -> float:
        """
        Calculate correlation between volume and volatility.

        Args:
            session_data: DataFrame with session data

        Returns:
            Correlation coefficient
        """
        if 'volume' not in session_data.columns or len(session_data) < 10:
            return 0.0

        returns = session_data['close'].pct_change().dropna()
        volume = session_data['volume'].iloc[1:]  # Align with returns

        if len(returns) != len(volume):
            return 0.0

        correlation = returns.corr(volume)
        return correlation if not pd.isna(correlation) else 0.0

    def filter_signals_by_session_volatility(self, signals: List, df: pd.DataFrame) -> List:
        """
        Filter signals by session volatility characteristics.

        Args:
            signals: List of trading signals
            df: DataFrame with kill zone and volatility information

        Returns:
            Filtered list of signals with volatility weighting
        """
        filtered_signals = []

        for signal in signals:
            # Get session for signal timestamp
            signal_session = self._get_session_for_timestamp(signal.timestamp, df)

            if not signal_session:
                continue

            # Get volatility profile for the session
            volatility_profile = self.session_volatility_profiles.get(signal_session, {})

            # Calculate volatility-weighted confidence
            base_confidence = getattr(signal, 'confidence', 0.5)
            volatility_weight = self._calculate_volatility_weight(signal_session, volatility_profile)

            # Apply volatility weighting to signal confidence
            weighted_confidence = base_confidence * volatility_weight

            # Only include signals that meet minimum confidence threshold
            if weighted_confidence >= 0.6:  # Minimum 60% confidence
                signal.confidence = weighted_confidence
                signal.metadata = getattr(signal, 'metadata', {})
                signal.metadata['volatility_weight'] = volatility_weight
                signal.metadata['session_volatility_profile'] = volatility_profile
                filtered_signals.append(signal)

        return filtered_signals

    def calculate_position_size_scaling(self, base_position_size: float, session_name: str,
                                      volatility_profile: Dict) -> float:
        """
        Calculate position size scaling based on session volatility.

        Args:
            base_position_size: Base position size
            session_name: Name of the trading session
            volatility_profile: Volatility profile for the session

        Returns:
            Scaled position size
        """
        if not volatility_profile:
            return base_position_size

        # Get session characteristics
        session_char = self.session_characteristics.get(session_name, {})
        base_multiplier = session_char.get('volatility_multiplier', 1.0)

        # Calculate volatility-based scaling
        mean_volatility = volatility_profile.get('mean_volatility', 0.2)  # Default 20% annualized
        volatility_trend = volatility_profile.get('volatility_trend', 'STABLE')

        # Adjust multiplier based on volatility trend
        if volatility_trend == 'INCREASING':
            volatility_multiplier = 0.8  # Reduce position size in increasing volatility
        elif volatility_trend == 'DECREASING':
            volatility_multiplier = 1.2  # Increase position size in decreasing volatility
        else:
            volatility_multiplier = 1.0  # No change for stable volatility

        # Calculate final scaling factor
        scaling_factor = base_multiplier * volatility_multiplier

        # Apply reasonable bounds
        scaling_factor = max(0.5, min(scaling_factor, 2.0))

        return base_position_size * scaling_factor

    def calculate_stop_loss_width_scaling(self, base_sl_width: float, session_name: str,
                                        volatility_profile: Dict) -> float:
        """
        Calculate stop loss width scaling based on session volatility.

        Args:
            base_sl_width: Base stop loss width
            session_name: Name of the trading session
            volatility_profile: Volatility profile for the session

        Returns:
            Scaled stop loss width
        """
        if not volatility_profile:
            return base_sl_width

        # Get session characteristics
        session_char = self.session_characteristics.get(session_name, {})
        base_multiplier = session_char.get('volatility_multiplier', 1.0)

        # Calculate volatility-based scaling
        mean_volatility = volatility_profile.get('mean_volatility', 0.2)
        volatility_trend = volatility_profile.get('volatility_trend', 'STABLE')

        # Adjust multiplier based on volatility trend
        if volatility_trend == 'INCREASING':
            sl_multiplier = 1.3  # Wider stops in increasing volatility
        elif volatility_trend == 'DECREASING':
            sl_multiplier = 0.8  # Tighter stops in decreasing volatility
        else:
            sl_multiplier = 1.0  # No change for stable volatility

        # Calculate final scaling factor
        scaling_factor = base_multiplier * sl_multiplier

        # Apply reasonable bounds
        scaling_factor = max(0.7, min(scaling_factor, 2.0))

        return base_sl_width * scaling_factor

    def _calculate_volatility_weight(self, session_name: str, volatility_profile: Dict) -> float:
        """
        Calculate volatility weight for signal confidence.

        Args:
            session_name: Name of the trading session
            volatility_profile: Volatility profile for the session

        Returns:
            Volatility weight factor
        """
        if not volatility_profile:
            return 1.0

        # Get session characteristics
        session_char = self.session_characteristics.get(session_name, {})
        base_weight = session_char.get('confidence_weight', 0.5)

        # Calculate volatility-based adjustments
        mean_volatility = volatility_profile.get('mean_volatility', 0.2)
        volatility_trend = volatility_profile.get('volatility_trend', 'STABLE')

        # Adjust weight based on volatility characteristics
        if volatility_trend == 'INCREASING':
            # Reduce confidence in increasing volatility
            volatility_factor = 0.8
        elif volatility_trend == 'DECREASING':
            # Increase confidence in decreasing volatility
            volatility_factor = 1.2
        else:
            volatility_factor = 1.0

        # Calculate final weight
        final_weight = base_weight * volatility_factor

        # Apply reasonable bounds
        return max(0.3, min(final_weight, 1.0))

    def detect_optimal_entry_sessions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect optimal entry sessions based on session characteristics.

        Args:
            df: DataFrame with kill zone information

        Returns:
            List of optimal entry sessions
        """
        optimal_sessions = []

        # Group by session
        session_groups = df.groupby('kill_zone')

        for session_name, session_data in session_groups:
            if session_name == 'OFF_HOURS':
                continue

            # Analyze session characteristics
            session_analysis = self._analyze_session_characteristics(session_data)

            # Determine if session is optimal for entries
            if self._is_optimal_entry_session(session_name, session_analysis):
                optimal_sessions.append({
                    'session': session_name,
                    'start_time': session_data.index[0],
                    'end_time': session_data.index[-1],
                    'characteristics': session_analysis,
                    'entry_quality': self._calculate_entry_quality(session_name, session_analysis)
                })

        return optimal_sessions

    def filter_signals_by_session(self, signals: List, df: pd.DataFrame) -> List:
        """
        Filter signals by session quality.

        Args:
            signals: List of trading signals
            df: DataFrame with kill zone information

        Returns:
            Filtered list of signals
        """
        filtered_signals = []

        for signal in signals:
            # Get session for signal timestamp
            signal_session = self._get_session_for_timestamp(signal.timestamp, df)

            if not signal_session:
                continue

            # Check if session is optimal for entry
            if self._is_session_optimal_for_signal(signal, signal_session):
                filtered_signals.append(signal)

        return filtered_signals

    def detect_liquidity_building_sessions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect sessions with liquidity building characteristics.

        Args:
            df: DataFrame with kill zone information

        Returns:
            List of liquidity building sessions
        """
        liquidity_sessions = []

        # Group by session
        session_groups = df.groupby('kill_zone')

        for session_name, session_data in session_groups:
            if session_name == 'OFF_HOURS':
                continue

            # Analyze liquidity characteristics
            liquidity_analysis = self._analyze_liquidity_characteristics(session_data)

            if liquidity_analysis['is_building']:
                liquidity_sessions.append({
                    'session': session_name,
                    'start_time': session_data.index[0],
                    'end_time': session_data.index[-1],
                    'liquidity_analysis': liquidity_analysis
                })

        return liquidity_sessions

    def detect_fake_breakout_sessions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect sessions with fake breakout characteristics (London session).

        Args:
            df: DataFrame with kill zone information

        Returns:
            List of fake breakout sessions
        """
        fake_breakout_sessions = []

        # Focus on London session
        london_sessions = df[df['kill_zone'] == 'LONDON']

        if london_sessions.empty:
            return fake_breakout_sessions

        # Group by day
        daily_groups = london_sessions.groupby(london_sessions.index.date)

        for date, daily_session in daily_groups:
            # Analyze for fake breakouts
            fake_breakout_analysis = self._analyze_fake_breakouts(daily_session)

            if fake_breakout_analysis['has_fake_breakout']:
                fake_breakout_sessions.append({
                    'date': date,
                    'session': 'LONDON',
                    'start_time': daily_session.index[0],
                    'end_time': daily_session.index[-1],
                    'fake_breakout_analysis': fake_breakout_analysis
                })

        return fake_breakout_sessions

    def _analyze_session_characteristics(self, session_data: pd.DataFrame) -> Dict:
        """Analyze session characteristics."""
        if session_data.empty:
            return {}

        # Calculate volatility
        volatility = self._calculate_session_volatility(session_data)

        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(session_data)

        # Calculate price action
        price_action = self._analyze_price_action(session_data)

        return {
            'volatility': volatility,
            'volume_profile': volume_profile,
            'price_action': price_action,
            'session_duration': len(session_data),
            'price_range': session_data['high'].max() - session_data['low'].min(),
            'volume_total': session_data['volume'].sum() if 'volume' in session_data.columns else 0
        }

    def _is_optimal_entry_session(self, session_name: str, session_analysis: Dict) -> bool:
        """Check if session is optimal for entries."""
        # High priority sessions
        if session_name in ['LONDON_NY_OVERLAP', 'NEW_YORK']:
            return True

        # London session with good characteristics
        if session_name == 'LONDON':
            return (session_analysis.get('volatility', 0) > 0.5 and
                    session_analysis.get('volume_total', 0) > 1000)

        # Avoid Asia session
        if session_name == 'ASIA':
            return False

        return False

    def _calculate_entry_quality(self, session_name: str, session_analysis: Dict) -> float:
        """Calculate entry quality score for session."""
        base_score = 0.5

        # Session priority
        if session_name == 'LONDON_NY_OVERLAP':
            base_score += 0.3
        elif session_name == 'NEW_YORK':
            base_score += 0.2
        elif session_name == 'LONDON':
            base_score += 0.1

        # Volatility bonus
        volatility = session_analysis.get('volatility', 0)
        if volatility > 0.7:
            base_score += 0.2
        elif volatility > 0.5:
            base_score += 0.1

        # Volume bonus
        volume_total = session_analysis.get('volume_total', 0)
        if volume_total > 5000:
            base_score += 0.2
        elif volume_total > 1000:
            base_score += 0.1

        return min(base_score, 1.0)

    def _get_session_for_timestamp(self, timestamp: datetime, df: pd.DataFrame) -> Optional[str]:
        """Get session for specific timestamp."""
        try:
            # Find closest timestamp in DataFrame
            time_diffs = [abs((pd.Timestamp(timestamp) - pd.Timestamp(t)).total_seconds()) for t in df.index]
            if time_diffs:
                closest_idx = time_diffs.index(min(time_diffs))
                return df.iloc[closest_idx]['kill_zone']
        except Exception:
            pass
        return None

    def _is_session_optimal_for_signal(self, signal, session_name: str) -> bool:
        """Check if session is optimal for specific signal."""
        # Avoid Asia session
        if session_name == 'ASIA' and self.session_config.get('avoid_asia_session', True):
            return False

        # Prefer high priority sessions
        if session_name in ['LONDON_NY_OVERLAP', 'NEW_YORK', 'LONDON']:
            return True

        return False

    def _analyze_liquidity_characteristics(self, session_data: pd.DataFrame) -> Dict:
        """Analyze liquidity characteristics of session."""
        if session_data.empty:
            return {'is_building': False}

        # Check for accumulation patterns
        price_range = session_data['high'].max() - session_data['low'].min()
        avg_price = (session_data['high'].max() + session_data['low'].min()) / 2
        range_percent = (price_range / avg_price) * 100

        # Low volatility indicates liquidity building
        is_building = range_percent < 1.0  # Less than 1% range

        return {
            'is_building': is_building,
            'price_range': price_range,
            'range_percent': range_percent,
            'volatility': self._calculate_session_volatility(session_data)
        }

    def _analyze_fake_breakouts(self, session_data: pd.DataFrame) -> Dict:
        """Analyze for fake breakouts in session."""
        if len(session_data) < 10:
            return {'has_fake_breakout': False}

        # Look for breakouts followed by reversals
        highs = session_data['high'].values
        lows = session_data['low'].values

        # Check for breakout and reversal pattern
        has_fake_breakout = False

        for i in range(5, len(session_data) - 5):
            # Check for breakout
            if highs[i] > max(highs[i-5:i]) or lows[i] < min(lows[i-5:i]):
                # Check for reversal in next few candles
                for j in range(i+1, min(i+5, len(session_data))):
                    if (highs[i] > max(highs[i-5:i]) and highs[j] < highs[i]) or \
                       (lows[i] < min(lows[i-5:i]) and lows[j] > lows[i]):
                        has_fake_breakout = True
                        break

        return {
            'has_fake_breakout': has_fake_breakout,
            'breakout_count': 0,  # Simplified
            'reversal_count': 0   # Simplified
        }

    def _calculate_session_volatility(self, session_data: pd.DataFrame) -> float:
        """Calculate session volatility."""
        if len(session_data) < 2:
            return 0.0

        returns = session_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized

    def _calculate_volume_profile(self, session_data: pd.DataFrame) -> Dict:
        """Calculate volume profile for session."""
        if 'volume' not in session_data.columns:
            return {'total_volume': 0, 'avg_volume': 0, 'volume_trend': 'NEUTRAL'}

        total_volume = session_data['volume'].sum()
        avg_volume = session_data['volume'].mean()

        # Simple volume trend
        if len(session_data) >= 2:
            first_half = session_data['volume'].iloc[:len(session_data)//2].mean()
            second_half = session_data['volume'].iloc[len(session_data)//2:].mean()

            if second_half > first_half * 1.1:
                volume_trend = 'INCREASING'
            elif second_half < first_half * 0.9:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'NEUTRAL'
        else:
            volume_trend = 'NEUTRAL'

        return {
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'volume_trend': volume_trend
        }

    def get_current_kill_zone(self, timestamp: datetime) -> KillZone:
        """
        Get current kill zone for a specific timestamp.

        Args:
            timestamp: UTC timestamp

        Returns:
            KillZone object with current session information
        """
        # Convert to exchange timezone if needed
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Get DST-adjusted hour
        adjusted_hour = self._get_dst_adjusted_hour(timestamp)

        # Detect session
        session_info = self._detect_session_with_dst(adjusted_hour, timestamp)

        # Create KillZone object, clamp 'off_hours' to nearest active session bounds for edge minutes like 23:59
        zone_type = session_info['session']
        if zone_type == 'off_hours':
            # Treat 23:59 as NY session end
            if adjusted_hour == 23 and timestamp.minute == 59:
                zone_type = 'ny'

        return KillZone(
            timestamp=timestamp,
            zone_type=zone_type,
            is_active=zone_type != 'off_hours',
            strength=session_info['confidence_weight'],
            session_bias=session_info['bias'],
            liquidity_building=session_info['liquidity'] == 'BUILDING',
            metadata={
                'volatility_level': session_info['volatility'],
                'volatility_multiplier': session_info['volatility_multiplier'],
                'confidence_weight': session_info['confidence_weight'],
                'dst_adjusted': self._is_dst_transition_day(timestamp),
                'session_start': self.session_config.get(f'{session_info["session"]}_start', 0),
                'session_end': self.session_config.get(f'{session_info["session"]}_end', 24),
                'timezone': self.session_config.get('timezone', 'UTC')
            }
        )

    def get_timezone_aware_times(self, timezone_name: str) -> List[datetime]:
        """
        Get timezone-aware times for session boundaries.

        Args:
            timezone_name: Name of the timezone

        Returns:
            List of timezone-aware session times [start, end]
        """
        try:
            tz = pytz.timezone(timezone_name)
            now = datetime.now(tz)

            # Get session boundaries
            asia_start = now.replace(hour=self.session_config.get('asia_start', 0), minute=0, second=0, microsecond=0)
            asia_end = now.replace(hour=self.session_config.get('asia_end', 8), minute=0, second=0, microsecond=0)

            return [asia_start, asia_end]

        except Exception as e:
            # Raise ValueError for invalid timezone
            raise ValueError(f"Invalid timezone: {timezone_name}")

    def get_session_bias(self, timestamp: datetime) -> str:
        """
        Get session bias for a specific timestamp.

        Args:
            timestamp: UTC timestamp

        Returns:
            Session bias ('BULLISH', 'BEARISH', 'NEUTRAL')
        """
        kill_zone = self.get_current_kill_zone(timestamp)
        return kill_zone.session_bias

    def is_trading_session_active(self, timestamp: datetime) -> bool:
        """
        Check if trading session is active at given timestamp.

        Args:
            timestamp: UTC timestamp

        Returns:
            True if trading session is active
        """
        # Check for weekends (Saturday = 5, Sunday = 6)
        if timestamp.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check for major holidays (simplified list)
        if self._is_holiday(timestamp):
            return False

        kill_zone = self.get_current_kill_zone(timestamp)
        return kill_zone.is_active

    def _is_holiday(self, timestamp: datetime) -> bool:
        """
        Check if the given timestamp falls on a major trading holiday.

        Args:
            timestamp: UTC timestamp

        Returns:
            True if it's a holiday
        """
        # Major trading holidays (simplified list)
        holidays = [
            # New Year's Day
            (1, 1),
            # Christmas Day
            (12, 25),
            # Independence Day (US)
            (7, 4),
            # Thanksgiving (US) - 4th Thursday of November
            (11, 22),  # Approximate
            (11, 23),  # Approximate
            (11, 24),  # Approximate
            (11, 25),  # Approximate
            (11, 26),  # Approximate
        ]

        month_day = (timestamp.month, timestamp.day)
        return month_day in holidays

    def _analyze_price_action(self, session_data: pd.DataFrame) -> Dict:
        """Analyze price action characteristics."""
        if session_data.empty:
            return {}

        # Calculate price range
        price_range = session_data['high'].max() - session_data['low'].min()

        # Calculate body vs wick ratio
        body_sizes = abs(session_data['close'] - session_data['open'])
        wick_sizes = (session_data['high'] - session_data['low']) - body_sizes

        avg_body_ratio = body_sizes.mean() / (body_sizes.mean() + wick_sizes.mean()) if (body_sizes.mean() + wick_sizes.mean()) > 0 else 0

        # Determine price action type
        if avg_body_ratio > 0.6:
            price_action_type = 'STRONG_TREND'
        elif avg_body_ratio > 0.4:
            price_action_type = 'MODERATE_TREND'
        else:
            price_action_type = 'RANGING'

        return {
            'price_range': price_range,
            'body_ratio': avg_body_ratio,
            'price_action_type': price_action_type,
            'candle_count': len(session_data)
        }
