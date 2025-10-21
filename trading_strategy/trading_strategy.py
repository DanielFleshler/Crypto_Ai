# Phase 1.6: Main TradingStrategy Class (Orchestrator)
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

# Import all required classes
from .data_loader import DataLoader
from .market_structure import MarketStructureDetector, MarketStructure
from .ict_concepts import ICTConceptsDetector, ICTConcept
from .elliott_wave import ElliottWaveDetector, ElliottWave
from .kill_zones import KillZoneDetector, KillZone

@dataclass
class Signal:
    """Trading signal with entry/exit points and risk management"""
    timestamp: datetime
    signal_type: str  # 'BUY' or 'SELL'
    entry_type: str
    price: float
    confidence: float
    stop_loss: float
    take_profits: List[float]
    risk_reward: float
    metadata: Dict

class TradingStrategy:
    """
    Main orchestrator class that combines all components:
    - Elliott Wave analysis
    - ICT concepts detection  
    - Market structure analysis
    - Kill zone timing
    - Multi-timeframe coordination
    """
    
    def __init__(self, base_path: str = None):
        # Initialize all components
        self.data_loader = DataLoader(base_path) if base_path else None
        self.market_structure = MarketStructureDetector()
        self.ict_detector = ICTConceptsDetector()
        self.elliott_detector = ElliottWaveDetector()
        self.killzone_detector = KillZoneDetector()
        
        # Strategy state
        self.current_pair = None
        self.htf_bias = 'NEUTRAL'  # Higher timeframe bias
        self.active_signals = []
        self.performance_metrics = {}
        
        # Timeframe hierarchy for MTF analysis
        self.htf_timeframes = ['1d', '4h', '1h']      # Higher timeframe (bias)
        self.mtf_timeframes = ['15m', '5m']           # Medium timeframe (structure)
        self.ltf_timeframes = ['1m']                  # Lower timeframe (entry) - not available in your data
        
    def analyze_htf_bias(self, df_htf: pd.DataFrame) -> str:
        """
        Determine Higher Timeframe bias from daily/4H/1H charts
        
        Args:
            df_htf: Higher timeframe DataFrame
            
        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        # Detect swing points and market structure
        swing_df = self.market_structure.detect_swing_points(df_htf)
        structures = self.market_structure.detect_market_structure(swing_df)
        
        # Get current bias from structure analysis
        bias = self.market_structure.get_current_bias(structures)
        
        # Additional confirmation from recent price action
        recent_data = df_htf.tail(20)
        
        # Simple trend confirmation using moving averages
        if len(recent_data) >= 20:
            sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
            current_price = recent_data['close'].iloc[-1]
            
            if current_price > sma_20 and bias in ['BULLISH', 'NEUTRAL']:
                return 'BULLISH'
            elif current_price < sma_20 and bias in ['BEARISH', 'NEUTRAL']:
                return 'BEARISH'
        
        return bias
    
    def analyze_mtf_structure(self, df_mtf: pd.DataFrame) -> Dict:
        """
        Analyze Medium Timeframe for entry opportunities
        
        Args:
            df_mtf: Medium timeframe DataFrame (15m/5m)
            
        Returns:
            Dict with structure analysis results
        """
        # Add kill zones
        df_with_zones = self.killzone_detector.mark_kill_zones(df_mtf)
        
        # Detect swing points and market structure
        swing_df = self.market_structure.detect_swing_points(df_with_zones)
        structures = self.market_structure.detect_market_structure(swing_df)
        
        # Detect ICT concepts
        fvgs = self.ict_detector.detect_fvg(df_with_zones)
        order_blocks = self.ict_detector.detect_order_blocks(df_with_zones, swing_df)
        breaker_blocks = self.ict_detector.detect_breaker_blocks(df_with_zones, order_blocks)
        ote_zones = self.ict_detector.detect_ote_zones(df_with_zones, swing_df)
        
        # Detect Elliott Waves
        wave1_candidates = self.elliott_detector.identify_wave_1(df_with_zones, swing_df)
        
        elliott_sequences = []
        for wave1 in wave1_candidates:
            wave2 = self.elliott_detector.identify_wave_2(df_with_zones, wave1, swing_df)
            if wave2:
                wave3 = self.elliott_detector.identify_wave_3(df_with_zones, wave1, wave2, swing_df)
                if wave3:
                    sequence = [wave1, wave2, wave3]
                    if self.elliott_detector.validate_elliott_wave_sequence(sequence):
                        elliott_sequences.append(sequence)
        
        return {
            'swing_points': swing_df,
            'structures': structures,
            'fvgs': fvgs,
            'order_blocks': order_blocks,
            'breaker_blocks': breaker_blocks,
            'ote_zones': ote_zones,
            'elliott_sequences': elliott_sequences,
            'kill_zones': df_with_zones
        }
    
    def generate_signals(self, htf_analysis: Dict, mtf_analysis: Dict) -> List[Signal]:
        """
        Generate trading signals based on Elliott Wave + ICT combination
        
        Args:
            htf_analysis: Higher timeframe analysis results
            mtf_analysis: Medium timeframe analysis results  
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        # Only generate signals if HTF bias is clear
        if self.htf_bias == 'NEUTRAL':
            return signals
        
        # Call all entry detection methods
        signals.extend(self.detect_wave2_to_wave3_entries(mtf_analysis))
        signals.extend(self.detect_wave3_continuation_entries(mtf_analysis))
        signals.extend(self.detect_wave4_to_wave5_entries(mtf_analysis))
        signals.extend(self.detect_reversal_after_wave5_entries(mtf_analysis))
        signals.extend(self.detect_wave_c_entries(mtf_analysis))
        
        # ICT-based entries (don't require Elliott waves)
        signals.extend(self.detect_fvg_entries(mtf_analysis))
        signals.extend(self.detect_order_block_entries(mtf_analysis))
        signals.extend(self.detect_ote_entries(mtf_analysis))
        
        return signals
    
    def detect_wave2_to_wave3_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect Wave 2 to Wave 3 entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for Wave 2 end entries
        """
        signals = []
        elliott_sequences = mtf_analysis.get("elliott_sequences", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        ote_zones = mtf_analysis.get("ote_zones", [])
        
        for sequence in elliott_sequences:
            if len(sequence) >= 2:  # Have Wave 1 and Wave 2
                wave1, wave2 = sequence[0], sequence[1]
                
                # Get price and Fibonacci levels
                price = wave2.end_price
                fibs = wave2.fibonacci_levels
                
                # Skip if price not between fib_0.382 and fib_0.786
                if not (fibs.get("fib_0.382", 0) <= price <= fibs.get("fib_0.786", float('inf'))):
                    continue
                
                # Compute confluence
                confluence = self._find_entry_confluence(price, fvgs, order_blocks, ote_zones)
                
                # Skip if confluence score < 0.7
                if confluence['score'] < 0.7:
                    continue
                
                # Compute stop loss
                if self.htf_bias == 'BULLISH':
                    stop_loss = fibs.get("fib_0.786", price) * 0.995
                else:  # BEARISH
                    stop_loss = fibs.get("fib_0.786", price) * 1.005
                
                # Compute take profits at extensions
                wave1_length = abs(wave1.end_price - wave1.start_price)
                extensions = [1.20, 1.414, 1.618, 2.272]
                take_profits = []
                
                for ext in extensions:
                    if self.htf_bias == 'BULLISH':
                        tp = price + (wave1_length * ext)
                    else:  # BEARISH
                        tp = price - (wave1_length * ext)
                    take_profits.append(tp)
                
                # Compute risk reward
                risk_reward = self._calculate_risk_reward(price, stop_loss, take_profits[0])
                
                # Create signal
                signal = Signal(
                    timestamp=wave2.end_time,
                    signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                    entry_type="WAVE_2_END",
                    price=price,
                    confidence=confluence['score'],
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_reward=risk_reward,
                    metadata={
                        'elliott_sequence': sequence,
                        'confluence': confluence,
                        'fibonacci_levels': fibs
                    }
                )
                signals.append(signal)
        
        return signals
    
    def detect_wave3_continuation_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect Wave 3 continuation entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for Wave 3 continuation entries
        """
        signals = []
        elliott_sequences = mtf_analysis.get("elliott_sequences", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        
        for sequence in elliott_sequences:
            if len(sequence) >= 3:  # Have Wave 1, 2, and 3
                wave1, wave2, wave3 = sequence[0], sequence[1], sequence[2]
                
                # Look for continuation after Wave 3
                current_price = wave3.end_price
                
                # Check for confluence at current price
                confluence = self._find_entry_confluence(current_price, fvgs, order_blocks, [])
                
                if confluence['score'] >= 0.6:  # Lower threshold for continuation
                    # Calculate targets based on Wave 3 extensions
                    wave3_length = abs(wave3.end_price - wave3.start_price)
                    extensions = [1.272, 1.414, 1.618]
                    take_profits = []
                    
                    for ext in extensions:
                        if self.htf_bias == 'BULLISH':
                            tp = current_price + (wave3_length * ext)
                        else:
                            tp = current_price - (wave3_length * ext)
                        take_profits.append(tp)
                    
                    # Stop loss at Wave 3 start
                    stop_loss = wave3.start_price
                    risk_reward = self._calculate_risk_reward(current_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 1.5:  # Minimum R:R
                        signal = Signal(
                            timestamp=wave3.end_time,
                            signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                            entry_type="WAVE_3_CONTINUATION",
                            price=current_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'elliott_sequence': sequence, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def detect_wave4_to_wave5_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect Wave 4 to Wave 5 entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for Wave 4 to Wave 5 entries
        """
        signals = []
        elliott_sequences = mtf_analysis.get("elliott_sequences", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        
        for sequence in elliott_sequences:
            if len(sequence) >= 4:  # Have Wave 1, 2, 3, and 4
                wave1, wave2, wave3, wave4 = sequence[0], sequence[1], sequence[2], sequence[3]
                
                # Look for Wave 5 entry after Wave 4
                current_price = wave4.end_price
                
                # Check for confluence at current price
                confluence = self._find_entry_confluence(current_price, fvgs, order_blocks, [])
                
                if confluence['score'] >= 0.7:
                    # Calculate targets based on Wave 1 length (Wave 5 often equals Wave 1)
                    wave1_length = abs(wave1.end_price - wave1.start_price)
                    take_profits = [current_price + wave1_length] if self.htf_bias == 'BULLISH' else [current_price - wave1_length]
                    
                    # Stop loss at Wave 4 start
                    stop_loss = wave4.start_price
                    risk_reward = self._calculate_risk_reward(current_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 2.0:
                        signal = Signal(
                            timestamp=wave4.end_time,
                            signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                            entry_type="WAVE_4_TO_WAVE_5",
                            price=current_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'elliott_sequence': sequence, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def detect_reversal_after_wave5_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect reversal after Wave 5 entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for reversal after Wave 5 entries
        """
        signals = []
        elliott_sequences = mtf_analysis.get("elliott_sequences", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        
        for sequence in elliott_sequences:
            if len(sequence) >= 5:  # Have complete 5-wave sequence
                wave1, wave2, wave3, wave4, wave5 = sequence[0], sequence[1], sequence[2], sequence[3], sequence[4]
                
                # Look for reversal after Wave 5 completion
                current_price = wave5.end_price
                
                # Check for confluence at current price
                confluence = self._find_entry_confluence(current_price, fvgs, order_blocks, [])
                
                if confluence['score'] >= 0.8:  # High confluence for reversal
                    # Calculate targets based on Wave 1 length (typical ABC correction)
                    wave1_length = abs(wave1.end_price - wave1.start_price)
                    take_profits = [current_price - wave1_length] if self.htf_bias == 'BULLISH' else [current_price + wave1_length]
                    
                    # Stop loss beyond Wave 5 end
                    stop_loss = wave5.end_price * (1.02 if self.htf_bias == 'BULLISH' else 0.98)
                    risk_reward = self._calculate_risk_reward(current_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 1.5:
                        signal = Signal(
                            timestamp=wave5.end_time,
                            signal_type="SELL" if self.htf_bias == 'BULLISH' else "BUY",  # Opposite direction for reversal
                            entry_type="REVERSAL_AFTER_WAVE_5",
                            price=current_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'elliott_sequence': sequence, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def detect_wave_c_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect Wave C entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for Wave C entries
        """
        signals = []
        elliott_sequences = mtf_analysis.get("elliott_sequences", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        
        # Look for ABC corrective patterns
        for sequence in elliott_sequences:
            if len(sequence) >= 3:  # Have at least ABC
                # Check if this could be an ABC correction
                wave_a, wave_b, wave_c = sequence[0], sequence[1], sequence[2]
                
                # Wave C should be in same direction as Wave A
                if (wave_a.end_price > wave_a.start_price) == (wave_c.end_price > wave_c.start_price):
                    current_price = wave_c.end_price
                    
                    # Check for confluence
                    confluence = self._find_entry_confluence(current_price, fvgs, order_blocks, [])
                    
                    if confluence['score'] >= 0.7:
                        # Calculate targets based on Wave A length
                        wave_a_length = abs(wave_a.end_price - wave_a.start_price)
                        take_profits = [current_price + wave_a_length] if self.htf_bias == 'BULLISH' else [current_price - wave_a_length]
                        
                        # Stop loss at Wave C start
                        stop_loss = wave_c.start_price
                        risk_reward = self._calculate_risk_reward(current_price, stop_loss, take_profits[0])
                        
                        if risk_reward >= 2.0:
                            signal = Signal(
                                timestamp=wave_c.end_time,
                                signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                                entry_type="WAVE_C",
                                price=current_price,
                                confidence=confluence['score'],
                                stop_loss=stop_loss,
                                take_profits=take_profits,
                                risk_reward=risk_reward,
                                metadata={'elliott_sequence': sequence, 'confluence': confluence}
                            )
                            signals.append(signal)
        
        return signals
    
    def detect_fvg_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect FVG (Fair Value Gap) entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for FVG entries
        """
        signals = []
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        ote_zones = mtf_analysis.get("ote_zones", [])
        
        for fvg in fvgs[-10:]:  # Last 10 FVGs
            # Check if FVG aligns with HTF bias
            if (self.htf_bias == 'BULLISH' and fvg.concept_type == 'FVG_BULLISH') or \
               (self.htf_bias == 'BEARISH' and fvg.concept_type == 'FVG_BEARISH'):
                
                entry_price = (fvg.start_price + fvg.end_price) / 2
                
                # Check for confluence
                confluence = self._find_entry_confluence(entry_price, fvgs, order_blocks, ote_zones)
                
                if confluence['score'] >= 0.6:
                    # Calculate stop loss and targets
                    if self.htf_bias == 'BULLISH':
                        stop_loss = fvg.start_price * 0.995  # Below FVG
                        take_profits = [fvg.end_price * 1.02, fvg.end_price * 1.05]
                    else:
                        stop_loss = fvg.end_price * 1.005  # Above FVG
                        take_profits = [fvg.start_price * 0.98, fvg.start_price * 0.95]
                    
                    risk_reward = self._calculate_risk_reward(entry_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 1.5:
                        signal = Signal(
                            timestamp=fvg.timestamp,
                            signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                            entry_type="FVG_ENTRY",
                            price=entry_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'fvg': fvg, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def detect_order_block_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect Order Block entry opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for Order Block entries
        """
        signals = []
        order_blocks = mtf_analysis.get("order_blocks", [])
        fvgs = mtf_analysis.get("fvgs", [])
        ote_zones = mtf_analysis.get("ote_zones", [])
        
        for ob in order_blocks[-10:]:  # Last 10 Order Blocks
            # Check if OB aligns with HTF bias
            if (self.htf_bias == 'BULLISH' and ob.concept_type == 'OB_BULLISH') or \
               (self.htf_bias == 'BEARISH' and ob.concept_type == 'OB_BEARISH'):
                
                entry_price = (ob.start_price + ob.end_price) / 2
                
                # Check for confluence
                confluence = self._find_entry_confluence(entry_price, fvgs, order_blocks, ote_zones)
                
                if confluence['score'] >= 0.7:  # Higher threshold for OBs
                    # Calculate stop loss and targets
                    if self.htf_bias == 'BULLISH':
                        stop_loss = ob.start_price * 0.995  # Below OB
                        take_profits = [ob.end_price * 1.03, ob.end_price * 1.06, ob.end_price * 1.10]
                    else:
                        stop_loss = ob.end_price * 1.005  # Above OB
                        take_profits = [ob.start_price * 0.97, ob.start_price * 0.94, ob.start_price * 0.90]
                    
                    risk_reward = self._calculate_risk_reward(entry_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 2.0:  # Higher R:R for OBs
                        signal = Signal(
                            timestamp=ob.timestamp,
                            signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                            entry_type="ORDER_BLOCK_ENTRY",
                            price=entry_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'order_block': ob, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def detect_ote_entries(self, mtf_analysis: Dict) -> List[Signal]:
        """
        Detect OTE (Optimal Trade Entry) zone opportunities
        
        Args:
            mtf_analysis: Medium timeframe analysis results
            
        Returns:
            List of Signal objects for OTE entries
        """
        signals = []
        ote_zones = mtf_analysis.get("ote_zones", [])
        fvgs = mtf_analysis.get("fvgs", [])
        order_blocks = mtf_analysis.get("order_blocks", [])
        
        for ote in ote_zones[-5:]:  # Last 5 OTE zones
            # Check if OTE aligns with HTF bias
            if (self.htf_bias == 'BULLISH' and ote.concept_type == 'OTE_BULLISH') or \
               (self.htf_bias == 'BEARISH' and ote.concept_type == 'OTE_BEARISH'):
                
                entry_price = (ote.start_price + ote.end_price) / 2
                
                # Check for confluence
                confluence = self._find_entry_confluence(entry_price, fvgs, order_blocks, ote_zones)
                
                if confluence['score'] >= 0.8:  # Highest threshold for OTE
                    # Calculate stop loss and targets
                    if self.htf_bias == 'BULLISH':
                        stop_loss = ote.start_price * 0.998  # Tight stop for OTE
                        take_profits = [ote.end_price * 1.05, ote.end_price * 1.10, ote.end_price * 1.15]
                    else:
                        stop_loss = ote.end_price * 1.002  # Tight stop for OTE
                        take_profits = [ote.start_price * 0.95, ote.start_price * 0.90, ote.start_price * 0.85]
                    
                    risk_reward = self._calculate_risk_reward(entry_price, stop_loss, take_profits[0])
                    
                    if risk_reward >= 3.0:  # Very high R:R for OTE
                        signal = Signal(
                            timestamp=ote.timestamp,
                            signal_type="BUY" if self.htf_bias == 'BULLISH' else "SELL",
                            entry_type="OTE_ENTRY",
                            price=entry_price,
                            confidence=confluence['score'],
                            stop_loss=stop_loss,
                            take_profits=take_profits,
                            risk_reward=risk_reward,
                            metadata={'ote_zone': ote, 'confluence': confluence}
                        )
                        signals.append(signal)
        
        return signals
    
    def _find_entry_confluence(self, price: float, fvgs: List, order_blocks: List, 
                              ote_zones: List) -> Dict:
        """
        Find confluence of ICT concepts at entry price
        
        Returns:
            Dict with confluence score and supporting concepts
        """
        confluence_score = 0.0
        supporting_concepts = []
        
        # Check for FVG confluence
        for fvg in fvgs[-5:]:  # Last 5 FVGs
            if fvg.start_price <= price <= fvg.end_price:
                confluence_score += 0.3
                supporting_concepts.append(f"FVG_{fvg.concept_type}")
        
        # Check for Order Block confluence
        for ob in order_blocks[-5:]:  # Last 5 OBs
            if ob.start_price <= price <= ob.end_price:
                confluence_score += 0.4
                supporting_concepts.append(f"OB_{ob.concept_type}")
        
        # Check for OTE confluence
        for ote in ote_zones[-3:]:  # Last 3 OTE zones
            if ote.start_price <= price <= ote.end_price:
                confluence_score += 0.5
                supporting_concepts.append(f"OTE_{ote.concept_type}")
        
        return {
            'score': min(confluence_score, 1.0),  # Cap at 1.0
            'concepts': supporting_concepts
        }
    
    def _calculate_wave2_stop(self, wave1: ElliottWave, wave2: ElliottWave) -> float:
        """
        Calculate stop loss for Wave 2 end entry
        Stop should be beyond 78.6% retracement (invalidation level)
        """
        fib_786 = wave2.fibonacci_levels.get('fib_0.786', wave1.start_price)
        
        # Add small buffer beyond 78.6% level
        if wave1.start_price < wave1.end_price:  # Bullish
            return fib_786 * 0.995  # 0.5% below 78.6%
        else:  # Bearish
            return fib_786 * 1.005  # 0.5% above 78.6%
    
    def _calculate_wave3_targets(self, wave1: ElliottWave, wave2: ElliottWave) -> List[float]:
        """
        Calculate take profit targets for Wave 3
        Based on Fibonacci extensions: 120%, 141.4%, 161.8%, 227.2%
        """
        wave1_length = abs(wave1.end_price - wave1.start_price)
        
        targets = []
        multipliers = [1.20, 1.414, 1.618, 2.272]  # As per your strategy
        
        for mult in multipliers:
            if wave1.start_price < wave1.end_price:  # Bullish
                target = wave2.end_price + (wave1_length * mult)
            else:  # Bearish
                target = wave2.end_price - (wave1_length * mult)
            targets.append(target)
        
        return targets
    
    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk:reward ratio"""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def run_analysis(self, pair: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run complete multi-timeframe analysis
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            start_date: Analysis start date
            end_date: Analysis end date
            
        Returns:
            Complete analysis results
        """
        print(f"🚀 Running analysis for {pair}")
        print(f"📅 Period: {start_date} to {end_date}")
        
        self.current_pair = pair
        
        # Load data (placeholder - you'll implement actual loading)
        if self.data_loader:
            pair_data = self.data_loader.load_pair_data(
                pair, ['1d', '4h', '1h', '15m', '5m'], start_date, end_date
            )
        else:
            print("⚠️  DataLoader not initialized - using placeholder analysis")
            return {
                'pair': pair,
                'htf_bias': 'NEUTRAL',
                'signals': [],
                'status': 'DATA_LOADER_REQUIRED'
            }
        
        # HTF Analysis (1D/4H/1H)
        htf_data = pair_data.get('1h')  # Use 1H as primary HTF for now
        self.htf_bias = self.analyze_htf_bias(htf_data) if htf_data is not None else 'NEUTRAL'
        
        # MTF Analysis (15m/5m)
        mtf_data = pair_data.get('15m')  # Use 15m as primary MTF
        mtf_analysis = self.analyze_mtf_structure(mtf_data) if mtf_data is not None else {}
        
        # Generate signals
        signals = self.generate_signals({}, mtf_analysis)
        
        self.active_signals = signals
        
        return {
            'pair': pair,
            'htf_bias': self.htf_bias,
            'mtf_analysis': mtf_analysis,
            'signals': signals,
            'signal_count': len(signals),
            'status': 'ANALYSIS_COMPLETE'
        }

print("🎯 TradingStrategy Main Class Created")
print("✅ Multi-timeframe coordination")
print("✅ HTF bias determination") 
print("✅ MTF structure analysis")
print("✅ Signal generation with confluence")
print("✅ Elliott Wave + ICT integration")
print("🚀 Ready for complete analysis!")