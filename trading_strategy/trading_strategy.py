import pandas as pd
from typing import List, Dict
from .data_loader import DataLoader
from .market_structure import MarketStructureDetector
from .ict_concepts import ICTConceptsDetector
from .elliott_wave import ElliottWaveDetector
from .kill_zones import KillZoneDetector
from .data_structures import Signal

class TradingStrategy:
    def __init__(self, base_path: str):
        self.data_loader = DataLoader(base_path)
        self.market_structure = MarketStructureDetector()
        self.ict_detector = ICTConceptsDetector()
        self.elliott_detector = ElliottWaveDetector()
        self.killzone_detector = KillZoneDetector()
        self.htf_bias='NEUTRAL'

    def analyze_htf_bias(self, df):
        swing = self.market_structure.detect_swing_points(df)
        structs = self.market_structure.detect_market_structure(swing)
        return self.market_structure.get_current_bias(structs)

    def analyze_mtf_structure(self, df):
        dfz=self.killzone_detector.mark_kill_zones(df)
        swing=self.market_structure.detect_swing_points(dfz)
        structs=self.market_structure.detect_market_structure(swing)
        
        # Detect Elliott Wave sequences
        elliott_sequences = []
        wave1_candidates = self.elliott_detector.identify_wave_1(dfz, swing)
        
        for wave1 in wave1_candidates[:5]:  # Check only first 5
            wave2 = self.elliott_detector.identify_wave_2(dfz, wave1, swing)
            if wave2:
                wave3 = self.elliott_detector.identify_wave_3(dfz, wave1, wave2, swing)
                sequence = [wave1, wave2]
                if wave3:
                    sequence.append(wave3)
                if self.elliott_detector.validate_elliott_wave_sequence(sequence):
                    elliott_sequences.append(sequence)
        
        return {
            'swing_points':swing,'structures':structs,
            'fvgs':self.ict_detector.detect_fvg(dfz),
            'order_blocks':self.ict_detector.detect_order_blocks(dfz,swing),
            'ote_zones':self.ict_detector.detect_ote_zones(dfz,swing),
            'elliott_sequences':elliott_sequences
        }

    def detect_wave2_to_wave3_entries(self, mtf)->List[Signal]:
        """Detect entries from Wave 2 to Wave 3"""
        signals = []
        try:
            swing_df = mtf.get('swing_points')
            if swing_df is None or len(swing_df) == 0:
                return signals
                
            # Look for Elliott wave sequences
            elliott_sequences = mtf.get('elliott_sequences', [])
            if not elliott_sequences:
                return signals
                
            for sequence in elliott_sequences:
                if len(sequence) >= 2:  # Need at least wave 1 and 2
                    wave1, wave2 = sequence[0], sequence[1]
                    
                    # Create signal for wave 2 to wave 3 entry
                    entry_price = wave2.end_price
                    
                    # Determine if this is a bullish or bearish pattern
                    is_bullish = wave1.start_price < wave1.end_price
                    
                    if is_bullish:
                        # Bullish pattern: BUY signal
                        signal_type = 'BUY'
                        stop_loss = entry_price * 0.98  # 2% below entry
                        take_profit1 = entry_price + (entry_price - stop_loss) * 2
                        take_profit2 = entry_price + (entry_price - stop_loss) * 3
                    else:
                        # Bearish pattern: SELL signal
                        signal_type = 'SELL'
                        stop_loss = entry_price * 1.02  # 2% above entry
                        take_profit1 = entry_price - (stop_loss - entry_price) * 2
                        take_profit2 = entry_price - (stop_loss - entry_price) * 3
                    
                    signal = Signal(
                        timestamp=wave2.end_time,
                        signal_type=signal_type,
                        entry_type='WAVE2_TO_WAVE3',
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profits=[take_profit1, take_profit2],
                        risk_reward=2.0,
                        confidence=0.8,
                        metadata={'wave_sequence': [wave1, wave2]}
                    )
                    signals.append(signal)
        except Exception as e:
            print(f"Error in detect_wave2_to_wave3_entries: {e}")
        return signals
    
    def detect_wave3_continuation_entries(self, mtf)->List[Signal]:
        """Detect Wave 3 continuation entries - Skip for now as logic is complex"""
        signals = []
        try:
            # Skip WAVE3_CONTINUATION for now as it needs more sophisticated logic
            # This should look for pullbacks within wave 3, not just at the end
            pass
        except Exception as e:
            print(f"Error in detect_wave3_continuation_entries: {e}")
        return signals
    
    def detect_wave4_to_wave5_entries(self, mtf)->List[Signal]:
        """Detect entries from Wave 4 to Wave 5"""
        signals = []
        try:
            swing_df = mtf.get('swing_points')
            if swing_df is None or len(swing_df) == 0:
                return signals
                
            elliott_sequences = mtf.get('elliott_sequences', [])
            for sequence in elliott_sequences:
                if len(sequence) >= 4:  # Need waves 1-4
                    wave1, wave2, wave3, wave4 = sequence[0], sequence[1], sequence[2], sequence[3]
                    
                    # Create signal for wave 4 to wave 5 entry
                    entry_price = wave4.end_price
                    stop_loss = wave3.end_price * 0.98
                    take_profit1 = entry_price + (entry_price - stop_loss) * 1.2
                    take_profit2 = entry_price + (entry_price - stop_loss) * 2.0
                    
                    signal = Signal(
                        timestamp=wave4.end_time,
                        signal_type='BUY',
                        entry_type='WAVE4_TO_WAVE5',
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profits=[take_profit1, take_profit2],
                        risk_reward=1.2,
                        confidence=0.6,
                        metadata={'wave_sequence': [wave1, wave2, wave3, wave4]}
                    )
                    signals.append(signal)
        except Exception as e:
            print(f"Error in detect_wave4_to_wave5_entries: {e}")
        return signals
    
    def detect_reversal_after_wave5_entries(self, mtf)->List[Signal]:
        """Detect reversal entries after Wave 5"""
        signals = []
        try:
            swing_df = mtf.get('swing_points')
            if swing_df is None or len(swing_df) == 0:
                return signals
                
            elliott_sequences = mtf.get('elliott_sequences', [])
            for sequence in elliott_sequences:
                if len(sequence) >= 5:  # Complete 5-wave sequence
                    wave5 = sequence[4]
                    
                    # Create signal for reversal after wave 5
                    entry_price = wave5.end_price * 0.99  # Slight pullback
                    stop_loss = wave5.end_price * 1.02
                    take_profit1 = entry_price - (stop_loss - entry_price) * 1.5
                    take_profit2 = entry_price - (stop_loss - entry_price) * 2.5
                    
                    signal = Signal(
                        timestamp=wave5.end_time,
                        signal_type='SELL',
                        entry_type='REVERSAL_AFTER_WAVE5',
                        price=entry_price,
                        stop_loss=stop_loss,
                        take_profits=[take_profit1, take_profit2],
                        risk_reward=1.5,
                        confidence=0.7,
                        metadata={'wave_sequence': sequence}
                    )
                    signals.append(signal)
        except Exception as e:
            print(f"Error in detect_reversal_after_wave5_entries: {e}")
        return signals
    
    def detect_wave_c_entries(self, mtf)->List[Signal]:
        """Detect Wave C entries - This should be for corrective patterns, not impulse"""
        signals = []
        try:
            swing_df = mtf.get('swing_points')
            if swing_df is None or len(swing_df) == 0:
                return signals
                
            # For now, skip WAVE_C entries as they need different logic
            # This should be for ABC corrective patterns, not 1-2-3 impulse patterns
            pass
        except Exception as e:
            print(f"Error in detect_wave_c_entries: {e}")
        return signals

    def generate_signals(self, htf, mtf)->List[Signal]:
        sigs=[]
        sigs+=self.detect_wave2_to_wave3_entries(mtf)
        sigs+=self.detect_wave3_continuation_entries(mtf)
        sigs+=self.detect_wave4_to_wave5_entries(mtf)
        sigs+=self.detect_reversal_after_wave5_entries(mtf)
        sigs+=self.detect_wave_c_entries(mtf)
        return sigs

    def run_analysis(self,pair,sd,ed):
        try:
            data=self.data_loader.load_pair_data(pair,['1h','15m'],sd,ed)
            
            # Validate data before processing
            if not data or '1h' not in data or '15m' not in data:
                print(f"Warning: No data available for {pair} in date range {sd} to {ed}")
                return {'signals':[],'htf_bias':'NEUTRAL'}
                
            if len(data['1h']) < 10 or len(data['15m']) < 10:
                print(f"Warning: Insufficient data for {pair} (HTF: {len(data['1h'])}, MTF: {len(data['15m'])})")
                return {'signals':[],'htf_bias':'NEUTRAL'}
            
            self.htf_bias=self.analyze_htf_bias(data['1h'])
            mtf=self.analyze_mtf_structure(data['15m'])
            signals=self.generate_signals({},mtf)
            return {'signals':signals,'htf_bias':self.htf_bias}
        except Exception as e:
            print(f"Error in run_analysis for {pair}: {e}")
            return {'signals':[],'htf_bias':'NEUTRAL'}
