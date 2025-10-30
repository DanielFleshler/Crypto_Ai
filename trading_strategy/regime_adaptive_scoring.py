"""
Regime-Adaptive Scoring Module

Implements market regime-specific adjustments for:
- Volume profile scoring (bullish vs bearish differences)
- Confidence adjustments based on trend strength
- Counter-trend setup detection and penalty

Fixes bearish market miscalibration issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class RegimeAdaptiveScoring:
    """
    Regime-adaptive scoring adjustments for different market conditions.
    
    Key insight: What works in bullish markets doesn't always work in bearish.
    - Bullish: High volume breakout = continuation (good)
    - Bearish: High volume breakout = potential capitulation/reversal (caution)
    """
    
    def __init__(self):
        """Initialize regime-adaptive scoring."""
        pass
    
    def calculate_regime_adaptive_volume_score(self, 
                                              relative_volume: float,
                                              htf_bias: str,
                                              signal_type: str,
                                              trend_strength: float = 0.5) -> Dict:
        """
        Calculate volume score with regime adaptation.
        
        FIXED BUG-VOLUME-BEARISH-001: Volume profile inverted in bearish markets
        
        Key Insights:
        - Bullish market + high volume = good (continuation)
        - Bearish market + very high volume = caution (potential capitulation/reversal)
        - Bearish market + moderate volume = good (controlled decline)
        
        Args:
            relative_volume: Volume relative to recent average
            htf_bias: HTF market bias ('BULLISH', 'BEARISH', 'NEUTRAL')
            signal_type: Signal direction ('BUY' or 'SELL')
            trend_strength: Strength of current trend (0-1)
            
        Returns:
            Dict with volume_score, adjustment_reason, and metadata
        """
        # Base volume score (unchanged logic)
        if relative_volume >= 2.0:
            base_score = 1.0
        elif relative_volume >= 1.5:
            base_score = 0.8
        elif relative_volume >= 1.0:
            base_score = 0.5
        else:
            base_score = 0.2
        
        # Apply regime-specific adjustments
        adjusted_score = base_score
        adjustment_reason = "neutral"
        
        # BULLISH MARKET
        if htf_bias == 'BULLISH':
            if signal_type == 'BUY':
                # Buying in bullish market - high volume is good (continuation)
                if relative_volume >= 1.5:
                    adjusted_score = base_score * 1.15  # Boost
                    adjustment_reason = "bullish_momentum_confirmation"
            else:  # SELL
                # Counter-trend short in bullish - penalize high volume (might be strong uptrend)
                if relative_volume >= 2.0:
                    adjusted_score = base_score * 0.85  # Penalty
                    adjustment_reason = "counter_trend_high_volume_penalty"
        
        # BEARISH MARKET (CRITICAL FIX)
        elif htf_bias == 'BEARISH':
            if signal_type == 'SELL':
                # Selling in bearish market
                if relative_volume >= 2.0:
                    # VERY high volume = potential capitulation/reversal
                    adjusted_score = base_score * 0.75  # Strong penalty
                    adjustment_reason = "bearish_capitulation_warning"
                elif 1.2 <= relative_volume < 2.0:
                    # Moderate-high volume = controlled decline (good)
                    adjusted_score = base_score * 1.1  # Boost
                    adjustment_reason = "bearish_controlled_decline"
                else:
                    # Low volume in bearish = weak trend (caution)
                    adjusted_score = base_score * 0.95
                    adjustment_reason = "bearish_weak_volume"
            else:  # BUY
                # Counter-trend long in bearish
                if relative_volume >= 2.0:
                    # Very high volume might be capitulation bottom
                    adjusted_score = base_score * 1.05  # Slight boost
                    adjustment_reason = "potential_capitulation_bottom"
                else:
                    # Normal counter-trend - neutral
                    pass
        
        # NEUTRAL MARKET
        else:  # NEUTRAL
            # In neutral markets, use base scoring
            pass
        
        # Apply trend strength modifier
        # Strong trends = trust volume more, weak trends = discount volume
        if trend_strength > 0.7:
            # Strong trend - trust volume signal
            adjusted_score = adjusted_score * 1.05
        elif trend_strength < 0.4:
            # Weak/choppy - discount volume
            adjusted_score = adjusted_score * 0.95
        
        # Cap at [0, 1]
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        return {
            'volume_score': adjusted_score,
            'base_score': base_score,
            'adjustment_factor': adjusted_score / base_score if base_score > 0 else 1.0,
            'adjustment_reason': adjustment_reason,
            'relative_volume': relative_volume,
            'regime': htf_bias
        }
    
    def apply_regime_confidence_adjustment(self,
                                          base_confidence: float,
                                          htf_bias: str,
                                          signal_type: str,
                                          trend_strength: float,
                                          confluence_count: int,
                                          volume_profile: Dict) -> Dict:
        """
        Apply regime-specific confidence adjustments.
        
        FIXED BUG-CONFIDENCE-BEARISH-001: Confidence inverted in bearish markets
        
        Key Adjustments:
        1. Counter-trend setups get penalty (especially in strong trends)
        2. Bearish markets are more unpredictable → slight conservatism
        3. Very high confluence in bearish + high volume → strong penalty (likely choppy)
        
        Args:
            base_confidence: Base confidence score (0-1)
            htf_bias: HTF market bias
            signal_type: Signal direction
            trend_strength: Trend strength (0-1)
            confluence_count: Number of confluence factors
            volume_profile: Volume profile data
            
        Returns:
            Dict with adjusted_confidence and reasoning
        """
        adjusted_confidence = base_confidence
        adjustments = []
        
        # 1. COUNTER-TREND PENALTY
        is_counter_trend = (
            (htf_bias == 'BULLISH' and signal_type == 'SELL') or
            (htf_bias == 'BEARISH' and signal_type == 'BUY')
        )
        
        if is_counter_trend:
            if trend_strength > 0.7:
                # Strong counter-trend setup = risky
                adjusted_confidence *= 0.85
                adjustments.append('strong_counter_trend_penalty')
            elif trend_strength > 0.5:
                # Moderate counter-trend
                adjusted_confidence *= 0.92
                adjustments.append('moderate_counter_trend_penalty')
        
        # 2. BEARISH MARKET CONSERVATISM (ENHANCED)
        if htf_bias == 'BEARISH':
            if signal_type == 'SELL':
                # Trend-following short in bearish market
                # Apply STRONGER conservatism (bearish markets more volatile and prone to reversals)
                adjusted_confidence *= 0.92  # Increased from 0.97
                adjustments.append('bearish_conservatism')
                
                # Check for bearish trap: high confluence + high volume
                volume_score = volume_profile.get('volume_score', 0.5)
                if confluence_count >= 5 and volume_score >= 0.85:
                    # Too perfect = might be trap (lowered threshold from 0.9)
                    adjusted_confidence *= 0.85  # Stronger penalty (was 0.90)
                    adjustments.append('bearish_perfection_trap')
                
                # ADDITIONAL: Check for excessive volume (capitulation risk)
                relative_volume = volume_profile.get('relative_volume', 1.0)
                if relative_volume >= 2.0:
                    # Very high volume in bearish = potential capitulation bottom
                    adjusted_confidence *= 0.88
                    adjustments.append('bearish_high_volume_caution')
            else:
                # Counter-trend long in bearish (already penalized above)
                pass
        
        # 3. BULLISH MARKET OPTIMIZATION
        elif htf_bias == 'BULLISH':
            if signal_type == 'BUY':
                # Trend-following long in bullish - boost confidence slightly
                if trend_strength > 0.6:
                    adjusted_confidence *= 1.03
                    adjustments.append('bullish_momentum_boost')
        
        # 4. WEAK TREND PENALTY (applies to all)
        if trend_strength < 0.4:
            # Choppy/ranging market - reduce confidence
            adjusted_confidence *= 0.93
            adjustments.append('weak_trend_penalty')
        
        # Cap at [0, 1]
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return {
            'adjusted_confidence': adjusted_confidence,
            'base_confidence': base_confidence,
            'adjustment_factor': adjusted_confidence / base_confidence if base_confidence > 0 else 1.0,
            'adjustments': adjustments,
            'is_counter_trend': is_counter_trend,
            'regime': htf_bias
        }
    
    def validate_counter_trend_fvg_logic(self,
                                        fvg_direction: str,
                                        htf_bias: str,
                                        signal_type: str,
                                        price_action_context: Dict) -> Dict:
        """
        Validate counter-trend FVG logic for bearish markets.
        
        FIXED BUG-FVG-BEARISH-001: Counter-trend FVG might cause bad entries
        
        Counter-trend FVG logic:
        - Bullish FVG in bearish market = resistance/rejection zone for SELL
        - Bearish FVG in bullish market = support/bounce zone for BUY
        
        But needs validation:
        - Are there too many bullish FVGs forming? (potential reversal)
        - Is price action showing reversal signs?
        
        Args:
            fvg_direction: 'BULLISH' or 'BEARISH'
            htf_bias: HTF market bias
            signal_type: Proposed signal type
            price_action_context: Recent price action data
            
        Returns:
            Dict with validation result and warnings
        """
        is_valid = True
        warnings = []
        confidence_penalty = 0.0
        
        # Check if FVG logic makes sense
        if htf_bias == 'BEARISH':
            if fvg_direction == 'BULLISH' and signal_type == 'SELL':
                # Bullish FVG as resistance in bearish market - standard logic
                
                # Check for potential reversal signs
                recent_fvg_count = price_action_context.get('recent_bullish_fvg_count', 0)
                if recent_fvg_count >= 3:
                    # Many bullish FVGs = market trying to reverse
                    warnings.append('multiple_bullish_fvgs_reversal_risk')
                    confidence_penalty = 0.15  # 15% penalty
                
                price_momentum = price_action_context.get('price_momentum', 0)
                if price_momentum > 0.02:  # 2% upward momentum
                    # Price moving up despite bearish bias
                    warnings.append('upward_momentum_in_bearish')
                    confidence_penalty = max(confidence_penalty, 0.10)
            
            elif fvg_direction == 'BEARISH' and signal_type == 'SELL':
                # Bearish FVG for continuation short - good
                pass
        
        elif htf_bias == 'BULLISH':
            if fvg_direction == 'BEARISH' and signal_type == 'BUY':
                # Bearish FVG as support in bullish market - standard logic
                
                # Check for reversal signs
                recent_fvg_count = price_action_context.get('recent_bearish_fvg_count', 0)
                if recent_fvg_count >= 3:
                    warnings.append('multiple_bearish_fvgs_reversal_risk')
                    confidence_penalty = 0.15
                
                price_momentum = price_action_context.get('price_momentum', 0)
                if price_momentum < -0.02:  # 2% downward momentum
                    warnings.append('downward_momentum_in_bullish')
                    confidence_penalty = max(confidence_penalty, 0.10)
        
        return {
            'is_valid': is_valid,
            'warnings': warnings,
            'confidence_penalty': confidence_penalty,
            'logic_type': 'counter_trend' if fvg_direction != htf_bias else 'trend_following'
        }
    
    def calculate_trend_strength(self, structures: list, lookback: int = 10) -> float:
        """
        Calculate trend strength from recent market structures.
        
        Args:
            structures: List of market structures
            lookback: Number of recent structures to analyze
            
        Returns:
            Trend strength (0-1)
        """
        if not structures:
            return 0.5  # Neutral
        
        recent = structures[-lookback:] if len(structures) > lookback else structures
        
        if len(recent) == 0:
            return 0.5
        
        # Count bullish vs bearish structures
        bullish_count = sum(1 for s in recent if hasattr(s, 'trend_direction') and s.trend_direction == 'BULLISH')
        bearish_count = sum(1 for s in recent if hasattr(s, 'trend_direction') and s.trend_direction == 'BEARISH')
        
        total = len(recent)
        max_directional = max(bullish_count, bearish_count)
        
        # Strength = how consistently directional
        return max_directional / total if total > 0 else 0.5

