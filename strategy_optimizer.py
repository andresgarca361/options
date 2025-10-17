"""
Advanced Strategy Optimizer

Dynamically generates and tests thousands of strategy combinations using:
- All probabilistic models (GBM, Jump-Diffusion, Heston, Fat-Tailed, Ensemble)
- Monte Carlo simulations for accurate probability distributions
- Advanced risk metrics (CVaR, Sharpe, Edge Score)
- Real options chain analysis

This is 100% probability-based - no AI price prediction, just statistical analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations, product
import pandas as pd

from advanced_options_engine import MarketData, AdvancedOptionsAnalyzer, MonteCarloEngine
from options_strategies import OptionsStrategy, StrategyLeg, StrategyMetrics


@dataclass
class StrategyCandidate:
    """A candidate strategy with its analysis results"""
    strategy_name: str
    legs: List[StrategyLeg]
    metrics: StrategyMetrics
    composite_score: float
    model_used: str
    

class AdvancedStrategyOptimizer:
    """
    Intelligent strategy optimizer that generates and tests thousands of combinations
    """
    
    def __init__(self, mc_engine: MonteCarloEngine = None):
        self.mc_engine = mc_engine or MonteCarloEngine(n_simulations=50000)
        self.strategy_calc = OptionsStrategy(self.mc_engine)
        self.models = ['ensemble', 'gbm', 'jump_diffusion', 'heston', 'fat_tailed']
    
    def generate_all_strategy_combinations(
        self,
        options_chain: List[Dict],
        current_price: float,
        max_legs: int = 4,
        min_liquidity: int = 5,  # Min open interest (lowered for more options)
        exhaustive: bool = True
    ) -> List[Tuple[str, List[StrategyLeg]]]:
        """
        Generate THOUSANDS of possible strategy combinations from options chain
        
        Strategy types generated (EXHAUSTIVE):
        - ALL single options (calls/puts, long/short)
        - ALL vertical spreads (every strike pair combination)
        - ALL straddles/strangles (every strike combination)
        - ALL butterflies (every 3-strike combination)
        - ALL iron condors (every 4-strike combination)
        - ALL ratio spreads (1x2, 1x3, 2x3 ratios)
        - ALL custom multi-leg combinations
        """
        
        # Filter for liquid options
        liquid_options = [
            opt for opt in options_chain 
            if opt.get('openInterest', 0) >= min_liquidity and opt.get('ask', 0) > 0
        ]
        
        if not liquid_options:
            return []
        
        # Separate calls and puts
        calls = [opt for opt in liquid_options if opt['type'] == 'call']
        puts = [opt for opt in liquid_options if opt['type'] == 'put']
        
        calls.sort(key=lambda x: x['strike'])
        puts.sort(key=lambda x: x['strike'])
        
        all_strategies = []
        
        # 1. ALL Single Options (Long AND Short)
        for opt in liquid_options:
            md = self._create_market_data(opt, current_price)
            # Long
            legs = [StrategyLeg(md, 1)]
            name = f"Long {opt['type'].capitalize()} ${opt['strike']:.0f}"
            all_strategies.append((name, legs))
            # Short
            legs = [StrategyLeg(md, -1)]
            name = f"Short {opt['type'].capitalize()} ${opt['strike']:.0f}"
            all_strategies.append((name, legs))
        
        # 2. ALL Vertical Spreads (Every strike pair)
        # Bull Call Spreads - ALL combinations
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                lower_md = self._create_market_data(calls[i], current_price)
                higher_md = self._create_market_data(calls[j], current_price)
                legs = [StrategyLeg(lower_md, 1), StrategyLeg(higher_md, -1)]
                name = f"Bull Call ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # Bear Call Spreads - ALL combinations
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                lower_md = self._create_market_data(calls[i], current_price)
                higher_md = self._create_market_data(calls[j], current_price)
                legs = [StrategyLeg(lower_md, -1), StrategyLeg(higher_md, 1)]
                name = f"Bear Call ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # Bull Put Spreads - ALL combinations
        for i in range(len(puts)):
            for j in range(i + 1, len(puts)):
                lower_md = self._create_market_data(puts[j], current_price)
                higher_md = self._create_market_data(puts[i], current_price)
                legs = [StrategyLeg(higher_md, -1), StrategyLeg(lower_md, 1)]
                name = f"Bull Put ${puts[j]['strike']:.0f}-${puts[i]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # Bear Put Spreads - ALL combinations
        for i in range(len(puts)):
            for j in range(i + 1, len(puts)):
                higher_md = self._create_market_data(puts[i], current_price)
                lower_md = self._create_market_data(puts[j], current_price)
                legs = [StrategyLeg(higher_md, 1), StrategyLeg(lower_md, -1)]
                name = f"Bear Put ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # 3. ALL Straddles (All strike combinations)
        for call in calls:
            matching_puts = [p for p in puts if abs(p['strike'] - call['strike']) < 1]
            for put in matching_puts:
                call_md = self._create_market_data(call, current_price)
                put_md = self._create_market_data(put, current_price)
                # Long straddle
                legs = [StrategyLeg(call_md, 1), StrategyLeg(put_md, 1)]
                name = f"Long Straddle ${call['strike']:.0f}"
                all_strategies.append((name, legs))
                # Short straddle
                legs = [StrategyLeg(call_md, -1), StrategyLeg(put_md, -1)]
                name = f"Short Straddle ${call['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # 4. ALL Strangles (Every call/put combination)
        for call in calls:
            for put in puts:
                call_md = self._create_market_data(call, current_price)
                put_md = self._create_market_data(put, current_price)
                # Long strangle
                legs = [StrategyLeg(call_md, 1), StrategyLeg(put_md, 1)]
                name = f"Long Strangle P${put['strike']:.0f}/C${call['strike']:.0f}"
                all_strategies.append((name, legs))
                # Short strangle
                legs = [StrategyLeg(call_md, -1), StrategyLeg(put_md, -1)]
                name = f"Short Strangle P${put['strike']:.0f}/C${call['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # 5. ALL Butterfly Spreads (All 3-leg combinations)
        for i in range(len(calls) - 2):
            for j in range(i + 1, len(calls) - 1):
                for k in range(j + 1, len(calls)):
                    lower_md = self._create_market_data(calls[i], current_price)
                    middle_md = self._create_market_data(calls[j], current_price)
                    higher_md = self._create_market_data(calls[k], current_price)
                    legs = [
                        StrategyLeg(lower_md, 1),
                        StrategyLeg(middle_md, -2),
                        StrategyLeg(higher_md, 1)
                    ]
                    name = f"Call Butterfly ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}-${calls[k]['strike']:.0f}"
                    all_strategies.append((name, legs))
        
        # Put butterflies
        for i in range(len(puts) - 2):
            for j in range(i + 1, len(puts) - 1):
                for k in range(j + 1, len(puts)):
                    lower_md = self._create_market_data(puts[i], current_price)
                    middle_md = self._create_market_data(puts[j], current_price)
                    higher_md = self._create_market_data(puts[k], current_price)
                    legs = [
                        StrategyLeg(lower_md, 1),
                        StrategyLeg(middle_md, -2),
                        StrategyLeg(higher_md, 1)
                    ]
                    name = f"Put Butterfly ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}-${puts[k]['strike']:.0f}"
                    all_strategies.append((name, legs))
        
        # 6. ALL Iron Condors (All 4-leg combinations)
        for i in range(len(puts) - 1):
            for j in range(i + 1, len(puts)):
                for k in range(len(calls) - 1):
                    for l in range(k + 1, len(calls)):
                        legs = [
                            StrategyLeg(self._create_market_data(puts[i], current_price), 1),
                            StrategyLeg(self._create_market_data(puts[j], current_price), -1),
                            StrategyLeg(self._create_market_data(calls[k], current_price), -1),
                            StrategyLeg(self._create_market_data(calls[l], current_price), 1)
                        ]
                        name = f"Iron Condor ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}-${calls[k]['strike']:.0f}-${calls[l]['strike']:.0f}"
                        all_strategies.append((name, legs))
        
        # 7. ALL Ratio Spreads (Multiple ratios)
        for i in range(len(calls) - 1):
            for j in range(i + 1, len(calls)):
                lower_md = self._create_market_data(calls[i], current_price)
                higher_md = self._create_market_data(calls[j], current_price)
                # 1x2 ratio
                legs = [StrategyLeg(lower_md, 1), StrategyLeg(higher_md, -2)]
                name = f"1x2 Call Ratio ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}"
                all_strategies.append((name, legs))
                # 1x3 ratio
                legs = [StrategyLeg(lower_md, 1), StrategyLeg(higher_md, -3)]
                name = f"1x3 Call Ratio ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}"
                all_strategies.append((name, legs))
                # 2x3 ratio
                legs = [StrategyLeg(lower_md, 2), StrategyLeg(higher_md, -3)]
                name = f"2x3 Call Ratio ${calls[i]['strike']:.0f}-${calls[j]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # Put ratios
        for i in range(len(puts) - 1):
            for j in range(i + 1, len(puts)):
                higher_md = self._create_market_data(puts[i], current_price)
                lower_md = self._create_market_data(puts[j], current_price)
                # 1x2 ratio
                legs = [StrategyLeg(higher_md, 1), StrategyLeg(lower_md, -2)]
                name = f"1x2 Put Ratio ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}"
                all_strategies.append((name, legs))
                # 1x3 ratio
                legs = [StrategyLeg(higher_md, 1), StrategyLeg(lower_md, -3)]
                name = f"1x3 Put Ratio ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}"
                all_strategies.append((name, legs))
                # 2x3 ratio
                legs = [StrategyLeg(higher_md, 2), StrategyLeg(lower_md, -3)]
                name = f"2x3 Put Ratio ${puts[i]['strike']:.0f}-${puts[j]['strike']:.0f}"
                all_strategies.append((name, legs))
        
        # 8. Iron Butterflies (All combinations)
        for i in range(len(puts)):
            for j in range(len(calls)):
                # Find matching puts/calls at same strike
                atm_strike_calls = [c for c in calls if abs(c['strike'] - puts[i]['strike']) < 1]
                atm_strike_puts = [p for p in puts if abs(p['strike'] - calls[j]['strike']) < 1]
                
                if atm_strike_calls and atm_strike_puts:
                    for atm_call in atm_strike_calls:
                        for atm_put in atm_strike_puts:
                            if abs(atm_call['strike'] - atm_put['strike']) < 1:  # Same strike
                                legs = [
                                    StrategyLeg(self._create_market_data(puts[i], current_price), 1),
                                    StrategyLeg(self._create_market_data(atm_put, current_price), -1),
                                    StrategyLeg(self._create_market_data(atm_call, current_price), -1),
                                    StrategyLeg(self._create_market_data(calls[j], current_price), 1)
                                ]
                                name = f"Iron Butterfly ${puts[i]['strike']:.0f}-${atm_call['strike']:.0f}-${calls[j]['strike']:.0f}"
                                all_strategies.append((name, legs))
        
        return all_strategies
    
    def _create_market_data(self, option: Dict, current_price: float) -> MarketData:
        """Convert option dict to MarketData object"""
        return MarketData(
            current_price=current_price,
            strike=option['strike'],
            time_to_expiry=option.get('days_to_expiry', 30) / 365.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            historical_volatility=option.get('historical_vol', 0.25),
            implied_volatility=option.get('implied_vol', 0.25),
            bid=option.get('bid', 0),
            ask=option.get('ask', 0),
            volume=option.get('volume', 0),
            open_interest=option.get('openInterest', 0),
            option_type=option['type']
        )
    
    def is_strategy_safe(self, metrics: StrategyMetrics) -> Tuple[bool, str]:
        """
        Validate strategy safety - reject dangerous strategies with extreme risk
        
        Returns: (is_safe, reason_if_unsafe)
        """
        max_profit = metrics.max_profit
        max_loss = abs(metrics.max_loss)
        risk_reward = metrics.risk_reward_ratio
        
        # Filter 1: Reject if risk/reward ratio is terrible (< 0.2:1)
        if risk_reward < 0.2 and max_loss > 0:
            return False, f"Terrible risk/reward ({risk_reward:.3f}:1 - need at least 0.2:1)"
        
        # Filter 2: Reject if max loss is > 10x max profit (extreme asymmetry)
        if max_loss > 0 and max_profit > 0:
            loss_to_profit_ratio = max_loss / max_profit
            if loss_to_profit_ratio > 10:
                return False, f"Extreme risk asymmetry (max loss ${max_loss:.2f} vs max profit ${max_profit:.2f})"
        
        # Filter 3: Reject if max loss is > $50 with tiny max profit (< $5)
        if max_loss > 50 and max_profit < 5:
            return False, f"Catastrophic risk (${max_loss:.2f} loss) for minimal reward (${max_profit:.2f})"
        
        # Filter 4: Reject if CVaR shows extreme tail risk relative to max profit
        if max_profit > 0 and metrics.cvar_95 < -max_profit * 5:
            return False, f"Severe tail risk (CVaR ${metrics.cvar_95:.2f} vs profit ${max_profit:.2f})"
        
        # Filter 5: Reject if expected value is significantly negative
        if metrics.expected_value < -abs(metrics.total_cost) * 0.5:
            return False, f"Expected value too negative (${metrics.expected_value:.2f})"
        
        return True, ""
    
    def calculate_composite_score(
        self,
        metrics: StrategyMetrics,
        preferences: Dict[str, float] = None
    ) -> float:
        """
        Calculate composite score for strategy ranking
        
        Default weights (can be customized):
        - Edge Score: 30%
        - Probability of Profit: 20%
        - Risk/Reward Ratio: 25% (increased from 20%)
        - Sharpe Ratio: 15%
        - Expected Value: 10%
        """
        if preferences is None:
            preferences = {
                'edge_score': 0.30,
                'probability_of_profit': 0.20,
                'risk_reward': 0.25,
                'sharpe_ratio': 0.15,
                'expected_value': 0.10
            }
        
        # Normalize components
        edge_norm = min(metrics.edge_score / 100, 1.0)
        pop_norm = min(metrics.probability_of_profit / 100, 1.0)
        
        # Risk/Reward with HEAVY penalty for poor ratios
        if metrics.risk_reward_ratio < 0.3:
            rr_norm = 0  # Zero score for terrible risk/reward
        elif metrics.risk_reward_ratio < 0.5:
            rr_norm = 0.2  # Very low score
        else:
            rr_norm = min(metrics.risk_reward_ratio / 5, 1.0)  # Cap at 5:1
        
        sharpe_norm = min(max(metrics.sharpe_ratio, 0) / 2, 1.0)  # Cap at 2
        
        # EV normalized by total cost (percentage return)
        ev_norm = 0
        if abs(metrics.total_cost) > 0:
            ev_pct = metrics.expected_value / abs(metrics.total_cost)
            ev_norm = min(max(ev_pct, -1), 1)  # Clamp to -1 to 1
            ev_norm = (ev_norm + 1) / 2  # Scale to 0-1
        
        score = (
            preferences['edge_score'] * edge_norm +
            preferences['probability_of_profit'] * pop_norm +
            preferences['risk_reward'] * rr_norm +
            preferences['sharpe_ratio'] * sharpe_norm +
            preferences['expected_value'] * ev_norm
        )
        
        return score * 100  # Scale to 0-100
    
    def optimize_strategies(
        self,
        options_chain: List[Dict],
        current_price: float,
        top_n: int = 3,
        model_type: str = 'ensemble',
        score_preferences: Dict[str, float] = None,
        max_strategies_to_test: int = 10000  # Test up to 10,000 strategies
    ) -> List[StrategyCandidate]:
        """
        Find optimal strategies using Monte Carlo and advanced models
        
        Returns top N strategies ranked by composite score
        """
        
        # Generate all possible strategies
        all_strategy_combos = self.generate_all_strategy_combinations(
            options_chain, current_price
        )
        
        if not all_strategy_combos:
            return []
        
        # Smart sampling if needed (only if we have way more than max)
        total_generated = len(all_strategy_combos)
        if total_generated > max_strategies_to_test:
            # Intelligent sampling: prioritize diversity across strategy types
            # Group by strategy type
            by_type = {}
            for name, legs in all_strategy_combos:
                strategy_type = name.split()[0]  # Get type from name
                if strategy_type not in by_type:
                    by_type[strategy_type] = []
                by_type[strategy_type].append((name, legs))
            
            # Sample proportionally from each type
            sampled = []
            per_type = max_strategies_to_test // len(by_type)
            for type_name, strategies in by_type.items():
                if len(strategies) <= per_type:
                    sampled.extend(strategies)
                else:
                    # Sample evenly from this type
                    step = len(strategies) / per_type
                    sampled.extend([strategies[int(i * step)] for i in range(per_type)])
            
            all_strategy_combos = sampled[:max_strategies_to_test]
        
        candidates = []
        
        # Test each strategy with the selected model
        rejected_count = 0
        for strategy_name, legs in all_strategy_combos:
            try:
                # Analyze with probabilistic model
                metrics = self.strategy_calc.analyze_strategy(
                    legs, strategy_name, model_type=model_type
                )
                
                # SAFETY CHECK: Filter out dangerous strategies
                is_safe, reason = self.is_strategy_safe(metrics)
                if not is_safe:
                    rejected_count += 1
                    continue
                
                # Calculate composite score
                score = self.calculate_composite_score(metrics, score_preferences)
                
                # Create candidate
                candidate = StrategyCandidate(
                    strategy_name=strategy_name,
                    legs=legs,
                    metrics=metrics,
                    composite_score=score,
                    model_used=model_type
                )
                
                candidates.append(candidate)
                
            except Exception as e:
                # Skip strategies that fail
                continue
        
        # Sort by composite score
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Return top N
        return candidates[:top_n]
    
    def multi_model_optimization(
        self,
        options_chain: List[Dict],
        current_price: float,
        top_n: int = 3,
        score_preferences: Dict[str, float] = None,
        max_strategies_to_test: int = 500
    ) -> Dict[str, List[StrategyCandidate]]:
        """
        Run optimization across ALL probabilistic models and find consensus
        
        Returns:
        - Top strategies from each model
        - Consensus top strategies (appear in multiple models)
        """
        
        results_by_model = {}
        all_candidates = []
        
        # Test with each model
        for model in self.models:
            candidates = self.optimize_strategies(
                options_chain,
                current_price,
                top_n=top_n * 2,  # Get more for consensus
                model_type=model,
                score_preferences=score_preferences,
                max_strategies_to_test=max_strategies_to_test
            )
            results_by_model[model] = candidates
            all_candidates.extend(candidates)
        
        # Find consensus strategies (appear in multiple models' top picks)
        strategy_votes = {}
        for candidate in all_candidates:
            if candidate.strategy_name not in strategy_votes:
                strategy_votes[candidate.strategy_name] = {
                    'count': 0,
                    'total_score': 0,
                    'candidate': candidate
                }
            strategy_votes[candidate.strategy_name]['count'] += 1
            strategy_votes[candidate.strategy_name]['total_score'] += candidate.composite_score
        
        # Calculate consensus score (votes * average score)
        consensus_strategies = []
        for name, data in strategy_votes.items():
            avg_score = data['total_score'] / data['count']
            consensus_score = data['count'] * avg_score  # More models = higher confidence
            
            candidate = data['candidate']
            candidate.composite_score = consensus_score
            consensus_strategies.append(candidate)
        
        # Sort consensus by score
        consensus_strategies.sort(key=lambda x: x.composite_score, reverse=True)
        
        results_by_model['consensus'] = consensus_strategies[:top_n]
        results_by_model['ensemble_only'] = results_by_model.get('ensemble', [])[:top_n]
        
        return results_by_model
    
    def classify_strategy_risk(self, strategy_name: str, legs: List[StrategyLeg]) -> Tuple[str, str]:
        """
        Classify strategy by risk type
        Returns: (risk_level, warning_message)
        """
        name_lower = strategy_name.lower()
        
        # Check for ratio spreads (can have unlimited risk)
        if 'ratio' in name_lower:
            # Count net short positions
            net_positions = {}
            for leg in legs:
                strike = leg.market_data.strike
                net_positions[strike] = net_positions.get(strike, 0) + leg.quantity
            
            # If we have net short positions, it's unlimited risk
            has_net_short = any(qty < 0 for qty in net_positions.values())
            if has_net_short:
                return "HIGH", "⚠️ RATIO SPREAD with unlimited risk if underlying moves significantly"
        
        # Check for naked options
        if len(legs) == 1 and legs[0].quantity < 0:
            return "HIGH", "⚠️ NAKED OPTION - Unlimited upside risk (call) or significant downside risk (put)"
        
        # Check for short straddles/strangles
        if 'short straddle' in name_lower or 'short strangle' in name_lower:
            return "HIGH", "⚠️ SHORT VOLATILITY strategy - Unlimited risk if underlying moves sharply"
        
        # Check for defined risk spreads
        if any(x in name_lower for x in ['bull', 'bear', 'butterfly', 'condor', 'vertical']):
            return "MODERATE", "✓ Defined risk strategy with limited maximum loss"
        
        # Long options
        if 'long' in name_lower and len(legs) <= 2:
            return "LOW", "✓ Limited risk strategy - max loss is premium paid"
        
        return "MODERATE", ""
    
    def explain_strategy(self, candidate: StrategyCandidate) -> str:
        """Generate human-readable explanation of strategy"""
        m = candidate.metrics
        
        # Classify strategy risk
        risk_level, risk_warning = self.classify_strategy_risk(
            candidate.strategy_name, 
            candidate.legs
        )
        
        explanation = f"""
**{candidate.strategy_name}**
"""
        
        # Add risk warning if applicable
        if risk_warning:
            explanation += f"\n{risk_warning}\n"
        
        explanation += f"""
**Why This Strategy:**
- Edge Score: {m.edge_score:.1f}/100 - Combined quality metric
- Probability of Profit: {m.probability_of_profit:.1f}%
- Expected Value: ${m.expected_value:.2f}

**Risk/Reward Profile:**
- Max Profit: ${m.max_profit:.2f}
- Max Loss: ${m.max_loss:.2f}
- Risk/Reward Ratio: {m.risk_reward_ratio:.2f}:1
- Worst-Case Loss (CVaR 95%): ${m.cvar_95:.2f}
- Risk Level: {risk_level}

**Position Details:**
"""
        
        for i, leg in enumerate(candidate.legs):
            direction = "LONG" if leg.quantity > 0 else "SHORT"
            explanation += f"\n- Leg {i+1}: {direction} {abs(leg.quantity)}x {leg.market_data.option_type.upper()} @ ${leg.market_data.strike:.0f}"
        
        if m.breakeven_points:
            be_str = ", ".join([f"${be:.2f}" for be in m.breakeven_points])
            explanation += f"\n\n**Breakeven Points:** {be_str}"
        
        explanation += f"\n\n**Model Used:** {candidate.model_used.upper()}"
        explanation += f"\n**Model Error Estimate:** ±{m.model_error_estimate*100:.1f}%"
        
        return explanation
