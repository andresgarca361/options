"""
Options Strategies Calculator

Supports multiple option strategies:
- Single options (call/put)
- Vertical spreads (bull/bear call/put spreads)
- Butterflies and Iron Butterflies
- Straddles and Strangles
- Iron Condors
- Calendars and Diagonals
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from advanced_options_engine import MarketData, AdvancedMetrics, AdvancedOptionsAnalyzer, MonteCarloEngine


@dataclass
class StrategyLeg:
    """Single leg of an option strategy"""
    market_data: MarketData
    quantity: int  # +1 for long, -1 for short
    

@dataclass
class StrategyMetrics:
    """Metrics for a complete strategy"""
    strategy_name: str
    total_cost: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    expected_value: float
    sharpe_ratio: float
    cvar_95: float
    edge_score: float
    risk_reward_ratio: float
    profit_at_current_price: float
    model_error_estimate: float
    

class OptionsStrategy:
    """Options strategy calculator and analyzer"""
    
    def __init__(self, mc_engine: MonteCarloEngine = None):
        self.mc_engine = mc_engine or MonteCarloEngine(n_simulations=50000)
        self.analyzer = AdvancedOptionsAnalyzer(self.mc_engine)
    
    def calculate_strategy_payoff(
        self,
        legs: List[StrategyLeg],
        final_prices: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate payoff for a multi-leg strategy
        Returns: (payoffs_array, initial_cost)
        """
        total_cost = 0.0
        total_payoff = np.zeros_like(final_prices)
        
        for leg in legs:
            md = leg.market_data
            qty = leg.quantity
            
            # Calculate premium
            if qty > 0:  # Long position - pay ask
                premium = md.ask if md.ask > 0 else (md.bid + md.ask) / 2
                total_cost += premium * qty
            else:  # Short position - receive bid
                premium = md.bid if md.bid > 0 else (md.bid + md.ask) / 2
                total_cost += premium * qty  # qty is negative, so this reduces cost
            
            # Calculate intrinsic value at expiration
            if md.option_type.lower() == 'call':
                intrinsic = np.maximum(final_prices - md.strike, 0)
            else:  # put
                intrinsic = np.maximum(md.strike - final_prices, 0)
            
            # Add to total payoff (consider quantity and direction)
            total_payoff += qty * intrinsic
        
        # Net payoff = intrinsic value - initial cost
        net_payoff = total_payoff - total_cost
        
        return net_payoff, total_cost
    
    def test_crash_scenarios(
        self,
        legs: List[StrategyLeg],
        current_price: float
    ) -> Dict[str, float]:
        """
        Test strategy against explicit crash scenarios
        Returns payoffs for -20%, -30%, -40%, -50% price moves
        """
        crash_scenarios = {
            'crash_20': current_price * 0.80,
            'crash_30': current_price * 0.70,
            'crash_40': current_price * 0.60,
            'crash_50': current_price * 0.50
        }
        
        results = {}
        for scenario_name, crash_price in crash_scenarios.items():
            payoffs, _ = self.calculate_strategy_payoff(legs, np.array([crash_price]))
            results[scenario_name] = payoffs[0]
        
        return results
    
    def analyze_strategy(
        self,
        legs: List[StrategyLeg],
        strategy_name: str,
        model_type: str = 'ensemble'
    ) -> StrategyMetrics:
        """Analyze a multi-leg option strategy"""
        
        if not legs:
            return self._default_strategy_metrics(strategy_name)
        
        # Use first leg's market data for simulation parameters
        base_md = legs[0].market_data
        S0 = base_md.current_price
        T = base_md.time_to_expiry
        mu = base_md.risk_free_rate - base_md.dividend_yield
        sigma = base_md.historical_volatility
        
        # Generate price paths
        if model_type == 'gbm':
            final_prices = self.mc_engine.geometric_brownian_motion(S0, mu, sigma, T)
        elif model_type == 'jump_diffusion':
            final_prices = self.mc_engine.jump_diffusion(S0, mu, sigma, T)
        elif model_type == 'heston':
            final_prices = self.mc_engine.heston_model(S0, sigma**2, mu, T=T)
        elif model_type == 'fat_tailed':
            final_prices = self.mc_engine.fat_tailed_distribution(S0, mu, sigma, T)
        else:  # ensemble
            final_prices = self.mc_engine.multi_model_ensemble(S0, mu, sigma, T)
        
        # EXPLICIT CRASH TESTING: Add crash scenarios to the price distribution
        crash_results = self.test_crash_scenarios(legs, S0)
        crash_prices = np.array([S0 * 0.80, S0 * 0.70, S0 * 0.60, S0 * 0.50])
        crash_payoffs, _ = self.calculate_strategy_payoff(legs, crash_prices)
        
        # Add crash scenarios to the distribution (10% weight to stress scenarios)
        n_crash = int(len(final_prices) * 0.10)
        if n_crash > 0:
            # Remove 10% of normal scenarios and replace with crash scenarios
            final_prices = final_prices[:-n_crash]
            # Replicate crash scenarios to fill the space
            crash_scenarios_extended = np.repeat(crash_prices, n_crash // len(crash_prices) + 1)[:n_crash]
            final_prices = np.concatenate([final_prices, crash_scenarios_extended])
        
        # Calculate strategy payoffs
        payoffs, total_cost = self.calculate_strategy_payoff(legs, final_prices)
        
        # Calculate metrics
        expected_value = np.mean(payoffs)
        prob_profit = np.sum(payoffs > 0) / len(payoffs) * 100
        
        # Max profit and loss (including crash scenarios)
        max_profit = np.max(payoffs)
        max_loss = min(np.min(payoffs), np.min(crash_payoffs))
        
        # Breakeven points (approximate from payoff function)
        breakeven_points = self._find_breakeven_points(legs, total_cost)
        
        # Risk metrics
        std_dev = np.std(payoffs)
        sharpe = (expected_value / std_dev) if std_dev > 0 else 0
        
        # CVaR
        sorted_payoffs = np.sort(payoffs)
        cvar_idx = int(0.05 * len(sorted_payoffs))
        cvar_95 = np.mean(sorted_payoffs[:cvar_idx]) if cvar_idx > 0 else sorted_payoffs[0]
        
        # Edge score
        edge_score = self._calculate_strategy_edge(
            expected_value, prob_profit / 100, sharpe, cvar_95, abs(total_cost)
        )
        
        # Risk-reward ratio
        risk_reward = abs(max_profit / max_loss) if max_loss != 0 else 0
        
        # Profit at current price
        current_payoffs, _ = self.calculate_strategy_payoff(legs, np.array([S0]))
        profit_at_current = current_payoffs[0]
        
        # Model error
        error_estimate = self._estimate_strategy_error(model_type, sigma, T, len(legs))
        
        return StrategyMetrics(
            strategy_name=strategy_name,
            total_cost=total_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            probability_of_profit=prob_profit,
            expected_value=expected_value,
            sharpe_ratio=sharpe,
            cvar_95=cvar_95,
            edge_score=edge_score,
            risk_reward_ratio=risk_reward,
            profit_at_current_price=profit_at_current,
            model_error_estimate=error_estimate
        )
    
    def _find_breakeven_points(self, legs: List[StrategyLeg], total_cost: float) -> List[float]:
        """Find breakeven points for the strategy"""
        # Get strike prices
        strikes = sorted(set(leg.market_data.strike for leg in legs))
        if not strikes:
            return []
        
        min_strike = min(strikes)
        max_strike = max(strikes)
        
        # Sample prices around strikes
        test_prices = np.linspace(
            max(min_strike * 0.7, 1), 
            max_strike * 1.3, 
            1000
        )
        
        # Calculate payoffs at each price
        payoffs, _ = self.calculate_strategy_payoff(legs, test_prices)
        
        # Find zero crossings (breakeven points)
        breakevens = []
        for i in range(len(payoffs) - 1):
            if (payoffs[i] <= 0 and payoffs[i+1] > 0) or (payoffs[i] >= 0 and payoffs[i+1] < 0):
                # Linear interpolation to find exact breakeven
                be = test_prices[i] + (test_prices[i+1] - test_prices[i]) * (
                    -payoffs[i] / (payoffs[i+1] - payoffs[i])
                )
                breakevens.append(be)
        
        return breakevens
    
    def _calculate_strategy_edge(
        self,
        expected_value: float,
        probability_of_profit: float,
        sharpe_ratio: float,
        cvar: float,
        cost: float
    ) -> float:
        """Calculate edge score for strategy"""
        if cost <= 0:
            return 0.0
        
        ev_component = expected_value / cost
        pop_component = probability_of_profit
        sharpe_component = max(0, min(sharpe_ratio / 2, 1))
        cvar_penalty = abs(min(cvar, 0)) / cost
        
        edge = (
            0.35 * ev_component + 
            0.30 * pop_component + 
            0.20 * sharpe_component - 
            0.15 * cvar_penalty
        )
        
        return edge * 100
    
    def _estimate_strategy_error(
        self, 
        model_type: str, 
        sigma: float, 
        T: float,
        num_legs: int
    ) -> float:
        """Estimate prediction error for strategy"""
        base_error = {
            'gbm': 0.05,
            'jump_diffusion': 0.08,
            'heston': 0.10,
            'fat_tailed': 0.07,
            'ensemble': 0.04
        }.get(model_type, 0.10)
        
        # More legs = more complexity = more error
        leg_factor = 1 + (num_legs - 1) * 0.1
        vol_factor = min(sigma / 0.3, 2.0)
        time_factor = min(T, 1.0)
        
        return base_error * leg_factor * vol_factor * (1 + time_factor * 0.5)
    
    def _default_strategy_metrics(self, strategy_name: str) -> StrategyMetrics:
        """Return default metrics when calculation fails"""
        return StrategyMetrics(
            strategy_name=strategy_name,
            total_cost=0.0,
            max_profit=0.0,
            max_loss=0.0,
            breakeven_points=[],
            probability_of_profit=0.0,
            expected_value=0.0,
            sharpe_ratio=0.0,
            cvar_95=0.0,
            edge_score=0.0,
            risk_reward_ratio=0.0,
            profit_at_current_price=0.0,
            model_error_estimate=0.0
        )


class StrategyBuilder:
    """Build common option strategies"""
    
    @staticmethod
    def long_call(call_data: MarketData) -> List[StrategyLeg]:
        """Simple long call"""
        return [StrategyLeg(call_data, 1)]
    
    @staticmethod
    def long_put(put_data: MarketData) -> List[StrategyLeg]:
        """Simple long put"""
        return [StrategyLeg(put_data, 1)]
    
    @staticmethod
    def bull_call_spread(
        lower_strike_call: MarketData,
        higher_strike_call: MarketData
    ) -> List[StrategyLeg]:
        """Bull call spread: Long lower strike, short higher strike"""
        return [
            StrategyLeg(lower_strike_call, 1),
            StrategyLeg(higher_strike_call, -1)
        ]
    
    @staticmethod
    def bear_put_spread(
        higher_strike_put: MarketData,
        lower_strike_put: MarketData
    ) -> List[StrategyLeg]:
        """Bear put spread: Long higher strike, short lower strike"""
        return [
            StrategyLeg(higher_strike_put, 1),
            StrategyLeg(lower_strike_put, -1)
        ]
    
    @staticmethod
    def long_straddle(
        call_data: MarketData,
        put_data: MarketData
    ) -> List[StrategyLeg]:
        """Long straddle: Long call and put at same strike"""
        return [
            StrategyLeg(call_data, 1),
            StrategyLeg(put_data, 1)
        ]
    
    @staticmethod
    def long_strangle(
        call_data: MarketData,
        put_data: MarketData
    ) -> List[StrategyLeg]:
        """Long strangle: Long OTM call and OTM put"""
        return [
            StrategyLeg(call_data, 1),
            StrategyLeg(put_data, 1)
        ]
    
    @staticmethod
    def iron_condor(
        lower_put: MarketData,
        lower_put_short: MarketData,
        higher_call_short: MarketData,
        higher_call: MarketData
    ) -> List[StrategyLeg]:
        """
        Iron Condor: 
        - Long lower put
        - Short higher put  
        - Short lower call
        - Long higher call
        """
        return [
            StrategyLeg(lower_put, 1),
            StrategyLeg(lower_put_short, -1),
            StrategyLeg(higher_call_short, -1),
            StrategyLeg(higher_call, 1)
        ]
    
    @staticmethod
    def butterfly_spread(
        lower_strike: MarketData,
        middle_strike: MarketData,
        higher_strike: MarketData,
        use_calls: bool = True
    ) -> List[StrategyLeg]:
        """
        Butterfly spread: 
        - Long 1 lower strike
        - Short 2 middle strikes
        - Long 1 higher strike
        """
        return [
            StrategyLeg(lower_strike, 1),
            StrategyLeg(middle_strike, -2),
            StrategyLeg(higher_strike, 1)
        ]
    
    @staticmethod
    def iron_butterfly(
        lower_put: MarketData,
        middle_call: MarketData,
        middle_put: MarketData,
        higher_call: MarketData
    ) -> List[StrategyLeg]:
        """
        Iron Butterfly:
        - Long lower put
        - Short middle put
        - Short middle call  
        - Long higher call
        """
        return [
            StrategyLeg(lower_put, 1),
            StrategyLeg(middle_put, -1),
            StrategyLeg(middle_call, -1),
            StrategyLeg(higher_call, 1)
        ]
    
    @staticmethod
    def covered_call(
        stock_price: float,
        call_data: MarketData
    ) -> List[StrategyLeg]:
        """
        Covered call: Own stock + short call
        Note: This simplifies stock ownership to a synthetic long position
        """
        return [StrategyLeg(call_data, -1)]
    
    @staticmethod
    def protective_put(
        stock_price: float,
        put_data: MarketData
    ) -> List[StrategyLeg]:
        """
        Protective put: Own stock + long put
        Note: This simplifies to just the put protection cost
        """
        return [StrategyLeg(put_data, 1)]


def create_strategy_from_options(
    strategy_type: str,
    options_data: List[Dict],
    current_price: float
) -> Tuple[List[StrategyLeg], str]:
    """
    Create a strategy from available options data
    
    Args:
        strategy_type: Type of strategy (e.g., 'bull_call_spread', 'iron_condor')
        options_data: List of option dictionaries
        current_price: Current stock price
        
    Returns:
        (list of StrategyLeg, strategy_description)
    """
    
    # Convert to MarketData objects
    calls = []
    puts = []
    
    for opt in options_data:
        md = MarketData(
            current_price=current_price,
            strike=opt.get('strike', 0),
            time_to_expiry=opt.get('days_to_expiry', 30) / 365.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            historical_volatility=opt.get('historical_vol', 0.25),
            implied_volatility=opt.get('implied_vol', 0.25),
            bid=opt.get('bid', 0),
            ask=opt.get('ask', 0),
            volume=opt.get('volume', 0),
            open_interest=opt.get('open_interest', 0),
            option_type=opt.get('type', 'call')
        )
        
        if opt.get('type') == 'call':
            calls.append((opt.get('strike', 0), md))
        else:
            puts.append((opt.get('strike', 0), md))
    
    calls.sort(key=lambda x: x[0])
    puts.sort(key=lambda x: x[0])
    
    # Build strategy based on type
    if strategy_type == 'long_call' and calls:
        atm_call = min(calls, key=lambda x: abs(x[0] - current_price))
        return [StrategyLeg(atm_call[1], 1)], f"Long Call @ ${atm_call[0]}"
    
    elif strategy_type == 'long_put' and puts:
        atm_put = min(puts, key=lambda x: abs(x[0] - current_price))
        return [StrategyLeg(atm_put[1], 1)], f"Long Put @ ${atm_put[0]}"
    
    elif strategy_type == 'bull_call_spread' and len(calls) >= 2:
        # Find ATM and OTM call
        atm_idx = min(range(len(calls)), key=lambda i: abs(calls[i][0] - current_price))
        if atm_idx < len(calls) - 1:
            lower = calls[atm_idx]
            higher = calls[atm_idx + 1]
            return [
                StrategyLeg(lower[1], 1),
                StrategyLeg(higher[1], -1)
            ], f"Bull Call Spread ${lower[0]}-${higher[0]}"
    
    elif strategy_type == 'bear_put_spread' and len(puts) >= 2:
        atm_idx = min(range(len(puts)), key=lambda i: abs(puts[i][0] - current_price))
        if atm_idx > 0:
            higher = puts[atm_idx]
            lower = puts[atm_idx - 1]
            return [
                StrategyLeg(higher[1], 1),
                StrategyLeg(lower[1], -1)
            ], f"Bear Put Spread ${lower[0]}-${higher[0]}"
    
    elif strategy_type == 'long_straddle' and calls and puts:
        atm_call = min(calls, key=lambda x: abs(x[0] - current_price))
        atm_put = min(puts, key=lambda x: abs(x[0] - current_price))
        return [
            StrategyLeg(atm_call[1], 1),
            StrategyLeg(atm_put[1], 1)
        ], f"Long Straddle @ ${atm_call[0]}"
    
    elif strategy_type == 'iron_condor' and len(calls) >= 2 and len(puts) >= 2:
        # Find strikes around current price
        mid_idx_call = min(range(len(calls)), key=lambda i: abs(calls[i][0] - current_price))
        mid_idx_put = min(range(len(puts)), key=lambda i: abs(puts[i][0] - current_price))
        
        if mid_idx_call < len(calls) - 1 and mid_idx_put > 0:
            return [
                StrategyLeg(puts[max(0, mid_idx_put - 1)][1], 1),
                StrategyLeg(puts[mid_idx_put][1], -1),
                StrategyLeg(calls[mid_idx_call][1], -1),
                StrategyLeg(calls[min(len(calls)-1, mid_idx_call + 1)][1], 1)
            ], "Iron Condor"
    
    return [], "Strategy could not be built"
