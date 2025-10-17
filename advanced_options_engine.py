"""
Advanced Options Pricing and Analysis Engine

This module implements sophisticated probabilistic models including:
- Jump-diffusion (Merton model)
- Stochastic volatility (Heston model)
- Fat-tailed distributions
- Monte Carlo simulations
- Advanced risk metrics (CVaR, Expected Shortfall, Sharpe Ratio)
"""

import numpy as np
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MarketData:
    """Market data structure for options pricing"""
    current_price: float
    strike: float
    time_to_expiry: float  # in years
    risk_free_rate: float
    dividend_yield: float
    historical_volatility: float
    implied_volatility: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    option_type: str  # 'call' or 'put'


@dataclass
class AdvancedMetrics:
    """Advanced metrics for option analysis"""
    expected_value: float
    probability_of_profit: float
    expected_payoff_if_win: float
    expected_payoff_if_loss: float
    cvar_95: float  # Conditional Value at Risk at 95%
    cvar_99: float  # CVaR at 99%
    expected_shortfall: float
    sharpe_ratio: float
    skewness: float
    kurtosis: float
    edge_score: float  # Combined metric
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_error_estimate: float


class MonteCarloEngine:
    """Monte Carlo simulation engine for options pricing"""
    
    def __init__(self, n_simulations: int = 50000, random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        if random_seed:
            np.random.seed(random_seed)
    
    def geometric_brownian_motion(
        self, 
        S0: float, 
        mu: float, 
        sigma: float, 
        T: float,
        n_paths: int = None
    ) -> np.ndarray:
        """Standard GBM simulation"""
        n = n_paths or self.n_simulations
        Z = np.random.standard_normal(n)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return ST
    
    def jump_diffusion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        jump_intensity: float = 0.1,  # lambda: jumps per year
        jump_mean: float = -0.02,  # mean jump size
        jump_std: float = 0.05,  # jump volatility
        n_paths: int = None
    ) -> np.ndarray:
        """Merton Jump-Diffusion model for fat tails and jumps"""
        n = n_paths or self.n_simulations
        
        # GBM component
        Z = np.random.standard_normal(n)
        diffusion = (mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z
        
        # Jump component
        n_jumps = np.random.poisson(jump_intensity * T, n)
        jump_component = np.zeros(n)
        
        for i in range(n):
            if n_jumps[i] > 0:
                jumps = np.random.normal(jump_mean, jump_std, n_jumps[i])
                jump_component[i] = np.sum(jumps)
        
        ST = S0 * np.exp(diffusion + jump_component)
        return ST
    
    def heston_model(
        self,
        S0: float,
        v0: float,  # initial variance
        mu: float,
        kappa: float = 2.0,  # mean reversion speed
        theta: float = 0.04,  # long-term variance
        sigma_v: float = 0.3,  # volatility of volatility
        rho: float = -0.7,  # correlation between price and vol
        T: float = 1.0,
        n_steps: int = 100,
        n_paths: int = None
    ) -> np.ndarray:
        """Heston Stochastic Volatility model"""
        n = n_paths or self.n_simulations
        dt = T / n_steps
        
        S = np.zeros((n, n_steps + 1))
        v = np.zeros((n, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(1, n_steps + 1):
            Z1 = np.random.standard_normal(n)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n)
            
            # Update variance (using max to prevent negative variance)
            v[:, t] = np.maximum(
                v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + 
                sigma_v * np.sqrt(v[:, t-1] * dt) * Z2,
                1e-10
            )
            
            # Update stock price
            S[:, t] = S[:, t-1] * np.exp(
                (mu - 0.5 * v[:, t-1]) * dt + 
                np.sqrt(v[:, t-1] * dt) * Z1
            )
        
        return S[:, -1]
    
    def fat_tailed_distribution(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        df: float = 5.0,  # degrees of freedom for Student's t
        n_paths: int = None
    ) -> np.ndarray:
        """Fat-tailed distribution using Student's t"""
        n = n_paths or self.n_simulations
        
        # Use Student's t instead of normal for fatter tails
        t_samples = student_t.rvs(df, size=n)
        # Scale to match desired volatility
        scale_factor = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        Z = t_samples / scale_factor
        
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return ST
    
    def multi_model_ensemble(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Ensemble of multiple models for robust predictions
        
        UPDATED WEIGHTS (Oct 2025): Increased tail risk weighting to better capture crashes
        - GBM reduced from 30% to 15% (underestimates tail risk)
        - Jump-Diffusion increased from 30% to 40% (captures sudden crashes)
        - Fat-Tailed increased from 20% to 25% (captures extreme scenarios)
        """
        if weights is None:
            weights = {
                'gbm': 0.15,
                'jump_diffusion': 0.40,
                'heston': 0.20,
                'fat_tailed': 0.25
            }
        
        n_per_model = int(self.n_simulations * weights['gbm'])
        paths = []
        
        # GBM
        if weights.get('gbm', 0) > 0:
            n_gbm = int(self.n_simulations * weights['gbm'])
            paths.append(self.geometric_brownian_motion(S0, mu, sigma, T, n_gbm))
        
        # Jump-Diffusion
        if weights.get('jump_diffusion', 0) > 0:
            n_jump = int(self.n_simulations * weights['jump_diffusion'])
            paths.append(self.jump_diffusion(S0, mu, sigma, T, n_paths=n_jump))
        
        # Heston
        if weights.get('heston', 0) > 0:
            n_heston = int(self.n_simulations * weights['heston'])
            paths.append(self.heston_model(S0, sigma**2, mu, T=T, n_paths=n_heston))
        
        # Fat-tailed
        if weights.get('fat_tailed', 0) > 0:
            n_fat = int(self.n_simulations * weights['fat_tailed'])
            paths.append(self.fat_tailed_distribution(S0, mu, sigma, T, n_paths=n_fat))
        
        # Combine all paths
        all_paths = np.concatenate(paths)
        # Ensure we have exactly n_simulations
        if len(all_paths) > self.n_simulations:
            all_paths = all_paths[:self.n_simulations]
        elif len(all_paths) < self.n_simulations:
            # Pad with GBM
            deficit = self.n_simulations - len(all_paths)
            all_paths = np.concatenate([
                all_paths, 
                self.geometric_brownian_motion(S0, mu, sigma, T, deficit)
            ])
        
        return all_paths


class AdvancedOptionsAnalyzer:
    """Advanced options analysis with comprehensive risk metrics"""
    
    def __init__(self, mc_engine: MonteCarloEngine = None):
        self.mc_engine = mc_engine or MonteCarloEngine(n_simulations=50000)
    
    def calculate_payoff(
        self, 
        final_prices: np.ndarray, 
        strike: float, 
        premium: float, 
        option_type: str
    ) -> np.ndarray:
        """Calculate option payoff for each simulated path"""
        if option_type.lower() == 'call':
            intrinsic = np.maximum(final_prices - strike, 0)
        else:  # put
            intrinsic = np.maximum(strike - final_prices, 0)
        
        return intrinsic - premium  # Net payoff
    
    def calculate_cvar(self, payoffs: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        sorted_payoffs = np.sort(payoffs)
        cutoff_index = int((1 - confidence_level) * len(sorted_payoffs))
        
        if cutoff_index == 0:
            return sorted_payoffs[0]
        
        # Average of worst (1-confidence_level)% outcomes
        worst_outcomes = sorted_payoffs[:cutoff_index]
        return np.mean(worst_outcomes) if len(worst_outcomes) > 0 else sorted_payoffs[0]
    
    def calculate_sharpe_ratio(
        self, 
        expected_return: float, 
        std_dev: float, 
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate Sharpe ratio for the option"""
        if std_dev == 0:
            return 0.0
        return (expected_return - risk_free_rate) / std_dev
    
    def calculate_edge_score(
        self,
        expected_value: float,
        probability_of_profit: float,
        sharpe_ratio: float,
        cvar: float,
        premium: float
    ) -> float:
        """
        Calculate combined edge score
        Higher is better, accounts for EV, PoP, risk-adjusted returns, and tail risk
        """
        if premium <= 0:
            return 0.0
        
        # Normalize components
        ev_component = expected_value / premium if premium > 0 else 0
        pop_component = probability_of_profit
        sharpe_component = max(0, min(sharpe_ratio / 2, 1))  # Normalize to 0-1
        
        # CVaR penalty (worse CVaR = higher penalty)
        cvar_penalty = abs(min(cvar, 0)) / premium if premium > 0 else 0
        
        # Weighted combination
        edge = (
            0.35 * ev_component + 
            0.30 * pop_component + 
            0.20 * sharpe_component - 
            0.15 * cvar_penalty
        )
        
        return edge * 100  # Scale to 0-100
    
    def analyze_option(
        self, 
        market_data: MarketData,
        model_type: str = 'ensemble',
        use_implied_vol: bool = False
    ) -> AdvancedMetrics:
        """
        Comprehensive option analysis using Monte Carlo simulation
        
        Args:
            market_data: Market data for the option
            model_type: 'gbm', 'jump_diffusion', 'heston', 'fat_tailed', or 'ensemble'
            use_implied_vol: If True, use implied vol; otherwise use historical vol
        """
        S0 = market_data.current_price
        K = market_data.strike
        T = market_data.time_to_expiry
        r = market_data.risk_free_rate
        q = market_data.dividend_yield
        sigma = market_data.implied_volatility if use_implied_vol else market_data.historical_volatility
        
        # Risk-neutral drift
        mu = r - q
        
        # Premium (use ask for buying)
        premium = market_data.ask if market_data.ask > 0 else (
            (market_data.bid + market_data.ask) / 2 if market_data.bid > 0 else market_data.bid
        )
        
        if premium <= 0:
            # Return default metrics if no valid premium
            return self._default_metrics()
        
        # Generate price paths based on model type
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
        
        # Calculate payoffs
        payoffs = self.calculate_payoff(
            final_prices, K, premium, market_data.option_type
        )
        
        # Basic metrics
        expected_value = np.mean(payoffs)
        prob_profit = np.sum(payoffs > 0) / len(payoffs)
        
        # Winning and losing scenarios
        winning_payoffs = payoffs[payoffs > 0]
        losing_payoffs = payoffs[payoffs <= 0]
        
        exp_payoff_win = np.mean(winning_payoffs) if len(winning_payoffs) > 0 else 0.0
        exp_payoff_loss = np.mean(losing_payoffs) if len(losing_payoffs) > 0 else 0.0
        
        # Risk metrics
        cvar_95 = self.calculate_cvar(payoffs, 0.95)
        cvar_99 = self.calculate_cvar(payoffs, 0.99)
        expected_shortfall = self.calculate_cvar(payoffs, 0.95)
        
        # Return metrics
        std_dev = np.std(payoffs)
        sharpe = self.calculate_sharpe_ratio(expected_value, std_dev, r)
        
        # Distribution metrics
        skewness = self._calculate_skewness(payoffs)
        kurtosis = self._calculate_kurtosis(payoffs)
        
        # Edge score
        edge = self.calculate_edge_score(
            expected_value, prob_profit, sharpe, cvar_95, premium
        )
        
        # Confidence intervals (95%)
        sorted_payoffs = np.sort(payoffs)
        ci_lower = sorted_payoffs[int(0.025 * len(sorted_payoffs))]
        ci_upper = sorted_payoffs[int(0.975 * len(sorted_payoffs))]
        
        # Model error estimate (based on model type)
        error_estimate = self._estimate_model_error(model_type, sigma, T)
        
        return AdvancedMetrics(
            expected_value=expected_value,
            probability_of_profit=prob_profit * 100,
            expected_payoff_if_win=exp_payoff_win,
            expected_payoff_if_loss=exp_payoff_loss,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe,
            skewness=skewness,
            kurtosis=kurtosis,
            edge_score=edge,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            model_error_estimate=error_estimate
        )
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution"""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution"""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4)
        kurt -= (3 * (n-1)**2) / ((n-2) * (n-3))
        return kurt
    
    def _estimate_model_error(self, model_type: str, sigma: float, T: float) -> float:
        """Estimate model prediction error as percentage"""
        base_error = {
            'gbm': 0.05,
            'jump_diffusion': 0.08,
            'heston': 0.10,
            'fat_tailed': 0.07,
            'ensemble': 0.04
        }.get(model_type, 0.10)
        
        # Increase error for higher volatility and longer time
        vol_factor = min(sigma / 0.3, 2.0)  # Cap at 2x
        time_factor = min(T, 1.0)  # Cap at 1 year
        
        return base_error * vol_factor * (1 + time_factor * 0.5)
    
    def _default_metrics(self) -> AdvancedMetrics:
        """Return default metrics when calculation fails"""
        return AdvancedMetrics(
            expected_value=0.0,
            probability_of_profit=0.0,
            expected_payoff_if_win=0.0,
            expected_payoff_if_loss=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            expected_shortfall=0.0,
            sharpe_ratio=0.0,
            skewness=0.0,
            kurtosis=0.0,
            edge_score=0.0,
            confidence_interval_lower=0.0,
            confidence_interval_upper=0.0,
            model_error_estimate=0.0
        )


class LiquidityAdjuster:
    """Adjust option prices for liquidity and market microstructure"""
    
    @staticmethod
    def calculate_effective_spread(bid: float, ask: float) -> float:
        """Calculate effective bid-ask spread"""
        if bid <= 0 or ask <= 0:
            return 0.0
        return (ask - bid) / ((ask + bid) / 2)
    
    @staticmethod
    def estimate_slippage(
        volume: int, 
        open_interest: int, 
        trade_size: int = 10
    ) -> float:
        """Estimate slippage based on liquidity"""
        if open_interest == 0:
            return 0.10  # 10% slippage for illiquid options
        
        liquidity_ratio = trade_size / open_interest
        
        if liquidity_ratio < 0.01:
            return 0.01  # 1% for very liquid
        elif liquidity_ratio < 0.05:
            return 0.03  # 3% for moderately liquid
        elif liquidity_ratio < 0.10:
            return 0.05  # 5% for less liquid
        else:
            return 0.10  # 10% for illiquid
    
    @staticmethod
    def adjust_premium_for_liquidity(
        premium: float,
        bid: float,
        ask: float,
        volume: int,
        open_interest: int,
        trade_size: int = 10
    ) -> Tuple[float, Dict[str, float]]:
        """
        Adjust premium for real-world trading costs
        Returns: (adjusted_premium, adjustment_details)
        """
        if premium <= 0:
            return premium, {}
        
        # Spread cost
        spread_cost = (ask - bid) / 2 if bid > 0 and ask > bid else 0
        
        # Slippage
        slippage_pct = LiquidityAdjuster.estimate_slippage(volume, open_interest, trade_size)
        slippage_cost = premium * slippage_pct
        
        # Commission (typical options commission)
        commission = 0.65  # per contract
        
        # Total adjustment
        adjusted_premium = premium + spread_cost + slippage_cost + commission
        
        details = {
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'slippage_pct': slippage_pct * 100,
            'commission': commission,
            'total_adjustment': adjusted_premium - premium
        }
        
        return adjusted_premium, details
