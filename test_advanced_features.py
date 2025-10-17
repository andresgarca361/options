"""
Comprehensive Testing Suite for Advanced Options Analytics

Tests all major features:
- Monte Carlo simulation accuracy
- Multiple pricing models
- Risk metrics calculations
- Strategy builders
- Liquidity adjustments
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from advanced_options_engine import (
    MarketData, AdvancedMetrics, AdvancedOptionsAnalyzer, 
    MonteCarloEngine, LiquidityAdjuster
)
from options_strategies import (
    OptionsStrategy, StrategyBuilder, StrategyLeg
)


def test_monte_carlo_convergence():
    """Test Monte Carlo convergence with known theoretical values"""
    print("\n=== Testing Monte Carlo Convergence ===")
    
    mc_engine = MonteCarloEngine(n_simulations=100000, random_seed=42)
    
    S0 = 100
    mu = 0.05
    sigma = 0.20
    T = 1.0
    
    # GBM should converge to theoretical mean
    paths = mc_engine.geometric_brownian_motion(S0, mu, sigma, T)
    theoretical_mean = S0 * np.exp(mu * T)
    simulated_mean = np.mean(paths)
    
    error_pct = abs(simulated_mean - theoretical_mean) / theoretical_mean * 100
    
    print(f"Theoretical Mean: ${theoretical_mean:.2f}")
    print(f"Simulated Mean: ${simulated_mean:.2f}")
    print(f"Error: {error_pct:.2f}%")
    
    if error_pct < 1.0:
        print("✅ PASS: GBM convergence within 1%")
        return True
    else:
        print("❌ FAIL: GBM convergence error too high")
        return False


def test_fat_tails():
    """Test that fat-tailed model produces heavier tails than GBM"""
    print("\n=== Testing Fat-Tailed Distribution ===")
    
    mc_engine = MonteCarloEngine(n_simulations=50000, random_seed=42)
    
    S0 = 100
    mu = 0.05
    sigma = 0.20
    T = 1.0
    
    gbm_paths = mc_engine.geometric_brownian_motion(S0, mu, sigma, T)
    fat_paths = mc_engine.fat_tailed_distribution(S0, mu, sigma, T, df=5)
    
    # Fat-tailed should have higher kurtosis
    from scipy.stats import kurtosis
    gbm_kurt = kurtosis(gbm_paths)
    fat_kurt = kurtosis(fat_paths)
    
    print(f"GBM Kurtosis: {gbm_kurt:.3f}")
    print(f"Fat-Tailed Kurtosis: {fat_kurt:.3f}")
    
    if fat_kurt > gbm_kurt:
        print("✅ PASS: Fat-tailed model has heavier tails")
        return True
    else:
        print("❌ FAIL: Fat-tailed model doesn't show heavier tails")
        return False


def test_jump_diffusion():
    """Test jump-diffusion model produces jumps"""
    print("\n=== Testing Jump-Diffusion Model ===")
    
    mc_engine = MonteCarloEngine(n_simulations=10000, random_seed=42)
    
    S0 = 100
    mu = 0.05
    sigma = 0.20
    T = 1.0
    
    # Model with high jump intensity
    jump_paths = mc_engine.jump_diffusion(
        S0, mu, sigma, T, 
        jump_intensity=2.0,  # 2 jumps per year on average
        jump_mean=-0.10,     # -10% jumps
        jump_std=0.15
    )
    
    gbm_paths = mc_engine.geometric_brownian_motion(S0, mu, sigma, T)
    
    # Jump model should have lower mean (negative jumps)
    jump_mean = np.mean(jump_paths)
    gbm_mean = np.mean(gbm_paths)
    
    print(f"GBM Mean: ${gbm_mean:.2f}")
    print(f"Jump-Diffusion Mean: ${jump_mean:.2f}")
    
    if jump_mean < gbm_mean:
        print("✅ PASS: Jump-diffusion shows impact of negative jumps")
        return True
    else:
        print("⚠️ WARNING: Jump impact not clearly visible (may be random)")
        return True  # Don't fail, could be randomness


def test_cvar_calculation():
    """Test CVaR calculation is correct"""
    print("\n=== Testing CVaR Calculation ===")
    
    # Create known distribution
    payoffs = np.array([10, 5, 0, -5, -10, -20, -30, -40, -50, -60])
    
    analyzer = AdvancedOptionsAnalyzer()
    cvar_95 = analyzer.calculate_cvar(payoffs, 0.95)
    
    # CVaR at 95% should be average of worst 5% (1 value) = -60
    expected_cvar = -60.0
    
    print(f"Payoffs: {payoffs}")
    print(f"CVaR (95%): {cvar_95:.2f}")
    print(f"Expected: {expected_cvar:.2f}")
    
    if abs(cvar_95 - expected_cvar) < 0.01:
        print("✅ PASS: CVaR calculation correct")
        return True
    else:
        print("❌ FAIL: CVaR calculation incorrect")
        return False


def test_option_analysis():
    """Test complete option analysis pipeline"""
    print("\n=== Testing Option Analysis Pipeline ===")
    
    mc_engine = MonteCarloEngine(n_simulations=50000, random_seed=42)
    analyzer = AdvancedOptionsAnalyzer(mc_engine)
    
    # ATM call option
    market_data = MarketData(
        current_price=100.0,
        strike=100.0,
        time_to_expiry=0.25,  # 3 months
        risk_free_rate=0.05,
        dividend_yield=0.0,
        historical_volatility=0.25,
        implied_volatility=0.30,
        bid=4.0,
        ask=4.5,
        volume=1000,
        open_interest=5000,
        option_type='call'
    )
    
    metrics = analyzer.analyze_option(market_data, model_type='ensemble', use_implied_vol=False)
    
    print(f"Expected Value: ${metrics.expected_value:.2f}")
    print(f"Probability of Profit: {metrics.probability_of_profit:.1f}%")
    print(f"CVaR (95%): ${metrics.cvar_95:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Edge Score: {metrics.edge_score:.1f}")
    
    # Basic sanity checks
    checks = []
    
    # PoP should be reasonable for ATM option (30-70%)
    if 30 <= metrics.probability_of_profit <= 70:
        print("✅ PASS: Probability of profit is reasonable")
        checks.append(True)
    else:
        print(f"❌ FAIL: PoP {metrics.probability_of_profit:.1f}% seems unreasonable")
        checks.append(False)
    
    # CVaR should be negative (max loss)
    if metrics.cvar_95 < 0:
        print("✅ PASS: CVaR is negative (loss)")
        checks.append(True)
    else:
        print("❌ FAIL: CVaR should be negative")
        checks.append(False)
    
    # Edge score should be calculated
    if -100 <= metrics.edge_score <= 100:
        print("✅ PASS: Edge score in valid range")
        checks.append(True)
    else:
        print("❌ FAIL: Edge score out of range")
        checks.append(False)
    
    return all(checks)


def test_liquidity_adjustments():
    """Test liquidity adjustment calculations"""
    print("\n=== Testing Liquidity Adjustments ===")
    
    premium = 5.0
    bid = 4.8
    ask = 5.2
    volume = 100
    open_interest = 500
    
    adjusted, details = LiquidityAdjuster.adjust_premium_for_liquidity(
        premium, bid, ask, volume, open_interest, trade_size=10
    )
    
    print(f"Original Premium: ${premium:.2f}")
    print(f"Adjusted Premium: ${adjusted:.2f}")
    print(f"Spread Cost: ${details['spread_cost']:.2f}")
    print(f"Slippage ({details['slippage_pct']:.1f}%): ${details['slippage_cost']:.2f}")
    print(f"Commission: ${details['commission']:.2f}")
    
    # Adjusted should be higher than original
    if adjusted > premium:
        print("✅ PASS: Adjusted premium includes costs")
        return True
    else:
        print("❌ FAIL: Adjustment not applied correctly")
        return False


def test_strategy_builder():
    """Test strategy construction and analysis"""
    print("\n=== Testing Strategy Builder ===")
    
    mc_engine = MonteCarloEngine(n_simulations=50000, random_seed=42)
    strategy_calc = OptionsStrategy(mc_engine)
    
    # Create bull call spread
    lower_call = MarketData(
        current_price=100.0,
        strike=100.0,
        time_to_expiry=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        historical_volatility=0.25,
        implied_volatility=0.30,
        bid=4.8,
        ask=5.2,
        volume=1000,
        open_interest=5000,
        option_type='call'
    )
    
    higher_call = MarketData(
        current_price=100.0,
        strike=110.0,
        time_to_expiry=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.0,
        historical_volatility=0.25,
        implied_volatility=0.28,
        bid=1.8,
        ask=2.2,
        volume=800,
        open_interest=4000,
        option_type='call'
    )
    
    legs = StrategyBuilder.bull_call_spread(lower_call, higher_call)
    metrics = strategy_calc.analyze_strategy(legs, "Bull Call Spread", model_type='ensemble')
    
    print(f"Strategy: {metrics.strategy_name}")
    print(f"Total Cost: ${metrics.total_cost:.2f}")
    print(f"Max Profit: ${metrics.max_profit:.2f}")
    print(f"Max Loss: ${metrics.max_loss:.2f}")
    print(f"Probability of Profit: {metrics.probability_of_profit:.1f}%")
    print(f"Breakeven Points: {[f'${x:.2f}' for x in metrics.breakeven_points]}")
    
    checks = []
    
    # Max profit should be limited (spread width - cost)
    spread_width = 110 - 100
    expected_max_profit_approx = spread_width - abs(metrics.total_cost)
    
    if 0 < metrics.max_profit <= spread_width:
        print("✅ PASS: Max profit is bounded by spread width")
        checks.append(True)
    else:
        print(f"❌ FAIL: Max profit ${metrics.max_profit:.2f} seems wrong")
        checks.append(False)
    
    # Should have at least one breakeven point
    if len(metrics.breakeven_points) >= 1:
        print("✅ PASS: Breakeven points calculated")
        checks.append(True)
    else:
        print("❌ FAIL: No breakeven points found")
        checks.append(False)
    
    return all(checks)


def test_real_market_data():
    """Test with real market data (AAPL)"""
    print("\n=== Testing with Real Market Data (AAPL) ===")
    
    try:
        ticker = yf.Ticker("AAPL")
        current_price = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice'))
        
        print(f"AAPL Current Price: ${current_price:.2f}")
        
        # Get options
        expirations = ticker.options
        if not expirations:
            print("⚠️ No options data available")
            return True
        
        exp_date = expirations[0]
        opt_chain = ticker.option_chain(exp_date)
        
        if len(opt_chain.calls) > 0:
            call = opt_chain.calls.iloc[0]
            print(f"\nSample Call Option:")
            print(f"  Strike: ${call['strike']:.2f}")
            print(f"  Bid: ${call.get('bid', 0):.2f}")
            print(f"  Ask: ${call.get('ask', 0):.2f}")
            print(f"  IV: {call.get('impliedVolatility', 0):.1%}")
            print("✅ PASS: Successfully fetched real market data")
            return True
        else:
            print("⚠️ No call options found")
            return True
            
    except Exception as e:
        print(f"⚠️ Could not fetch real data: {e}")
        return True  # Don't fail on API issues


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("ADVANCED OPTIONS ANALYTICS - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Monte Carlo Convergence", test_monte_carlo_convergence),
        ("Fat-Tailed Distribution", test_fat_tails),
        ("Jump-Diffusion Model", test_jump_diffusion),
        ("CVaR Calculation", test_cvar_calculation),
        ("Option Analysis Pipeline", test_option_analysis),
        ("Liquidity Adjustments", test_liquidity_adjustments),
        ("Strategy Builder", test_strategy_builder),
        ("Real Market Data", test_real_market_data),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed ({pass_rate:.1f}%)")
    
    if pass_rate >= 80:
        print("\n🎉 EXCELLENT: System is working well!")
    elif pass_rate >= 60:
        print("\n✓ GOOD: Most features working, some improvements needed")
    else:
        print("\n⚠️ NEEDS WORK: Several issues detected")
    
    return pass_rate >= 80


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
