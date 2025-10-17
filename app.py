"""
Advanced Options Analytics Platform

Features:
- Multiple probabilistic models (GBM, Jump-Diffusion, Heston, Fat-Tailed, Ensemble)
- Monte Carlo simulations with 50,000+ price paths
- Advanced risk metrics (CVaR, Expected Shortfall, Sharpe Ratio)
- Multi-strategy support (Spreads, Butterflies, Straddles, Iron Condors, etc.)
- Liquidity adjustments for real-world trading
- Comprehensive error analysis and confidence intervals
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
from typing import List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from advanced_options_engine import (
    MarketData, AdvancedMetrics, AdvancedOptionsAnalyzer, 
    MonteCarloEngine, LiquidityAdjuster
)
from options_strategies import (
    OptionsStrategy, StrategyBuilder, StrategyLeg,
    create_strategy_from_options
)
from strategy_optimizer import AdvancedStrategyOptimizer, StrategyCandidate


st.set_page_config(page_title="Advanced Options Analytics", layout='wide', page_icon="📊")


def fetch_stock_data(symbol: str) -> Tuple[float, float, float]:
    """Fetch stock price and calculate historical metrics with robust error handling"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Try multiple price fields
        current_price = (
            info.get('currentPrice') or 
            info.get('regularMarketPrice') or 
            info.get('previousClose') or 
            0
        )
        
        if current_price <= 0:
            st.error(f"Could not fetch valid price for {symbol}")
            return 0, 0.08, 0.20
        
        # Historical data with fallback
        try:
            hist = ticker.history(period="1y")
            if len(hist) < 30:
                st.warning(f"Limited historical data for {symbol}, using defaults")
                return current_price, 0.08, 0.20
            
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                st.warning(f"Insufficient return data for {symbol}, using defaults")
                return current_price, 0.08, 0.20
            
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Sanity check and bounds
            annual_return = max(-0.5, min(1.0, annual_return))
            annual_volatility = max(0.05, min(2.0, annual_volatility))
            
            return current_price, annual_return, annual_volatility
            
        except Exception as hist_error:
            st.warning(f"Could not calculate historical metrics: {hist_error}. Using defaults.")
            return current_price, 0.08, 0.20
        
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return 0, 0.08, 0.20


def fetch_options_data(symbol: str) -> List[Dict]:
    """Fetch options chain from Yahoo Finance with robust error handling"""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        
        if not expirations:
            st.warning(f"No options available for {symbol}")
            return []
        
        all_options = []
        successful_expirations = 0
        
        for exp_date in expirations[:8]:  # Limit to first 8 expirations
            try:
                opt_chain = ticker.option_chain(exp_date)
                
                # Process calls with validation
                if hasattr(opt_chain, 'calls') and len(opt_chain.calls) > 0:
                    for _, call in opt_chain.calls.iterrows():
                        try:
                            # Validate required fields
                            if pd.isna(call.get('strike')):
                                continue
                                
                            all_options.append({
                                'contractSymbol': str(call.get('contractSymbol', '')),
                                'strike': float(call['strike']),
                                'type': 'call',
                                'last': float(call.get('lastPrice', 0)) if pd.notna(call.get('lastPrice')) else 0,
                                'bid': float(call.get('bid', 0)) if pd.notna(call.get('bid')) else 0,
                                'ask': float(call.get('ask', 0)) if pd.notna(call.get('ask')) else 0,
                                'expirationDate': exp_date,
                                'impliedVolatility': float(call.get('impliedVolatility', 0.2)) if pd.notna(call.get('impliedVolatility')) else 0.2,
                                'volume': int(call.get('volume', 0)) if pd.notna(call.get('volume')) else 0,
                                'openInterest': int(call.get('openInterest', 0)) if pd.notna(call.get('openInterest')) else 0
                            })
                        except (ValueError, TypeError, KeyError):
                            continue
                
                # Process puts with validation
                if hasattr(opt_chain, 'puts') and len(opt_chain.puts) > 0:
                    for _, put in opt_chain.puts.iterrows():
                        try:
                            if pd.isna(put.get('strike')):
                                continue
                                
                            all_options.append({
                                'contractSymbol': str(put.get('contractSymbol', '')),
                                'strike': float(put['strike']),
                                'type': 'put',
                                'last': float(put.get('lastPrice', 0)) if pd.notna(put.get('lastPrice')) else 0,
                                'bid': float(put.get('bid', 0)) if pd.notna(put.get('bid')) else 0,
                                'ask': float(put.get('ask', 0)) if pd.notna(put.get('ask')) else 0,
                                'expirationDate': exp_date,
                                'impliedVolatility': float(put.get('impliedVolatility', 0.2)) if pd.notna(put.get('impliedVolatility')) else 0.2,
                                'volume': int(put.get('volume', 0)) if pd.notna(put.get('volume')) else 0,
                                'openInterest': int(put.get('openInterest', 0)) if pd.notna(put.get('openInterest')) else 0
                            })
                        except (ValueError, TypeError, KeyError):
                            continue
                
                successful_expirations += 1
                    
            except Exception as exp_error:
                st.warning(f"Skipping expiration {exp_date}: {exp_error}")
                continue
        
        if successful_expirations == 0:
            st.error(f"Could not fetch any valid options data for {symbol}")
        
        return all_options
        
    except Exception as e:
        st.error(f"Error fetching options for {symbol}: {e}")
        return []


def create_payoff_diagram(legs: List[StrategyLeg], strategy_name: str, current_price: float):
    """Create interactive payoff diagram"""
    if not legs:
        return None
    
    strikes = [leg.market_data.strike for leg in legs]
    min_price = max(min(strikes) * 0.8, 1)
    max_price = max(strikes) * 1.2
    
    prices = np.linspace(min_price, max_price, 200)
    
    strategy_calc = OptionsStrategy()
    payoffs, initial_cost = strategy_calc.calculate_strategy_payoff(legs, prices)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prices,
        y=payoffs,
        mode='lines',
        name='P&L at Expiration',
        line=dict(color='blue', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,100,255,0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=current_price, line_dash="dash", line_color="green", opacity=0.7,
                  annotation_text=f"Current: ${current_price:.2f}")
    
    fig.update_layout(
        title=f"{strategy_name} - Profit/Loss Diagram",
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def display_advanced_metrics(metrics: AdvancedMetrics, premium: float):
    """Display advanced metrics in organized columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Value", f"${metrics.expected_value:.2f}")
        st.metric("Probability of Profit", f"{metrics.probability_of_profit:.1f}%")
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    
    with col2:
        st.metric("Win Payoff (Avg)", f"${metrics.expected_payoff_if_win:.2f}")
        st.metric("Loss Payoff (Avg)", f"${metrics.expected_payoff_if_loss:.2f}")
        st.metric("CVaR (95%)", f"${metrics.cvar_95:.2f}")
    
    with col3:
        st.metric("Edge Score", f"{metrics.edge_score:.1f}/100")
        st.metric("Skewness", f"{metrics.skewness:.2f}")
        st.metric("Model Error", f"±{metrics.model_error_estimate*100:.1f}%")
    
    with st.expander("📊 95% Confidence Interval"):
        st.write(f"**Lower Bound:** ${metrics.confidence_interval_lower:.2f}")
        st.write(f"**Upper Bound:** ${metrics.confidence_interval_upper:.2f}")
        st.write(f"**Expected Value:** ${metrics.expected_value:.2f}")
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            x=[metrics.confidence_interval_lower, metrics.expected_value, metrics.confidence_interval_upper],
            name="Payoff Distribution",
            boxmean='sd'
        ))
        fig.update_layout(height=200, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 Advanced Options Analytics Platform")
    st.markdown("""
    **Next-Generation Options Analysis** featuring Monte Carlo simulations, advanced probabilistic models, 
    and comprehensive risk metrics to give you the edge in options trading.
    """)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a stock ticker").upper()
        
        st.subheader("Model Settings")
        model_type = st.selectbox(
            "Pricing Model",
            ["ensemble", "gbm", "jump_diffusion", "heston", "fat_tailed"],
            help="Ensemble combines multiple models for best accuracy"
        )
        
        n_simulations = st.select_slider(
            "Monte Carlo Simulations",
            options=[10000, 25000, 50000, 100000],
            value=50000,
            help="More simulations = more accuracy but slower"
        )
        
        st.subheader("Analysis Type")
        analysis_mode = st.radio(
            "Mode",
            ["Single Options", "Strategy Builder", "🤖 AI Strategy Optimizer"],
            help="Analyze individual options, build strategies, or let AI find the best opportunities"
        )
        
        # Default values
        strategy_type = "long_call"
        optimization_method = "Ensemble Model (Fast)"
        max_strategies = 500
        
        if analysis_mode == "Strategy Builder":
            strategy_type = st.selectbox(
                "Strategy Type",
                [
                    "long_call",
                    "long_put", 
                    "bull_call_spread",
                    "bear_put_spread",
                    "long_straddle",
                    "iron_condor"
                ]
            )
        
        if analysis_mode == "🤖 AI Strategy Optimizer":
            st.subheader("Optimizer Settings")
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["Ensemble Model (Fast)", "Multi-Model Consensus (Thorough)"],
                help="Ensemble uses best model, Consensus tests all models for agreement"
            )
            
            max_strategies = st.slider(
                "Strategies to Test",
                100, 10000, 5000, 
                step=500,
                help="Test thousands of strategy combinations - more = better results but slower"
            )
            
            st.info("💡 The AI will analyze thousands of combinations and show you the top 3 opportunities based on probability and profit potential.")
    
    if not symbol:
        st.warning("Please enter a stock symbol")
        return
    
    with st.spinner(f"Fetching data for {symbol}..."):
        current_price, hist_return, hist_vol = fetch_stock_data(symbol)
        
        if current_price <= 0:
            st.error(f"Could not fetch data for {symbol}")
            return
        
        st.success(f"**{symbol}** - Current Price: **${current_price:.2f}** | " + 
                  f"Historical Return: **{hist_return:.1%}** | Volatility: **{hist_vol:.1%}**")
        
        options_data = fetch_options_data(symbol)
        
        if not options_data:
            st.error("No options data available")
            return
    
    mc_engine = MonteCarloEngine(n_simulations=n_simulations)
    analyzer = AdvancedOptionsAnalyzer(mc_engine)
    
    if analysis_mode == "Single Options":
        st.header("📈 Single Options Analysis")
        
        expirations = sorted(list(set(opt['expirationDate'] for opt in options_data)))
        selected_exp = st.selectbox("Select Expiration", expirations, key="single_exp")
        
        # Guard against 0-day or expired options
        if selected_exp:
            days_check = (datetime.strptime(selected_exp, '%Y-%m-%d') - datetime.now()).days
            if days_check < 1:
                st.error(f"⚠️ Selected expiration ({selected_exp}) has {days_check} days remaining. " +
                        "Options with less than 1 day to expiration are not supported due to model limitations. " +
                        "Please select a later expiration date.")
                st.stop()
        
        exp_options = [opt for opt in options_data if opt['expirationDate'] == selected_exp]
        
        tab_calls, tab_puts = st.tabs(["📞 Calls", "📉 Puts"])
        
        with tab_calls:
            calls = [opt for opt in exp_options if opt['type'] == 'call']
            calls.sort(key=lambda x: x['strike'])
            
            if calls:
                st.subheader("Call Options")
                
                for call in calls[:10]:  # Limit display
                    days_to_exp = (datetime.strptime(call['expirationDate'], '%Y-%m-%d') - datetime.now()).days
                    
                    md = MarketData(
                        current_price=current_price,
                        strike=call['strike'],
                        time_to_expiry=days_to_exp / 365.0,
                        risk_free_rate=0.05,
                        dividend_yield=0.0,
                        historical_volatility=hist_vol,
                        implied_volatility=call['impliedVolatility'],
                        bid=call['bid'],
                        ask=call['ask'],
                        volume=call['volume'],
                        open_interest=call['openInterest'],
                        option_type='call'
                    )
                    
                    with st.expander(f"${call['strike']:.0f} Call - Bid: ${call['bid']:.2f} / Ask: ${call['ask']:.2f}"):
                        metrics = analyzer.analyze_option(md, model_type=model_type, use_implied_vol=False)
                        display_advanced_metrics(metrics, call['ask'])
                        
                        adjusted_premium, liq_details = LiquidityAdjuster.adjust_premium_for_liquidity(
                            call['ask'], call['bid'], call['ask'], 
                            call['volume'], call['openInterest']
                        )
                        
                        st.info(f"""
                        **Liquidity Adjustments:**
                        - Spread Cost: ${liq_details.get('spread_cost', 0):.2f}
                        - Slippage ({liq_details.get('slippage_pct', 0):.1f}%): ${liq_details.get('slippage_cost', 0):.2f}
                        - Commission: ${liq_details.get('commission', 0):.2f}
                        - **Adjusted Premium: ${adjusted_premium:.2f}**
                        """)
        
        with tab_puts:
            puts = [opt for opt in exp_options if opt['type'] == 'put']
            puts.sort(key=lambda x: x['strike'], reverse=True)
            
            if puts:
                st.subheader("Put Options")
                
                for put in puts[:10]:
                    days_to_exp = (datetime.strptime(put['expirationDate'], '%Y-%m-%d') - datetime.now()).days
                    
                    md = MarketData(
                        current_price=current_price,
                        strike=put['strike'],
                        time_to_expiry=days_to_exp / 365.0,
                        risk_free_rate=0.05,
                        dividend_yield=0.0,
                        historical_volatility=hist_vol,
                        implied_volatility=put['impliedVolatility'],
                        bid=put['bid'],
                        ask=put['ask'],
                        volume=put['volume'],
                        open_interest=put['openInterest'],
                        option_type='put'
                    )
                    
                    with st.expander(f"${put['strike']:.0f} Put - Bid: ${put['bid']:.2f} / Ask: ${put['ask']:.2f}"):
                        metrics = analyzer.analyze_option(md, model_type=model_type, use_implied_vol=False)
                        display_advanced_metrics(metrics, put['ask'])
    
    else:  # Strategy Builder
        st.header("🎯 Strategy Builder & Analysis")
        
        expirations = sorted(list(set(opt['expirationDate'] for opt in options_data)))
        
        if not expirations:
            st.error("No expiration dates available")
            return
            
        selected_exp = st.selectbox("Select Expiration", expirations, key="strategy_exp")
        
        if not selected_exp:
            st.warning("Please select an expiration date")
            return
        
        exp_options = [opt for opt in options_data if opt['expirationDate'] == selected_exp]
        
        days_to_exp = (datetime.strptime(selected_exp, '%Y-%m-%d') - datetime.now()).days
        
        # Guard against 0-day or expired options (minimum 1 day)
        if days_to_exp < 1:
            st.error(f"⚠️ Selected expiration ({selected_exp}) has {days_to_exp} days remaining. " +
                    "Options with less than 1 day to expiration are not supported due to model limitations. " +
                    "Please select a later expiration date.")
            return
        
        for opt in exp_options:
            opt['days_to_expiry'] = days_to_exp
            opt['historical_vol'] = hist_vol
            opt['implied_vol'] = opt['impliedVolatility']
        
        legs, strategy_desc = create_strategy_from_options(
            strategy_type, exp_options, current_price
        )
        
        if legs:
            st.success(f"✅ **Strategy Created:** {strategy_desc}")
            
            strategy_calc = OptionsStrategy(mc_engine)
            strategy_metrics = strategy_calc.analyze_strategy(legs, strategy_desc, model_type)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cost", f"${strategy_metrics.total_cost:.2f}")
                st.metric("Max Profit", f"${strategy_metrics.max_profit:.2f}")
            
            with col2:
                st.metric("Max Loss", f"${strategy_metrics.max_loss:.2f}")
                st.metric("Risk/Reward", f"{strategy_metrics.risk_reward_ratio:.2f}")
            
            with col3:
                st.metric("Probability of Profit", f"{strategy_metrics.probability_of_profit:.1f}%")
                st.metric("Expected Value", f"${strategy_metrics.expected_value:.2f}")
            
            with col4:
                st.metric("Edge Score", f"{strategy_metrics.edge_score:.1f}/100")
                st.metric("Sharpe Ratio", f"{strategy_metrics.sharpe_ratio:.2f}")
            
            if strategy_metrics.breakeven_points:
                be_str = ", ".join([f"${be:.2f}" for be in strategy_metrics.breakeven_points])
                st.info(f"**Breakeven Points:** {be_str}")
            
            st.subheader("📊 Payoff Diagram")
            payoff_fig = create_payoff_diagram(legs, strategy_desc, current_price)
            if payoff_fig:
                st.plotly_chart(payoff_fig, use_container_width=True)
            
            with st.expander("📋 Strategy Legs Details"):
                for i, leg in enumerate(legs):
                    position = "LONG" if leg.quantity > 0 else "SHORT"
                    st.write(f"**Leg {i+1}:** {position} {abs(leg.quantity)}x " + 
                            f"{leg.market_data.option_type.upper()} @ ${leg.market_data.strike:.0f}")
        else:
            st.warning("Could not build strategy with available options")
    
    if analysis_mode == "🤖 AI Strategy Optimizer":
        st.header("🤖 AI Strategy Optimizer - Finding Best Opportunities")
        
        expirations = sorted(list(set(opt['expirationDate'] for opt in options_data)))
        
        if not expirations:
            st.error("No expiration dates available")
            return
            
        selected_exp = st.selectbox("Select Expiration", expirations, key="optimizer_exp")
        
        if not selected_exp:
            st.warning("Please select an expiration date")
            return
        
        exp_options = [opt for opt in options_data if opt['expirationDate'] == selected_exp]
        
        days_to_exp = (datetime.strptime(selected_exp, '%Y-%m-%d') - datetime.now()).days
        
        if days_to_exp < 1:
            st.error(f"⚠️ Selected expiration ({selected_exp}) has {days_to_exp} days remaining. " +
                    "Options with less than 1 day to expiration are not supported. " +
                    "Please select a later expiration date.")
            return
        
        for opt in exp_options:
            opt['days_to_expiry'] = days_to_exp
            opt['historical_vol'] = hist_vol
            opt['implied_vol'] = opt['impliedVolatility']
        
        if st.button("🚀 Find Optimal Strategies", type="primary"):
            with st.spinner("🔬 Analyzing thousands of strategy combinations using Monte Carlo simulations..."):
                
                optimizer = AdvancedStrategyOptimizer(mc_engine)
                
                if optimization_method == "Ensemble Model (Fast)":
                    # Use ensemble model only
                    top_strategies = optimizer.optimize_strategies(
                        exp_options,
                        current_price,
                        top_n=3,
                        model_type='ensemble',
                        max_strategies_to_test=max_strategies
                    )
                else:
                    # Multi-model consensus
                    results = optimizer.multi_model_optimization(
                        exp_options,
                        current_price,
                        top_n=3,
                        max_strategies_to_test=max_strategies
                    )
                    top_strategies = results.get('consensus', [])
                
                if not top_strategies:
                    st.warning("Could not find optimal strategies with current options chain")
                else:
                    st.success(f"✅ Found {len(top_strategies)} optimal strategies from analyzing {max_strategies}+ combinations!")
                    
                    for rank, candidate in enumerate(top_strategies, 1):
                        with st.container():
                            st.markdown(f"### 🏆 Rank #{rank}: {candidate.strategy_name}")
                            st.markdown(f"**Composite Score: {candidate.composite_score:.1f}/100**")
                            
                            m = candidate.metrics
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Edge Score", f"{m.edge_score:.1f}/100")
                                st.metric("Total Cost", f"${m.total_cost:.2f}")
                            
                            with col2:
                                st.metric("Probability of Profit", f"{m.probability_of_profit:.1f}%")
                                st.metric("Expected Value", f"${m.expected_value:.2f}")
                            
                            with col3:
                                st.metric("Max Profit", f"${m.max_profit:.2f}")
                                st.metric("Max Loss", f"${m.max_loss:.2f}")
                            
                            with col4:
                                st.metric("Risk/Reward", f"{m.risk_reward_ratio:.2f}:1")
                                st.metric("CVaR (95%)", f"${m.cvar_95:.2f}")
                            
                            if m.breakeven_points:
                                be_str = ", ".join([f"${be:.2f}" for be in m.breakeven_points])
                                st.info(f"**Breakeven Points:** {be_str}")
                            
                            # Payoff diagram
                            payoff_fig = create_payoff_diagram(candidate.legs, candidate.strategy_name, current_price)
                            if payoff_fig:
                                st.plotly_chart(payoff_fig, use_container_width=True)
                            
                            # Explanation
                            with st.expander("📋 Strategy Details & Analysis"):
                                explanation = optimizer.explain_strategy(candidate)
                                st.markdown(explanation)
                            
                            st.markdown("---")
    
    st.markdown("---")
    st.markdown("""
    ### 📚 Model Information
    
    - **GBM**: Geometric Brownian Motion - Standard model assuming log-normal returns
    - **Jump-Diffusion**: Merton model with sudden price jumps for tail risk
    - **Heston**: Stochastic volatility model for realistic vol dynamics  
    - **Fat-Tailed**: Student's t-distribution for extreme events
    - **Ensemble**: Combines all models (recommended for best accuracy)
    
    **Risk Metrics Explained:**
    - **CVaR**: Conditional Value at Risk - average loss in worst 5% scenarios
    - **Sharpe Ratio**: Risk-adjusted return measure
    - **Edge Score**: Combined metric (0-100) factoring in EV, PoP, and risk
    """)


if __name__ == "__main__":
    main()
