"""
alpha_vantage_options_ev_streamlit.py

Purpose:
- Fetch option chain for a chosen US equity symbol from Alpha Vantage (REALTIME_OPTIONS).
- Compute probability-of-profit and expected-value per option using lognormal model.
- Display results in a clean interactive Streamlit dashboard.

Requirements:
- Python 3.8+
- requests, pandas, streamlit
- Set ALPHAVANTAGE_API_KEY as environment variable

Usage:
    streamlit run alpha_vantage_options_ev_streamlit.py
"""
import os
import math
import time
from datetime import datetime, date
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ---------- Helpers ----------
def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def days_to_years(days):
    return days / 365.0

def lognormal_params(S0, mu, sigma_real, T_years):
    m = math.log(S0) + (mu - 0.5 * sigma_real ** 2) * T_years
    s = sigma_real * math.sqrt(T_years)
    return m, s

def prob_greater_than_X(X, m, s):
    if s <= 0:
        return 1.0 if math.exp(m) > X else 0.0
    z = (math.log(X) - m) / s
    return 1.0 - normal_cdf(z)

def expectation_ST_above_Y(Y, m, s):
    if s <= 0:
        return math.exp(m) if math.exp(m) > Y else 0.0
    term = (math.log(Y) - m - s ** 2) / s
    return math.exp(m + 0.5 * s ** 2) * (1.0 - normal_cdf(term))

def conditional_mean_ST_above_Y(Y, m, s):
    p = prob_greater_than_X(Y, m, s)
    if p == 0:
        return 0.0
    return expectation_ST_above_Y(Y, m, s) / p

def get_api_key():
    key = os.getenv('EODHD_API_KEY')
    if not key:
        key = "demo"  # Fallback to demo
    return key

def fetch_yahoo_stock_price(symbol):
    """Fetch current stock price from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return info.get('currentPrice', info.get('regularMarketPrice', 0))

def calculate_historical_metrics(symbol):
    """Calculate historical drift and volatility from stock data"""
    try:
        ticker = yf.Ticker(symbol)
        # Get 1 year of daily data
        hist = ticker.history(period="1y")
        
        if len(hist) < 30:  # Need at least 30 days
            return 0.08, 0.20  # Fallback defaults
        
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Annualize metrics
        annual_return = returns.mean() * 252  # 252 trading days
        annual_volatility = returns.std() * (252 ** 0.5)
        
        # Ensure reasonable bounds
        annual_return = max(-0.5, min(1.0, annual_return))  # Between -50% and 100%
        annual_volatility = max(0.05, min(2.0, annual_volatility))  # Between 5% and 200%
        
        return annual_return, annual_volatility
        
    except Exception:
        return 0.08, 0.20  # Fallback defaults

def fetch_yahoo_options_data(symbol):
    """Fetch real options data from Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    
    # Get all expiration dates
    expirations = ticker.options
    if not expirations:
        raise Exception(f"No options data available for {symbol}")
    
    all_options = []
    
    for exp_date in expirations:
        try:
            # Get options chain for this expiration
            opt_chain = ticker.option_chain(exp_date)
            
            # Process calls
            for _, call in opt_chain.calls.iterrows():
                all_options.append({
                    'contractSymbol': call.get('contractSymbol', ''),
                    'strike': float(call['strike']),
                    'type': 'call',
                    'last': float(call.get('lastPrice', 0)) if pd.notna(call.get('lastPrice')) else None,
                    'bid': float(call.get('bid', 0)) if pd.notna(call.get('bid')) else None,
                    'ask': float(call.get('ask', 0)) if pd.notna(call.get('ask')) else None,
                    'expirationDate': exp_date,
                    'impliedVolatility': float(call.get('impliedVolatility', 0.2)) if pd.notna(call.get('impliedVolatility')) else 0.2,
                    'volume': int(call.get('volume', 0)) if pd.notna(call.get('volume')) else 0,
                    'openInterest': int(call.get('openInterest', 0)) if pd.notna(call.get('openInterest')) else 0
                })
            
            # Process puts
            for _, put in opt_chain.puts.iterrows():
                all_options.append({
                    'contractSymbol': put.get('contractSymbol', ''),
                    'strike': float(put['strike']),
                    'type': 'put',
                    'last': float(put.get('lastPrice', 0)) if pd.notna(put.get('lastPrice')) else None,
                    'bid': float(put.get('bid', 0)) if pd.notna(put.get('bid')) else None,
                    'ask': float(put.get('ask', 0)) if pd.notna(put.get('ask')) else None,
                    'expirationDate': exp_date,
                    'impliedVolatility': float(put.get('impliedVolatility', 0.2)) if pd.notna(put.get('impliedVolatility')) else 0.2,
                    'volume': int(put.get('volume', 0)) if pd.notna(put.get('volume')) else 0,
                    'openInterest': int(put.get('openInterest', 0)) if pd.notna(put.get('openInterest')) else 0
                })
                
        except Exception as e:
            st.warning(f"Could not fetch options for expiration {exp_date}: {e}")
            continue
    
    return {"data": all_options}

def create_demo_options_data(current_price=234.35):
    """Create realistic demo options data for testing"""
    from datetime import datetime, timedelta
    import random
    
    # Create options for multiple expirations (weekly and monthly)
    today = datetime.now()
    expirations = []
    
    # Add weekly expirations for next 8 weeks
    for weeks in range(1, 9):
        exp_date = today + timedelta(weeks=weeks)
        # Move to Friday
        exp_date = exp_date + timedelta(days=(4 - exp_date.weekday()) % 7)
        expirations.append(exp_date.strftime('%Y-%m-%d'))
    
    # Add monthly expirations for next 6 months
    for months in range(3, 13, 3):  # 3, 6, 9, 12 months out
        exp_date = today + timedelta(days=months*30)
        # Move to third Friday of month
        exp_date = exp_date.replace(day=15)
        exp_date = exp_date + timedelta(days=(4 - exp_date.weekday()) % 7)
        expirations.append(exp_date.strftime('%Y-%m-%d'))
    
    # Remove duplicates and sort
    expirations = sorted(list(set(expirations)))
    
    options_data = {"data": []}
    
    # Generate strikes around current price with wider range
    price_range = max(50, current_price * 0.3)  # At least $50 range or 30% of price
    min_strike = max(5, int((current_price - price_range) / 5) * 5)
    max_strike = int((current_price + price_range) / 5) * 5
    strikes = list(range(min_strike, max_strike + 5, 5))  # Every $5
    
    for exp_date in expirations:
        for strike in strikes:
            days_to_exp = (datetime.strptime(exp_date, '%Y-%m-%d') - today).days
            
            # Calculate realistic option prices using simple Black-Scholes approximation
            moneyness = strike / current_price
            time_value = max(0.01, 0.1 * (days_to_exp / 30) * abs(moneyness - 1) + 0.02)
            
            # Call option
            call_intrinsic = max(0, current_price - strike)
            call_price = call_intrinsic + time_value
            call_bid = max(0.01, call_price - 0.05)
            call_ask = call_price + 0.05
            
            # Put option  
            put_intrinsic = max(0, strike - current_price)
            put_price = put_intrinsic + time_value
            put_bid = max(0.01, put_price - 0.05)
            put_ask = put_price + 0.05
            
            # Add call
            options_data["data"].append({
                "contractSymbol": f"{exp_date.replace('-', '')[:6]}C{strike:08.0f}",
                "strike": strike,
                "currency": "USD",
                "type": "call",
                "last": round(call_price, 2),
                "bid": round(call_bid, 2),
                "ask": round(call_ask, 2),
                "expirationDate": exp_date,
                "impliedVolatility": round(0.15 + random.uniform(-0.05, 0.05), 4)
            })
            
            # Add put
            options_data["data"].append({
                "contractSymbol": f"{exp_date.replace('-', '')[:6]}P{strike:08.0f}",
                "strike": strike,
                "currency": "USD", 
                "type": "put",
                "last": round(put_price, 2),
                "bid": round(put_bid, 2),
                "ask": round(put_ask, 2),
                "expirationDate": exp_date,
                "impliedVolatility": round(0.15 + random.uniform(-0.05, 0.05), 4)
            })
    
    return options_data

def compute_ev_for_option_row(option, S0, mu, sigma_real, today):
    try:
        strike = float(option.get('strike', option.get('strikePrice', option.get('strike_price'))))
    except Exception:
        return None

    bid = option.get('bid')
    ask = option.get('ask')
    last = option.get('last') or option.get('lastPrice')
    try:
        bid = float(bid) if bid not in (None, '', 'null') else None
        ask = float(ask) if ask not in (None, '', 'null') else None
        last = float(last) if last not in (None, '', 'null') else None
    except Exception:
        bid = ask = last = None
    # Use ask price as premium for buying options
    premium = ask if ask is not None else last if last is not None else None
    if premium is None:
        return None

    exp_raw = option.get('expirationDate') or option.get('expiration_date') or option.get('expiration')
    if not exp_raw:
        return None
    exp_date = None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            exp_date = datetime.strptime(exp_raw, fmt).date()
            break
        except Exception:
            continue
    if exp_date is None:
        try:
            exp_date = date.fromisoformat(exp_raw)
        except Exception:
            return None

    days = (exp_date - today).days
    T = days_to_years(days) if days>0 else 1/365.0
    
    # Get option type
    opt_type = (option.get('optionType') or option.get('type') or option.get('putCall') or '').lower()
    is_call = 'call' in opt_type or opt_type == 'c'
    
    # Step 1: Lognormal Distribution Parameters (using exact formula)
    m = math.log(S0) + (mu - 0.5 * sigma_real ** 2) * T
    s = sigma_real * math.sqrt(T)
    
    # Step 2: Breakeven Price
    if is_call:
        X = strike + premium  # For calls: K + C
    else:
        X = strike - premium  # For puts: K - C
    
    # Step 3: Probability of Profit (using exact formula)
    if is_call:
        # For calls: P(S_T > X)
        if s <= 0:
            p_profit = 1.0 if S0 > X else 0.0
        else:
            z = (math.log(X) - m) / s
            p_profit = 1.0 - normal_cdf(z)
    else:
        # For puts: P(S_T < X)
        if s <= 0:
            p_profit = 1.0 if S0 < X else 0.0
        else:
            z = (math.log(X) - m) / s
            p_profit = normal_cdf(z)
    
    # Step 4: Conditional Expected Payoff (using exact formula)
    if is_call:
        if s <= 0:
            exp_ST_cond = math.exp(m) if math.exp(m) > X else 0.0
        else:
            numerator = math.exp(m + 0.5 * s ** 2) * (1.0 - normal_cdf((math.log(X) - m - s ** 2) / s))
            denominator = 1.0 - normal_cdf((math.log(X) - m) / s)
            exp_ST_cond = numerator / denominator if denominator > 0 else 0.0
        avg_payoff_if_win = max(exp_ST_cond - X, 0.0)
    else:
        # For puts: E[S_T | S_T < X]
        if s <= 0:
            exp_ST_cond = math.exp(m) if math.exp(m) < X else 0.0
        else:
            numerator = math.exp(m + 0.5 * s ** 2) * normal_cdf((math.log(X) - m - s ** 2) / s)
            denominator = normal_cdf((math.log(X) - m) / s)
            exp_ST_cond = numerator / denominator if denominator > 0 else 0.0
        avg_payoff_if_win = max(X - exp_ST_cond, 0.0)
    
    # Step 5: Expected Value per Share (using exact formula)
    EV = p_profit * avg_payoff_if_win - premium
    
    # Step 6: Additional metrics
    required_move_pct = (X / S0 - 1) * 100
    
    # Calculate mid price
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    
    # ===== ENHANCED: DUAL VOLATILITY CALCULATIONS =====
    # Now also calculate using IMPLIED VOLATILITY for comparison
    iv = option.get('impliedVolatility') or option.get('iv') or option.get('implied_volatility')
    try: 
        implied_vol = float(iv) if iv not in (None, '', 'null') else 0.2
    except Exception: 
        implied_vol = 0.2
    
    # Calculate with IMPLIED VOLATILITY (market's forecast)
    m_iv = math.log(S0) + (mu - 0.5 * implied_vol ** 2) * T
    s_iv = implied_vol * math.sqrt(T)
    
    # Breakeven is same (based on premium)
    # Probability of profit using IMPLIED VOL
    if is_call:
        if s_iv <= 0:
            p_profit_iv = 1.0 if S0 > X else 0.0
        else:
            z_iv = (math.log(X) - m_iv) / s_iv
            p_profit_iv = 1.0 - normal_cdf(z_iv)
    else:
        if s_iv <= 0:
            p_profit_iv = 1.0 if S0 < X else 0.0
        else:
            z_iv = (math.log(X) - m_iv) / s_iv
            p_profit_iv = normal_cdf(z_iv)
    
    # Expected payoff using IMPLIED VOL
    if is_call:
        if s_iv <= 0:
            exp_ST_cond_iv = math.exp(m_iv) if math.exp(m_iv) > X else 0.0
        else:
            numerator_iv = math.exp(m_iv + 0.5 * s_iv ** 2) * (1.0 - normal_cdf((math.log(X) - m_iv - s_iv ** 2) / s_iv))
            denominator_iv = 1.0 - normal_cdf((math.log(X) - m_iv) / s_iv)
            exp_ST_cond_iv = numerator_iv / denominator_iv if denominator_iv > 0 else 0.0
        avg_payoff_if_win_iv = max(exp_ST_cond_iv - X, 0.0)
    else:
        if s_iv <= 0:
            exp_ST_cond_iv = math.exp(m_iv) if math.exp(m_iv) < X else 0.0
        else:
            numerator_iv = math.exp(m_iv + 0.5 * s_iv ** 2) * normal_cdf((math.log(X) - m_iv - s_iv ** 2) / s_iv)
            denominator_iv = normal_cdf((math.log(X) - m_iv) / s_iv)
            exp_ST_cond_iv = numerator_iv / denominator_iv if denominator_iv > 0 else 0.0
        avg_payoff_if_win_iv = max(X - exp_ST_cond_iv, 0.0)
    
    # Expected Value using IMPLIED VOL
    EV_iv = p_profit_iv * avg_payoff_if_win_iv - premium
    
    # Calculate volatility edge (our forecast vs market's)
    vol_edge = sigma_real - implied_vol
    edge_signal = "🔥BUY" if vol_edge > 0.05 else "💸SELL" if vol_edge < -0.05 else "⚖️NEUTRAL"
    
    res = {
        'contractSymbol': option.get('contractSymbol') or option.get('contract'),
        'optionType': 'Call' if is_call else 'Put',
        'strike': strike,
        'premium': premium,
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'breakeven': X,
        'days_to_expiry': days,
        'expiration_date': exp_date.strftime('%b %d, %Y') if exp_date else '',
        
        # HISTORICAL VOLATILITY METRICS (our model)
        'prob_profit_pct': p_profit * 100,
        'avg_payoff_if_win': avg_payoff_if_win,
        'EV_per_share': EV,
        
        # IMPLIED VOLATILITY METRICS (market model)
        'prob_profit_pct_iv': p_profit_iv * 100,
        'avg_payoff_if_win_iv': avg_payoff_if_win_iv,
        'EV_per_share_iv': EV_iv,
        
        # VOLATILITY COMPARISON
        'implied_vol_pct': implied_vol * 100,
        'historical_vol_pct': sigma_real * 100,
        'vol_edge_pct': vol_edge * 100,
        'edge_signal': edge_signal,
        
        'required_move_pct': required_move_pct
    }
    
    return res

# ---------- Streamlit app ----------
st.set_page_config(page_title="Options EV Dashboard", layout='wide')
st.title("Options Probability & EV Dashboard")
st.write("Calculate probability of profit and expected value for options using lognormal distribution")
st.write("🔥 **Real-time options data** powered by Yahoo Finance")

# Quality thresholds explanation
st.info("""
📊 **Quality Ratings:** ✅ GOOD BUY (EV > $0.10) | ⚠️ MARGINAL (EV > $0) | ❌ AVOID (EV ≤ $0)  
💡 **Smart Filtering:** Only considers reasonably-priced options near current stock price
""")

symbol = st.text_input('Stock Symbol', placeholder='Enter symbol (e.g., AAPL)', value='AAPL').upper()

if symbol:
    st.info(f'Fetching data for {symbol}...')
    
    # Fetch real stock price from Yahoo Finance
    try:
        S0 = fetch_yahoo_stock_price(symbol)
        if S0 <= 0:
            raise Exception("Invalid price")
    except Exception as e:
        st.error(f"Could not fetch stock price for {symbol}: {e}")
        st.stop()

    if S0 is not None and S0 > 0:
        st.success(f'Underlying price S0 = {S0:.4f}')
    else:
        st.error('Could not fetch underlying stock price')
        st.stop()

    # Calculate real historical metrics
    with st.spinner('Calculating historical metrics...'):
        historical_mu, historical_sigma = calculate_historical_metrics(symbol)
    
    st.info(f"📈 Historical Analysis: {historical_mu:.1%} annual return, {historical_sigma:.1%} volatility")
    
    with st.expander("📊 Advanced Settings (Optional)", expanded=False):
        mu = st.number_input('Expected annual drift μ (decimal)', value=historical_mu, help="Expected annual return of the stock")
        sigma_real = st.number_input('Expected annual volatility σ (decimal)', value=historical_sigma, help="Expected annual volatility of the stock")
    
    # Use the values (either default historical or user-modified)
    if 'mu' not in locals():
        mu = historical_mu
    if 'sigma_real' not in locals():
        sigma_real = historical_sigma

    # Fetch real options data from Yahoo Finance
    try:
        with st.spinner(f'Fetching options data for {symbol}...'):
            options_json = fetch_yahoo_options_data(symbol)
        
        cand = options_json.get('data', [])
        if not cand:
            st.error(f'No options data available for {symbol}')
            st.stop()
            
        num_expirations = len(set([opt.get('expirationDate') for opt in cand]))
        st.success(f"✅ Found {len(cand)} real options contracts across {num_expirations} expiration dates")
        
    except Exception as e:
        st.error(f"Could not fetch options data for {symbol}: {str(e)}")
        st.info("💡 Try a different symbol like AAPL, MSFT, TSLA, NVDA, etc.")
        st.stop()

    today = date.today()
    results = []
    for opt in cand:
        res = compute_ev_for_option_row(opt, S0, mu, sigma_real, today)
        if res: results.append(res)

    if results:
        df_out = pd.DataFrame(results)
        
        # Group by expiration date
        expiration_dates = sorted(df_out['expiration_date'].unique())
        
        # Create tabs for different expiration dates
        if len(expiration_dates) > 1:
            selected_expiration = st.selectbox("Select Expiration Date", expiration_dates)
        else:
            selected_expiration = expiration_dates[0] if expiration_dates else None
        
        if selected_expiration:
            # Filter data for selected expiration
            exp_data = df_out[df_out['expiration_date'] == selected_expiration].copy()
            
            # Separate calls and puts
            calls_data = exp_data[exp_data['optionType'] == 'Call'].copy()
            puts_data = exp_data[exp_data['optionType'] == 'Put'].copy()
            calls_data = calls_data.sort_values('strike') if not calls_data.empty else calls_data
            puts_data = puts_data.sort_values('strike') if not puts_data.empty else puts_data
            
            # Enhanced dual-volatility options chain table
            st.subheader(f"Options Chain - {selected_expiration}")
            st.write(f"**Historical Analysis:** μ = {mu:.1%} annual drift, σ = {sigma_real:.1%} annual volatility")
            
            # Volatility comparison info box
            st.info("📊 **Dual Volatility Analysis:** POP(H) = Historical model | POP(IV) = Market implied | Edge = Trading signal")
            
            # Get all unique strikes from both calls and puts
            all_strikes = sorted(set(list(calls_data['strike']) + list(puts_data['strike'])))
            
            # Create the table data
            table_data = []
            for strike in all_strikes:
                # Get call data for this strike
                call_row = calls_data[calls_data['strike'] == strike]
                put_row = puts_data[puts_data['strike'] == strike]
                
                # Call data with dual volatility
                if not call_row.empty:
                    call_bid = call_row['bid'].values[0] if len(call_row) > 0 and not pd.isna(call_row['bid'].values[0]) else "-"
                    call_ask = call_row['ask'].values[0] if len(call_row) > 0 and not pd.isna(call_row['ask'].values[0]) else "-"
                    call_prob_hist = call_row['prob_profit_pct'].values[0] if len(call_row) > 0 else None
                    call_prob_iv = call_row['prob_profit_pct_iv'].values[0] if len(call_row) > 0 else None
                    call_edge = call_row['edge_signal'].values[0] if len(call_row) > 0 else "-"
                else:
                    call_bid = call_ask = "-"
                    call_prob_hist = call_prob_iv = None
                    call_edge = "-"
                
                # Put data with dual volatility
                if not put_row.empty:
                    put_bid = put_row['bid'].values[0] if len(put_row) > 0 and not pd.isna(put_row['bid'].values[0]) else "-"
                    put_ask = put_row['ask'].values[0] if len(put_row) > 0 and not pd.isna(put_row['ask'].values[0]) else "-"
                    put_prob_hist = put_row['prob_profit_pct'].values[0] if len(put_row) > 0 else None
                    put_prob_iv = put_row['prob_profit_pct_iv'].values[0] if len(put_row) > 0 else None
                    put_edge = put_row['edge_signal'].values[0] if len(put_row) > 0 else "-"
                else:
                    put_bid = put_ask = "-"
                    put_prob_hist = put_prob_iv = None
                    put_edge = "-"
                
                # Format numbers
                if call_bid != "-": call_bid = f"${call_bid:.2f}"
                if call_ask != "-": call_ask = f"${call_ask:.2f}"
                if put_bid != "-": put_bid = f"${put_bid:.2f}"
                if put_ask != "-": put_ask = f"${put_ask:.2f}"
                
                table_data.append({
                    'C Bid': call_bid,
                    'C Ask': call_ask,
                    'C POP(H)': f"{call_prob_hist:.0f}%" if call_prob_hist else "-",
                    'C POP(IV)': f"{call_prob_iv:.0f}%" if call_prob_iv else "-",
                    'C Edge': call_edge,
                    'Strike': f"${strike:.0f}",
                    'P Edge': put_edge,
                    'P POP(H)': f"{put_prob_hist:.0f}%" if put_prob_hist else "-",
                    'P POP(IV)': f"{put_prob_iv:.0f}%" if put_prob_iv else "-",
                    'P Bid': put_bid,
                    'P Ask': put_ask
                })
            
            # Find best options with quality filters
            best_call = None
            best_put = None
            best_call_score = -float('inf')
            best_put_score = -float('inf')
            
            # Filter calls: should be reasonably close to current price (within 50% up/down)
            reasonable_call_range = (S0 * 0.7, S0 * 1.5)
            for _, row in calls_data.iterrows():
                if (reasonable_call_range[0] <= row['strike'] <= reasonable_call_range[1] and 
                    row['EV_per_share'] > best_call_score):
                    best_call_score = row['EV_per_share']
                    best_call = row['strike']
                    
            # Filter puts: should be reasonably close to current price (within 50% up/down)  
            reasonable_put_range = (S0 * 0.5, S0 * 1.3)
            for _, row in puts_data.iterrows():
                if (reasonable_put_range[0] <= row['strike'] <= reasonable_put_range[1] and 
                    row['EV_per_share'] > best_put_score):
                    best_put_score = row['EV_per_share']
                    best_put = row['strike']
            
            # Style the dataframe with highlighting
            def highlight_best(row):
                styles = [''] * len(row)
                strike_val = float(row['Strike'].replace('$', ''))
                
                # Highlight best call (green background for call columns)
                if best_call and abs(strike_val - best_call) < 0.01:
                    styles[0] = 'background-color: #90EE90'  # C Bid
                    styles[1] = 'background-color: #90EE90'  # C Ask  
                    styles[2] = 'background-color: #90EE90'  # C POP(H)
                    styles[3] = 'background-color: #90EE90'  # C POP(IV)
                    styles[4] = 'background-color: #90EE90'  # C Edge
                
                # Highlight best put (green background for put columns)
                if best_put and abs(strike_val - best_put) < 0.01:
                    styles[6] = 'background-color: #90EE90'  # P Edge
                    styles[7] = 'background-color: #90EE90'  # P POP(H)
                    styles[8] = 'background-color: #90EE90'  # P POP(IV)
                    styles[9] = 'background-color: #90EE90'  # P Bid
                    styles[10] = 'background-color: #90EE90'  # P Ask
                    
                return styles
            
            # Display the table with highlighting
            chain_df = pd.DataFrame(table_data)
            styled_df = chain_df.style.apply(highlight_best, axis=1)
            st.dataframe(styled_df, width='stretch', hide_index=True)
            
            # Enhanced recommendation system with quality filters
            col1, col2 = st.columns(2)
            with col1:
                if best_call is not None:
                    # Quality check for calls: EV should be positive and strike reasonable
                    call_quality = "✅ GOOD BUY" if best_call_score > 0.10 else "⚠️ MARGINAL" if best_call_score > 0 else "❌ AVOID"
                    st.success(f"🔥 **Best Call:** ${best_call:.0f} strike (EV: ${best_call_score:.2f}) - {call_quality}")
                else:
                    st.warning("No good call options found")
            
            with col2:
                if best_put is not None:
                    # Quality check for puts: EV should be positive and strike reasonable relative to stock price
                    put_quality = "✅ GOOD BUY" if best_put_score > 0.10 else "⚠️ MARGINAL" if best_put_score > 0 else "❌ AVOID"
                    st.success(f"🔥 **Best Put:** ${best_put:.0f} strike (EV: ${best_put_score:.2f}) - {put_quality}")
                else:
                    st.warning("No good put options found")
            
            # Professional explanation box
            st.markdown("""
            **Legend:**
            - **POP(H):** Probability using your historical volatility forecast
            - **POP(IV):** Probability using market's implied volatility  
            - **Edge:** 🔥BUY = underpriced (IV < historical) | 💸SELL = overpriced (IV > historical) | ⚖️NEUTRAL = fairly valued
            - **Green highlighting:** Best expected value opportunities based on historical model
            """)
            
            # Summary for selected expiration
            st.subheader("Summary for Selected Expiration")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Call Options", len(calls_data))
            with col2: 
                st.metric("Put Options", len(puts_data))
            with col3:
                call_pos_ev = len(calls_data[calls_data['EV_per_share'] > 0])
                st.metric("Calls +EV", call_pos_ev)
            with col4:
                put_pos_ev = len(puts_data[puts_data['EV_per_share'] > 0])
                st.metric("Puts +EV", put_pos_ev)
    else:
        st.warning('No option results computed.')
