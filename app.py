import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

# Black-Scholes Option Pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    return price

# Delta Calculation
def delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return si.norm.cdf(d1)
    else:
        return si.norm.cdf(d1) - 1

# Simulate Delta Hedging
def delta_hedging(S0, K, T, r, sigma, option_type="call", N=100, trading_cost=0.002):
    dt = T / N  # Time step
    t = np.linspace(0, T, N + 1)
    S = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(N)))
    S = np.insert(S, 0, S0)  # Insert initial stock price

    delta_hedge = [delta(S[0], K, T, r, sigma, option_type)]
    portfolio_value = [0]
    cash = [-delta_hedge[0] * S[0]]  # Shorting stock to hedge

    for i in range(1, N + 1):
        T_remaining = T - t[i]
        new_delta = delta(S[i], K, T_remaining, r, sigma, option_type)
        trading_cost = max(0.001, 0.0005 * (1 + np.random.rand()))
        transaction_cost = abs(new_delta - delta_hedge[-1]) * S[i] * trading_cost
        cash.append(cash[-1] * np.exp(r * dt) - (new_delta - delta_hedge[-1]) * S[i] - transaction_cost)
        delta_hedge.append(new_delta)
        portfolio_value.append(delta_hedge[-1] * S[i] + cash[-1])

    option_price = black_scholes(S0, K, T, r, sigma, option_type)

    plt.figure(figsize=(10, 5))
    plt.plot(t, portfolio_value, label="Hedged Portfolio Value")
    plt.axhline(option_price, color='r', linestyle='--', label="Option Price")
    plt.xlabel("Time to Expiry")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Delta Hedging Simulation")
    plt.show()

    return portfolio_value, option_price, t


# Fetch Stock Data from Yahoo Finance
#def get_stock_data(ticker):
#    stock = yf.Ticker(ticker)
#    hist = stock.history(period="1y")
#    S0 = hist["Close"].iloc[-1]  # Latest closing price
#    sigma = np.std(np.log(hist["Close"] / hist["Close"].shift(1))) * np.sqrt(252)  # Annualized volatility
#    if sigma == 0 or np.isnan(sigma):  # Handle cases where sigma might be zero
#        sigma = 0.01  # Assign a small default volatility
#    return S0, sigma


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")  # Use last 6 months instead
    S0 = hist["Close"].iloc[-1]
    sigma = hist["Close"].pct_change().rolling(30).std().dropna().mean() * np.sqrt(252)
    return S0, sigma

# Streamlit UI
st.title("Options Delta Hedging Simulator")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
T = st.number_input("Time to Expiry (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.number_input("Risk-Free Rate (as decimal)", min_value=0.0, max_value=0.1, value=0.05, step=0.001)

if st.button("Run Delta Hedging Simulation"):
    S0, sigma = get_stock_data(ticker)
    K = S0  # At-the-money option
    
    # Run delta hedging for both call and put
    call_portfolio, call_option_price, t = delta_hedging(S0, K, T, r, sigma, "call", N=500, trading_cost=0.002)
    put_portfolio, put_option_price, _ = delta_hedging(S0, K, T, r, sigma, "put", N=500, trading_cost=0.002)
    
    # Display Call Option Price
    st.write(f"### Call Option Price: ${call_option_price:.2f}")
    fig_call, ax_call = plt.subplots(figsize=(10, 5))
    ax_call.plot(t, call_portfolio, label="Hedged Call Portfolio Value")
    ax_call.axhline(call_option_price, color='r', linestyle='--', label="Call Option Price")
    ax_call.set_xlabel("Time to Expiry")
    ax_call.set_ylabel("Value")
    ax_call.legend()
    ax_call.set_title("Delta Hedging Simulation for Call Option")
    st.pyplot(fig_call)
    
    # Display Put Option Price
    st.write(f"### Put Option Price: ${put_option_price:.2f}")
    fig_put, ax_put = plt.subplots(figsize=(10, 5))
    ax_put.plot(t, put_portfolio, label="Hedged Put Portfolio Value", linestyle='dotted')
    ax_put.axhline(put_option_price, color='b', linestyle='--', label="Put Option Price")
    ax_put.set_xlabel("Time to Expiry")
    ax_put.set_ylabel("Value")
    ax_put.legend()
    ax_put.set_title("Delta Hedging Simulation for Put Option")
    st.pyplot(fig_put)
