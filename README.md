# Displacement-Run-Length-Strategy-Nifty50-75-min-Renko-Based-
This project implements a volatility-normalized trend-following framework using a Displacement &amp; Run-Length model, combined with ADX and RSI filters, and executed via systematic options spreads on index derivatives (NIFTY).

Strategy Intuition
Markets do not move in absolute price terms.
They move relative to volatility.

This strategy:
-Converts price movement into standardized “brick units”
-Measures directional strength via run-length scoring
-Filters trades using trend strength (ADX) and momentum (RSI)

Core Components
1. Volatility-Normalized Displacement

    Price movement is scaled using dynamic brick size:
    Brick Size = SMA(Close, n) × BRICK_PCT
    Displacement = (Price – Reversal Level) / Brick Size



2. Run-Length (Trend Strength)

    Displacement is discretized into integer trend scores:
  
    Displacement	Run-Length
    < 1 brick	±1
    1–2 bricks	±2
    2–3 bricks	±3
    
    Positive → Uptrend
    
    Negative → Downtrend



3. Trade Filters

    Long Entry
    
    Run-Length > 1
    
    ADX > 15
    
    RSI > 50
    
    Short Entry
    
    Run-Length < -1
    
    ADX > 15
    
    RSI between 33–50

4.Performace Metrics
    CAGR-45%
    sharpe ratio-1.64
    calmar ratio-2.32
    max drawdown-10.5%
    WIN RATE-48%
