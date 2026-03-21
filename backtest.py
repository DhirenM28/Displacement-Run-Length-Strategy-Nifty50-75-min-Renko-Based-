# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:47:25 2026

@author: Admin
"""

def run_backtest(df, run_thresh=RUN_THRESH, adx_thresh=ADX_THRESH, trail_mult=TRAIL_MULT):
    """Execute backtest with ATR-based trailing stops only."""
    df = df.copy()
    

    df['long_condition'] = (df['run_length'] >= run_thresh) & (df['ADX'] > adx_thresh) & (df['RSI'] > 50)
    df['short_condition'] = (df['run_length'] <= -run_thresh) & (df['ADX'] > adx_thresh) & (df['RSI'] < 50) & (df['RSI'] > 33)
    df['is_first_candle'] = (df['Datetime'].dt.hour == 9) & (df['Datetime'].dt.minute == 15)
    
    position = None
    entry_price = None
    entry_datetime = None
    entry_adx = None
    exit_adx = None
    trades = []
    swing_high = np.nan
    swing_low = np.nan
    long_trail = np.nan
    short_trail = np.nan
    
    for idx, row in df.iterrows():
        dt = row['Datetime']
        close = row['Close']
        high = row['High']
        low = row['Low']
        open_price = row['Open']
        atr = row['ATR']
        is_first_candle = row['is_first_candle']
        is_last_candle = row['is_last_candle']
        
        # Gap handling
        if is_first_candle and position is not None and not is_last_candle:
            if position == 'LONG' and not np.isnan(long_trail) and not is_last_candle:
                if open_price < long_trail:
                    trades.append({
                        'Entry_Datetime': entry_datetime,
                        'Entry_price': entry_price,
                        'Exit_Datetime': dt,
                        'Exit_Price': open_price,
                        'Type': 'LONG',
                        'PnL': open_price - entry_price,
                        'PnL_Pct': ((open_price - entry_price) / entry_price) * 100,
                        'Exit_Reason': 'Gap Down - SL Hit',
                        'Entry_ADX': entry_adx,
                        'Exit_ADX': row['ADX'],
                    })
                    position = None
                    swing_high = np.nan
                    long_trail = np.nan
            elif position == 'SHORT' and not np.isnan(short_trail):
                if open_price > short_trail:
                    trades.append({
                        'Entry_Datetime': entry_datetime,
                        'Entry_price': entry_price,
                        'Exit_Datetime': dt,
                        'Exit_Price': open_price,
                        'Type': 'SHORT',
                        'PnL': entry_price - open_price,
                        'PnL_Pct': ((entry_price - open_price) / entry_price) * 100,
                        'Exit_Reason': 'Gap Up - SL Hit',
                        'Entry_ADX': entry_adx,
                        'Exit_ADX': row['ADX'],
                    })
                    position = None
                    swing_low = np.nan
                    short_trail = np.nan
        
        # Update trail stops
        if position == 'LONG':
            swing_high = np.nanmax([swing_high, high])
            if not np.isnan(swing_high) and not np.isnan(atr):
                new_trail = swing_high - trail_mult * atr
                long_trail = max(long_trail, new_trail) if not np.isnan(long_trail) else new_trail
            
            if not is_last_candle and not np.isnan(long_trail) and close < long_trail:

                trades.append({
                    'Entry_Datetime': entry_datetime,
                    'Entry_price': entry_price,
                    'Exit_Datetime': dt,
                    'Exit_Price': close,
                    'Type': 'LONG',
                    'PnL': close - entry_price,
                    'PnL_Pct': ((close - entry_price) / entry_price) * 100,
                    'Exit_Reason': 'Trail_Stop',
                    'Entry_ADX': entry_adx,
                    'Exit_ADX': row['ADX'],
                })
                position = None
                swing_high = np.nan
                long_trail = np.nan
        
        elif position == 'SHORT':
            swing_low = np.nanmin([swing_low, low])
            if not np.isnan(swing_low) and not np.isnan(atr):
                new_trail = swing_low + trail_mult * atr
                short_trail = min(short_trail, new_trail) if not np.isnan(short_trail) else new_trail
            
            if not is_last_candle and not np.isnan(short_trail) and close > short_trail:

                trades.append({
                    'Entry_Datetime': entry_datetime,
                    'Entry_price': entry_price,
                    'Exit_Datetime': dt,
                    'Exit_Price': close,
                    'Type': 'SHORT',
                    'PnL': entry_price - close,
                    'PnL_Pct': ((entry_price - close) / entry_price) * 100,
                    'Exit_Reason': 'Trail_Stop',
                    'Entry_ADX': entry_adx,
                    'Exit_ADX': row['ADX'],
                })
                position = None
                swing_low = np.nan
                short_trail = np.nan
        
        # Entry logic
        if position is None and not is_last_candle:
            if row['long_condition']:
                position = 'LONG'
                entry_price = close
                entry_datetime = dt
                swing_high = high
                long_trail = np.nan  # No initial stop, only trailing
                entry_adx = row['ADX']
                
            elif row['short_condition']:
                position = 'SHORT'
                entry_price = close
                entry_datetime = dt
                swing_low = low
                short_trail = np.nan  # No initial stop, only trailing
                entry_adx = row['ADX']
        
        # Signal exits/reversals
        elif position == 'LONG' and not is_last_candle:
            if row['short_condition']:
                trades.append({
                    'Entry_Datetime': entry_datetime,
                    'Entry_price': entry_price,
                    'Exit_Datetime': dt,
                    'Exit_Price': close,
                    'Type': 'LONG',
                    'PnL': close - entry_price,
                    'PnL_Pct': ((close - entry_price) / entry_price) * 100,
                    'Exit_Reason': 'Reverse_to_Short',
                    'Entry_ADX': entry_adx,
                    'Exit_ADX': row['ADX'],                
                })
                position = 'SHORT'
                entry_price = close
                entry_datetime = dt
                swing_low = low
                short_trail = np.nan
                swing_high = np.nan
                long_trail = np.nan
                entry_adx = row['ADX']
        
        elif position == 'SHORT' and not is_last_candle:
            if row['long_condition']:
                trades.append({
                    'Entry_Datetime': entry_datetime,
                    'Entry_price': entry_price,
                    'Exit_Datetime': dt,
                    'Exit_Price': close,
                    'Type': 'SHORT',
                    'PnL': entry_price - close,
                    'PnL_Pct': ((entry_price - close) / entry_price) * 100,
                    'Exit_Reason': 'Reverse_to_Long',
                    'Entry_ADX': entry_adx,
                    'Exit_ADX': row['ADX'],
                    
                    
                })
                position = 'LONG'
                entry_price = close
                entry_datetime = dt
                swing_high = high
                long_trail = np.nan
                swing_low = np.nan
                short_trail = np.nan
                entry_adx = row['ADX']
    
    # Close any open position at end
    if position is not None:
        last_close = df.iloc[-1]['Close']
        last_dt = df.iloc[-1]['Datetime']
        last_adx = df.iloc[-1]['ADX']

        pnl = (last_close - entry_price) if position == 'LONG' else (entry_price - last_close)
        trades.append({
            'Entry_Datetime': entry_datetime,
            'Entry_price': entry_price,
            'Exit_Datetime': last_dt,
            'Exit_Price': last_close,
            'Type': position,
            'PnL': pnl,
            'PnL_Pct': (pnl / entry_price) * 100,
            'Exit_Reason': 'End_of_Data',
            'Entry_ADX': entry_adx,
            'Exit_ADX': last_adx,

        })
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Entry_Datetime'] = pd.to_datetime(trades_df['Entry_Datetime'])
        trades_df['Exit_Datetime'] = pd.to_datetime(trades_df['Exit_Datetime'])
        trades_df['Entry_date'] = trades_df['Entry_Datetime']
        trades_df['Entry_time'] = trades_df['Entry_Datetime']
        trades_df['Exit_date'] = trades_df['Exit_Datetime']
        trades_df['Exit_time'] = trades_df['Exit_Datetime']
        trades_df['Signal']=trades_df['Type']
    return trades_df
