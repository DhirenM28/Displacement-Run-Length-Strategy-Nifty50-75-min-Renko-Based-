# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:47:35 2026

@author: Admin
"""



# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
# pip install pandas

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "NIFTY 50_minute.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "debashis74017/nifty-50-minute-data",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


import pandas as pd

df.set_index("date",inplace=True)
df.index=pd.to_datetime(df.index)

resample_spot=df.resample("75min",origin="start_day",offset="9h15min").agg({
    'open': "first",
    'high': 'max',
    'low': 'min',
    'close' : 'last'}).dropna()



import numpy as np
import talib
import datetime as dt

RUN_THRESH = 1
ADX_THRESH = 15.00
TRAIL_MULT = 0.95
WINDOW = 10
BRICK_PCT = 0.0015
PERIOD = 14

df_raw=resample_spot
df_raw.columns=['Open','High','Low','Close']
df_raw["Datetime"]=df_raw.index

df_raw['date_only'] = df_raw['Datetime'].dt.date
df_raw['is_last_candle'] = df_raw['date_only'] != df_raw['date_only'].shift(-1)



def calculate_brick_size(df, window=WINDOW, brick_pct=BRICK_PCT):
    """Calculate dynamic brick size: brick_pct of window-period SMA of Close."""
    df = df.copy()
    df['brick_size'] = brick_pct * talib.SMA(df['Close'].values.astype(np.float64), timeperiod=window)
    return df


def calculate_displacement_and_run_length(df):
    """Calculate displacement and run length with reversal logic."""
    df = df.copy()
    df['price_change'] = df['Close'].diff()
    df['direction'] = np.where(df['price_change'] > 0, 1,
                               np.where(df['price_change'] < 0, -1, 0))
    
    reversal_level = np.nan
    prev_direction = 0
    run_lengths = []
    displacements = []
    
    for idx, row in df.iterrows():
        direction = row['direction']
        close = row['Close']
        brick_size = row['brick_size']
        
        if np.isnan(reversal_level):
            reversal_level = close
        
        if direction != 0 and direction != prev_direction and prev_direction != 0:
            reversal_level = close
        
        if not np.isnan(brick_size) and brick_size != 0:
            local_disp = (close - reversal_level) / brick_size
        else:
            local_disp = 0.0
        displacements.append(local_disp)
        
        if direction == 0:
            run_length = 0.0
        else:
            disp_sign = np.sign(local_disp)
            run_length = (np.floor(abs(local_disp)) + 1.0) * disp_sign
        
        run_lengths.append(run_length)
        
        if direction != 0:
            prev_direction = direction
    
    df['displacement'] = displacements
    df['run_length'] = run_lengths
    return df


def calculate_adx(df, period=PERIOD):
    """Calculate ADX using TA-Lib."""
    df = df.copy()
    df['ADX'] = talib.ADX(
        df['High'].values.astype(np.float64),
        df['Low'].values.astype(np.float64),
        df['Close'].values.astype(np.float64),
        timeperiod=period
    )
    return df


def calculate_atr(df, period=PERIOD):
    """Calculate ATR using TA-Lib."""
    df = df.copy()
    df['ATR'] = talib.ATR(
        df['High'].values.astype(np.float64),
        df['Low'].values.astype(np.float64),
        df['Close'].values.astype(np.float64),
        timeperiod=period
    )
    return df


def calculate_rsi(df, period=14):
    """Calculate RSI using TA-Lib."""
    df = df.copy()
    df['RSI'] = talib.RSI(df['Close'].values.astype(np.float64), timeperiod=period)
    return df



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



def calculate_metrics(trades_df):
    """Calculate performance metrics including yearly breakdown."""
    if trades_df.empty:
        return {
            'Total_Trades': 0,
            'Total_PnL': 0,
            'Max_Drawdown': 0,
            'Avg_Points': 0,
            'Win_Rate': 0,
            'PnL_DD_Ratio': 0
        }
    
    trades_df['Cumulative_PnL'] = trades_df['PnL'].cumsum()
    trades_df['Running_Max'] = trades_df['Cumulative_PnL'].cummax()
    trades_df['Drawdown'] = trades_df['Running_Max'] - trades_df['Cumulative_PnL']
    
    total_trades = len(trades_df)
    total_pnl = trades_df['PnL'].sum()
    max_drawdown = trades_df['Drawdown'].max()
    avg_points = total_pnl / total_trades if total_trades > 0 else 0
    winning_trades = (trades_df['PnL'] > 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    pnl_dd_ratio = (total_pnl / max_drawdown) if max_drawdown != 0 else 0
    
    trades_df['Year'] = trades_df['Exit_Datetime'].dt.year
    yearly_pnl = trades_df.groupby('Year')['PnL'].sum().to_dict()
    
    metrics = {
        'Total_Trades': total_trades,
        'Total_PnL': round(total_pnl, 2),
        'Max_Drawdown': round(max_drawdown, 2),
        'Avg_Points': round(avg_points, 2),
        'Win_Rate': round(win_rate, 2),
        'PnL_DD_Ratio': round(pnl_dd_ratio, 2)
    }
    
    for year, pnl in yearly_pnl.items():
        metrics[f'PnL_{year}'] = round(pnl, 2)
    
    return metrics


# Prepare data
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].astype(np.float64)

# Calculate indicators
print("Calculating indicators...")
df = calculate_brick_size(df_raw)
df = calculate_displacement_and_run_length(df)
df = calculate_adx(df)
df = calculate_atr(df)
df=  calculate_rsi(df)

# Run backtest
print("Running backtest...")
trades_df = run_backtest(df)

# Calculate metrics
print("Calculating metrics...")
metrics = calculate_metrics(trades_df)

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Parameters Used:")
print(f"  RUN_THRESH: {RUN_THRESH}")
print(f"  ADX_THRESH: {ADX_THRESH}")
print(f"  TRAIL_MULT: {TRAIL_MULT}")
print(f"  WINDOW: {WINDOW}")
print(f"  BRICK_PCT: {BRICK_PCT}")
print(f"  PERIOD: {PERIOD}")
print("\nMetrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
print("="*60)

# Save to Excel
print("\nSaving results to Excel...")
with pd.ExcelWriter('tradesheet_displacement22.xlsx', engine='openpyxl') as writer:
    # Save trades
    trades_df.to_excel(writer, sheet_name='Trades', index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    
    # Save parameters
    params_df = pd.DataFrame([{
        'RUN_THRESH': RUN_THRESH,
        'ADX_THRESH': ADX_THRESH,
        'TRAIL_MULT': TRAIL_MULT,
        'WINDOW': WINDOW,
        'BRICK_PCT': BRICK_PCT,
        'PERIOD': PERIOD
    }])
    params_df.to_excel(writer, sheet_name='Parameters', index=False)

print("Results saved to 'tradesheet_displacement.xlsx'")
print(f"Total trades: {len(trades_df)}")
print("Done!")