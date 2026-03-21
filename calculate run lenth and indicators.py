# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:08:11 2026

@author: Admin
"""
import numpy as np
import talib

RUN_THRESH = 1
ADX_THRESH = 15.00
TRAIL_MULT = 0.95
WINDOW = 10
BRICK_PCT = 0.0015
PERIOD = 14

df_raw=resample_spot
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