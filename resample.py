# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 19:18:12 2026

@author: Admin
"""


import pandas as pd
df=pd.read_excel(r"1mindata.xlsx")
df.set_index("date",inplace=True)
df.index=pd.to_datetime(df.index)

min_spot=df.resample("75min",origin="start_day",offset="9h15min").agg({
    'open': "first",
    'high': 'max',
    'low': 'min',
    'close' : 'last'}).dropna()