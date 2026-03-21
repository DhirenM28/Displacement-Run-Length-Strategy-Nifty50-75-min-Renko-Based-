
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
