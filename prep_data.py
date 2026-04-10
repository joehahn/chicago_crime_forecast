#!/usr/bin/env python3
# prep_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-10

import pandas as pd
import numpy as np
from itertools import product

# ============================================================
# Load data/crimes.csv into df_filtered
# ============================================================
print('\n=== Loading data/crimes.csv into df_filtered ===')
df_filtered = pd.read_csv('data/crimes.csv')

# ============================================================
# Profile df_filtered
# ============================================================
print('\n=== Profile of df_filtered ===')
print(f'Shape: {df_filtered.shape}')
print(f'\nColumn types:\n{df_filtered.dtypes}')
print(f'\nFirst few rows:\n{df_filtered.head()}')
print(f'\nDescriptive stats:\n{df_filtered.describe(include="all")}')
print(f'\nMissing values:\n{df_filtered.isnull().sum()}')

# ============================================================
# Rename 'date' -> 'timestamp', derive month and year
# Keep top 20 primary_types -> df_20
# ============================================================
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['month'] = df_filtered['timestamp'].dt.month
df_filtered['year'] = df_filtered['timestamp'].dt.year

top_20 = df_filtered['primary_type'].value_counts().head(20).index.tolist()
df_20 = df_filtered[df_filtered['primary_type'].isin(top_20)].copy()
print(f'\n=== df_20: {len(df_20)} records ===')

# ============================================================
# Display counts of primary_type in df_20
# ============================================================
print('\n=== primary_type counts in df_20 ===')
print(df_20['primary_type'].value_counts().to_string())

# ============================================================
# Display 1 random record in df_20 (all columns)
# ============================================================
print('\n=== 1 random record in df_20 (all columns) ===')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_20.sample(1, random_state=42).to_string())

# ============================================================
# Column types in df_20
# ============================================================
print('\n=== Column types in df_20 ===')
print(df_20.dtypes)

# ============================================================
# Min and max date
# ============================================================
print('\n=== Min and Max dates ===')
print(f'Min date: {df_20["timestamp"].min()}')
print(f'Max date: {df_20["timestamp"].max()}')

# ============================================================
# Group df_20 by year, month, ward, primary_type -> df_avg
# ============================================================
df_avg = df_20.groupby(['year', 'month', 'ward', 'primary_type'], as_index=False).agg(
    date=('timestamp', 'min'),
    arrest=('arrest', 'mean'),
    domestic=('domestic', 'mean'),
    latitude=('latitude', 'mean'),
    longitude=('longitude', 'mean'),
    count_0=('id', 'count'),
)

# Cast ward as integer
df_avg['ward'] = df_avg['ward'].astype(int)

# Reorder: date first, count_0 last
cols_avg = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude', 'count_0']
df_avg = df_avg[cols_avg]

# Order by date, ward, primary_type
df_avg = df_avg.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'\n=== df_avg: {len(df_avg)} records ===')

# ============================================================
# Ward with greatest sum(count_0) for THEFT
# ============================================================
theft_ward_totals = df_avg[df_avg['primary_type'] == 'THEFT'].groupby('ward')['count_0'].sum()
best_theft_ward = theft_ward_totals.idxmax()
print(f'\n=== Ward with greatest sum(count_0) for primary_type=THEFT: {best_theft_ward} ===')
print(theft_ward_totals.sort_values(ascending=False).head(5))

# ============================================================
# All records in df_avg with primary_type=THEFT and ward=42
# ============================================================
print('\n=== df_avg: primary_type=THEFT, ward=42 ===')
pd.set_option('display.max_rows', None)
print(df_avg[(df_avg['primary_type'] == 'THEFT') & (df_avg['ward'] == 42)].to_string())

# ============================================================
# All records in df_avg with primary_type=ARSON and ward=42
# ============================================================
print('\n=== df_avg: primary_type=ARSON, ward=42 ===')
print(df_avg[(df_avg['primary_type'] == 'ARSON') & (df_avg['ward'] == 42)].to_string())

# ============================================================
# Zero-pad missing records -> df_pad
# ============================================================
time_periods = df_avg[['year', 'month']].drop_duplicates().sort_values(['year', 'month'])
time_tuples = list(zip(time_periods['year'], time_periods['month']))
all_wards = sorted(df_avg['ward'].unique())
all_types = sorted(df_avg['primary_type'].unique())

complete = pd.DataFrame(
    [(y, m, w, t) for (y, m) in time_tuples for w in all_wards for t in all_types],
    columns=['year', 'month', 'ward', 'primary_type']
)

df_pad = complete.merge(df_avg, on=['year', 'month', 'ward', 'primary_type'], how='left')

# Fill count_0=0 for missing records
df_pad['count_0'] = df_pad['count_0'].fillna(0).astype(int)

# Fill date for missing records using year/month
missing_date_mask = df_pad['date'].isna()
df_pad.loc[missing_date_mask, 'date'] = pd.to_datetime(
    df_pad.loc[missing_date_mask, 'year'].astype(str) + '-' +
    df_pad.loc[missing_date_mask, 'month'].astype(str).str.zfill(2) + '-01'
)

# Reorder columns and sort
df_pad = df_pad[cols_avg].sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'\n=== df_pad: {len(df_pad)} records ===')

# ============================================================
# All records in df_pad with primary_type=ARSON and ward=42
# ============================================================
print('\n=== df_pad: primary_type=ARSON, ward=42 ===')
print(df_pad[(df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)].to_string())

# ============================================================
# For each column replace NaN with random non-NaN values -> df_nan
# ============================================================
df_nan = df_pad.copy()
rng = np.random.default_rng(42)
for col in df_nan.columns:
    nan_mask = df_nan[col].isna()
    if nan_mask.any():
        non_nan_vals = df_nan.loc[~nan_mask, col].values
        n_missing = int(nan_mask.sum())
        df_nan.loc[nan_mask, col] = rng.choice(non_nan_vals, size=n_missing)

# ============================================================
# All records in df_nan with primary_type=ARSON and ward=22
# ============================================================
print('\n=== df_nan: primary_type=ARSON, ward=22 ===')
print(df_nan[(df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)].to_string())

# df_null is df_nan
df_null = df_nan

# ============================================================
# Print 5 random records in df_null
# ============================================================
print('\n=== 5 random records in df_null ===')
print(df_null.sample(5, random_state=42).to_string())

# ============================================================
# Partition by ward, primary_type; add shifted columns -> df_target
# ============================================================
def add_shifts(group):
    group = group.sort_values('date').copy()
    group['count_previous'] = group['count_0'].shift(1)
    group['count_1'] = group['count_0'].shift(-1)
    group['count_2'] = group['count_0'].shift(-2)
    group['count_3'] = group['count_0'].shift(-3)
    group['count_4'] = group['count_0'].shift(-4)
    return group

df_target = df_null.groupby(['ward', 'primary_type'], group_keys=False).apply(add_shifts)
df_target['delta_count'] = df_target['count_0'] - df_target['count_previous']

# Reorder: date first, delta_count + count_* last (keep count_previous for later drop)
base_cols = ['date', 'year', 'month', 'ward', 'primary_type', 'arrest', 'domestic', 'latitude', 'longitude']
end_cols  = ['delta_count', 'count_0', 'count_previous', 'count_1', 'count_2', 'count_3', 'count_4']
df_target = df_target[base_cols + end_cols]
df_target = df_target.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

# ============================================================
# Print all records in df_target with primary_type=THEFT and ward=27
# ============================================================
print('\n=== df_target: primary_type=THEFT, ward=27 ===')
print(df_target[(df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)].to_string())

# ============================================================
# Add ran_num and TTVFI -> df_tt
# ============================================================
rng2 = np.random.default_rng(99)
df_tt = df_target.copy()
df_tt['ran_num'] = rng2.uniform(0.0, 1.0, size=len(df_tt))
df_tt['TTVFI'] = np.where(df_tt['ran_num'] <= 0.667, 'train', 'test')

# ============================================================
# Set TTVFI='validate' for last 6 dates -> df_ttv
# ============================================================
df_ttv = df_tt.copy()
last_6_dates = sorted(df_ttv['date'].unique())[-6:]
df_ttv.loc[df_ttv['date'].isin(last_6_dates), 'TTVFI'] = 'validate'

# ============================================================
# Set TTVFI='forecast' for last 2 dates -> df_ttvf
# ============================================================
df_ttvf = df_ttv.copy()
last_2_dates = sorted(df_ttvf['date'].unique())[-2:]
df_ttvf.loc[df_ttvf['date'].isin(last_2_dates), 'TTVFI'] = 'forecast'

# ============================================================
# Set TTVFI='ignore' for the greatest date -> df_ttvfi
# ============================================================
df_ttvfi = df_ttvf.copy()
last_date = [sorted(df_ttvfi['date'].unique())[-1]]
df_ttvfi.loc[df_ttvfi['date'].isin(last_date), 'TTVFI'] = 'ignore'

# ============================================================
# Drop columns -> dt_monthly
# ============================================================
dt_monthly = df_ttvfi.drop(columns=['arrest', 'domestic', 'count_previous', 'ran_num'])

# Drop records with delta_count=NaN
dt_monthly = dt_monthly.dropna(subset=['delta_count']).reset_index(drop=True)

# ============================================================
# Prettyprint ALL records in df_ttvfi with primary_type=THEFT and ward=22
# ============================================================
print('\n=== df_ttvfi: primary_type=THEFT, ward=22 (all records) ===')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
subset = df_ttvfi[(df_ttvfi['primary_type'] == 'THEFT') & (df_ttvfi['ward'] == 22)]
print(subset.to_string(index=False))

# ============================================================
# Save dt_monthly as data/crimes_monthly.csv
# ============================================================
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print(f'\nSaved data/crimes_monthly.csv  ({len(dt_monthly)} records)')
