#!/usr/bin/env python3
# prep_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-8

import pandas as pd
import numpy as np

# Load data
print('Loading data/crimes.csv ...')
df_filtered = pd.read_csv('data/crimes.csv')
print(f'df_filtered shape: {df_filtered.shape}')

# Profile df_filtered
print('\n--- Profile df_filtered ---')
print(df_filtered.dtypes)
print()
print(df_filtered.describe(include='all'))

# Rename 'date' to 'timestamp', derive month and year
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['month'] = df_filtered['timestamp'].dt.month
df_filtered['year'] = df_filtered['timestamp'].dt.year

# Keep only top 20 primary_types
top20 = df_filtered['primary_type'].value_counts().nlargest(20).index
df_20 = df_filtered[df_filtered['primary_type'].isin(top20)].copy()
print(f'\nNumber of records in df_20: {len(df_20)}')

# Display counts of primary_type in df_20
print('\n--- primary_type counts in df_20 ---')
print(df_20['primary_type'].value_counts().to_string())

# Display 1 random record
print('\n--- 1 random record in df_20 (all columns) ---')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_20.sample(1).to_string())

# Column types in df_20
print('\n--- Column types in df_20 ---')
print(df_20.dtypes)

# Min/max date
print('\n--- Date range in df_20 ---')
print(f'Min date: {df_20["timestamp"].min()}')
print(f'Max date: {df_20["timestamp"].max()}')

# Group by year, month, ward, primary_type
df_avg = df_20.groupby(['year', 'month', 'ward', 'primary_type']).agg(
    mean_arrest=('arrest', 'mean'),
    mean_domestic=('domestic', 'mean'),
    mean_latitude=('latitude', 'mean'),
    mean_longitude=('longitude', 'mean'),
    count_0=('id', 'count'),
).reset_index()

# Drop 'mean_' prefix from column names
df_avg.columns = [c.replace('mean_', '') for c in df_avg.columns]

# Cast ward as integer
df_avg['ward'] = df_avg['ward'].astype(int)

# Add 'day' column = 1
df_avg['day'] = 1

# Reorder so count_0 is last
cols = [c for c in df_avg.columns if c != 'count_0'] + ['count_0']
df_avg = df_avg[cols]

# Order by year, month, ward, primary_type
df_avg = df_avg.sort_values(['year', 'month', 'ward', 'primary_type']).reset_index(drop=True)
print(f'\nNumber of records in df_avg: {len(df_avg)}')

# Set df_date = df_avg and add 'date' column derived from year, month, day
df_date = df_avg.copy()
df_date['date'] = pd.to_datetime(df_date[['year', 'month', 'day']])

# Which ward has greatest sum(count_0) for THEFT?
theft_by_ward = df_date[df_date['primary_type'] == 'THEFT'].groupby('ward')['count_0'].sum()
best_ward = theft_by_ward.idxmax()
print(f'\nWard with greatest sum(count_0) for THEFT: {best_ward}')

# Show all THEFT, ward=42 records in df_date
print('\n--- df_date records: primary_type=THEFT, ward=42 ---')
mask = (df_date['primary_type'] == 'THEFT') & (df_date['ward'] == 42)
print(df_date[mask].to_string())

# Show all ARSON, ward=42 records in df_date
print('\n--- df_date records: primary_type=ARSON, ward=42 ---')
mask = (df_date['primary_type'] == 'ARSON') & (df_date['ward'] == 42)
print(df_date[mask].to_string())

# Zero-pad missing records
# Build full grid of (date, ward, primary_type)
all_dates = sorted(df_date['date'].unique())
all_wards = sorted(df_date['ward'].unique())
all_types = sorted(df_date['primary_type'].unique())

full_index = pd.MultiIndex.from_product(
    [all_dates, all_wards, all_types],
    names=['date', 'ward', 'primary_type']
)
df_full = pd.DataFrame(index=full_index).reset_index()

# Left-merge to find missing records, zero-pad count_0
df_pad = df_full.merge(df_date, on=['date', 'ward', 'primary_type'], how='left')
df_pad['count_0'] = df_pad['count_0'].fillna(0).astype(int)

# Fill year, month, day from date
df_pad['year'] = df_pad['date'].dt.year
df_pad['month'] = df_pad['date'].dt.month
df_pad['day'] = df_pad['date'].dt.day

# Order by date, ward, primary_type
df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)
print(f'\nNumber of records in df_pad: {len(df_pad)}')

# Show all ARSON, ward=42 records in df_pad
print('\n--- df_pad records: primary_type=ARSON, ward=42 ---')
mask = (df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)
print(df_pad[mask].to_string())

# Replace NaN values with random selections of non-NaN values
df_nan = df_pad.copy()
rng = np.random.default_rng(seed=None)
for col in df_nan.columns:
    nan_mask = df_nan[col].isna()
    if nan_mask.any():
        non_nan_vals = df_nan.loc[~nan_mask, col].values
        df_nan.loc[nan_mask, col] = rng.choice(non_nan_vals, size=nan_mask.sum())

# Show ARSON, ward=22 records in df_nan
print('\n--- df_nan records: primary_type=ARSON, ward=22 ---')
mask = (df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)
print(df_nan[mask].to_string())

# df_null = df_nan (note: instructions use df_null to mean df_nan)
df_null = df_nan.copy()

# Print 5 random records in df_null
print('\n--- 5 random records in df_null ---')
print(df_null.sample(5).to_string())

# Partition df_null by ward, primary_type, order by date, compute shifted columns
def compute_shifts(group):
    group = group.sort_values('date')
    group['count_previous'] = group['count_0'].shift(1)
    group['count_1'] = group['count_0'].shift(-1)
    group['count_2'] = group['count_0'].shift(-2)
    group['count_3'] = group['count_0'].shift(-3)
    group['count_4'] = group['count_0'].shift(-4)
    return group

df_target = df_null.groupby(['ward', 'primary_type'], group_keys=False).apply(compute_shifts)
df_target['delta_count'] = df_target['count_0'] - df_target['count_previous']

# Reorder: date first, then delta_count, count_0, count_1, count_2, count_3, count_4 last
shift_cols = ['delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']
other_cols = [c for c in df_target.columns if c not in ['date'] + shift_cols]
df_target = df_target[['date'] + other_cols + shift_cols]
df_target = df_target.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

# Print THEFT, ward=27 records in df_target
print('\n--- df_target records: primary_type=THEFT, ward=27 ---')
mask = (df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)
print(df_target[mask].to_string())

# Add ran_num and TTV columns
df_ttv = df_target.copy()
df_ttv['ran_num'] = rng.uniform(0, 1, size=len(df_ttv))
df_ttv['TTV'] = np.where(df_ttv['ran_num'] <= 0.667, 'train', 'test')

# Set TTV='validate' for date >= 2025-01-01
df_ttv.loc[df_ttv['date'] >= '2025-01-01', 'TTV'] = 'validate'

# Set TTV='forecast' for the 2 greatest dates
last_few_dates = sorted(df_ttv['date'].unique())[-2:]
df_ttv.loc[df_ttv['date'].isin(last_few_dates), 'TTV'] = 'forecast'

# Set TTV='incomplete' for the greatest date
last_date = sorted(df_ttv['date'].unique())[-1]
df_ttv.loc[df_ttv['date'] == last_date, 'TTV'] = 'incomplete'

# Drop columns: arrest, domestic, count_previous, ran_num -> name result dt_monthly
dt_monthly = df_ttv.drop(columns=['arrest', 'domestic', 'count_previous', 'ran_num'])

# Drop records with delta_count=NaN
dt_monthly = dt_monthly.dropna(subset=['delta_count']).reset_index(drop=True)

# Prettyprint ALL records: primary_type=THEFT, ward=22
print('\n--- dt_monthly records: primary_type=THEFT, ward=22 ---')
pd.set_option('display.max_rows', None)
mask = (dt_monthly['primary_type'] == 'THEFT') & (dt_monthly['ward'] == 22)
print(dt_monthly[mask].to_string())

# Save to file
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print('\nSaved data/crimes_monthly.csv')
