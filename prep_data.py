#!/usr/bin/env python3
# prep_data.py
# by Joe Hahn
# jmh.datasciences@gmail.com
# 2026-April-11

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# ---- Load data ----
print('\n=== Loading data/crimes.csv into df_filtered ===')
df_filtered = pd.read_csv('data/crimes.csv')
print(f'df_filtered shape: {df_filtered.shape}')

# ---- Profile df_filtered ----
print('\n=== Profile of df_filtered ===')
print(df_filtered.info())
print(df_filtered.describe(include='all'))

# ---- Rename date -> timestamp, derive year and month ----
print('\n=== Rename date -> timestamp, derive month ===')
df_filtered = df_filtered.rename(columns={'date': 'timestamp'})
df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
df_filtered['year'] = df_filtered['timestamp'].dt.year
df_filtered['month'] = df_filtered['timestamp'].dt.month

# ---- Keep top 20 primary_types -> df_20 ----
top_20 = df_filtered['primary_type'].value_counts().nlargest(20).index.tolist()
df_20 = df_filtered[df_filtered['primary_type'].isin(top_20)].copy()
print(f'\nNumber of records in df_20: {len(df_20)}')

# ---- Counts of primary_type in df_20 ----
print('\n=== Counts of primary_type in df_20 ===')
print(df_20['primary_type'].value_counts())

# ---- 1 random record in df_20 ----
print('\n=== 1 random record in df_20 (all columns) ===')
print(df_20.sample(1, random_state=42).to_string())

# ---- Column types in df_20 ----
print('\n=== Column types in df_20 ===')
print(df_20.dtypes)

# ---- Min/Max date in df_20 ----
print('\n=== Min and Max date in df_20 ===')
print(f'Min date: {df_20["timestamp"].min()}')
print(f'Max date: {df_20["timestamp"].max()}')

# ---- Group by year, month, ward, primary_type -> df_avg ----
print('\n=== Creating df_avg ===')
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

# Add day = 1
df_avg['day'] = 1

# Reorder so count_0 is last
non_count_cols = [c for c in df_avg.columns if c != 'count_0']
df_avg = df_avg[non_count_cols + ['count_0']]

# Order by year, month, ward, primary_type
df_avg = df_avg.sort_values(['year', 'month', 'ward', 'primary_type']).reset_index(drop=True)

print(f'Number of records in df_avg: {len(df_avg)}')

# ---- Set df_date = df_avg, add date column ----
print('\n=== Adding date column to df_date ===')
df_date = df_avg.copy()
df_date['date'] = pd.to_datetime(df_date[['year', 'month', 'day']])

# ---- Ward with greatest sum(count_0) for THEFT ----
print('\n=== Ward with greatest sum(count_0) for THEFT ===')
theft_by_ward = df_date[df_date['primary_type'] == 'THEFT'].groupby('ward')['count_0'].sum()
best_ward = theft_by_ward.idxmax()
print(f'Ward {best_ward} has greatest THEFT count_0 sum = {theft_by_ward[best_ward]}')

# ---- THEFT records for ward=42 ----
print('\n=== THEFT records for ward=42 in df_date ===')
print(df_date[(df_date['primary_type'] == 'THEFT') & (df_date['ward'] == 42)].to_string())

# ---- ARSON records for ward=42 ----
print('\n=== ARSON records for ward=42 in df_date ===')
print(df_date[(df_date['primary_type'] == 'ARSON') & (df_date['ward'] == 42)].to_string())

# ---- Zero-pad missing records -> df_pad ----
print('\n=== Zero-padding missing records -> df_pad ===')
all_dates = df_date['date'].unique()
all_wards = sorted(df_date['ward'].unique())
all_types = sorted(df_date['primary_type'].unique())

full_idx = pd.MultiIndex.from_product(
    [all_dates, all_wards, all_types],
    names=['date', 'ward', 'primary_type']
)
full_df = pd.DataFrame(index=full_idx).reset_index()

df_pad = full_df.merge(df_date, on=['date', 'ward', 'primary_type'], how='left')

# Fill count_0 = 0 for missing records
df_pad['count_0'] = df_pad['count_0'].fillna(0)

# Fill year, month, day from date for missing records
df_pad['year'] = df_pad['date'].dt.year
df_pad['month'] = df_pad['date'].dt.month
df_pad['day'] = df_pad['date'].dt.day

# Order by date, ward, primary_type
df_pad = df_pad.sort_values(['date', 'ward', 'primary_type']).reset_index(drop=True)

print(f'Number of records in df_pad: {len(df_pad)}')

# ---- ARSON records for ward=42 in df_pad ----
print('\n=== ARSON records for ward=42 in df_pad ===')
print(df_pad[(df_pad['primary_type'] == 'ARSON') & (df_pad['ward'] == 42)].to_string())

# ---- Replace NaN with random non-NaN values -> df_nan ----
print('\n=== Replacing NaN values -> df_nan ===')
df_nan = df_pad.copy()
rng = np.random.default_rng(seed=42)
for col in df_nan.columns:
    mask = df_nan[col].isna()
    if mask.any():
        non_nan_vals = df_nan[col].dropna().values
        if len(non_nan_vals) > 0:
            df_nan.loc[mask, col] = rng.choice(non_nan_vals, size=mask.sum())

# ---- ARSON records for ward=22 in df_nan ----
print('\n=== ARSON records for ward=22 in df_nan ===')
print(df_nan[(df_nan['primary_type'] == 'ARSON') & (df_nan['ward'] == 22)].to_string())

# ---- df_null = df_nan, print 5 random records ----
df_null = df_nan.copy()
print('\n=== 5 random records in df_null ===')
print(df_null.sample(5, random_state=7).to_string())

# ---- Create df_target with shifted count columns ----
print('\n=== Creating df_target ===')
df_target = df_null.sort_values(['ward', 'primary_type', 'date']).copy()

grp = df_target.groupby(['ward', 'primary_type'])
df_target['count_previous'] = grp['count_0'].shift(1)
df_target['count_1'] = grp['count_0'].shift(-1)
df_target['count_2'] = grp['count_0'].shift(-2)
df_target['count_3'] = grp['count_0'].shift(-3)
df_target['count_4'] = grp['count_0'].shift(-4)
df_target['delta_count'] = df_target['count_0'] - df_target['count_previous']

# Reorder: date first, end cols last
end_cols = ['delta_count', 'count_0', 'count_1', 'count_2', 'count_3', 'count_4']
middle_cols = [c for c in df_target.columns if c not in ['date'] + end_cols]
df_target = df_target[['date'] + middle_cols + end_cols]

# ---- Print THEFT/ward=27 in df_target ----
print('\n=== THEFT records for ward=27 in df_target ===')
print(df_target[(df_target['primary_type'] == 'THEFT') & (df_target['ward'] == 27)].to_string())

# ---- Create TTVFI column -> df_ttvfi ----
print('\n=== Creating TTVFI column -> df_ttvfi ===')
df_ttvfi = df_target.copy()
df_ttvfi['ran_num'] = rng.uniform(0, 1, len(df_ttvfi))
df_ttvfi['TTVFI'] = 'train'
df_ttvfi.loc[df_ttvfi['ran_num'] > 0.667, 'TTVFI'] = 'test'

# Last 6 dates -> validate
sorted_dates = sorted(df_ttvfi['date'].unique())
last_6_dates = sorted_dates[-6:]
df_ttvfi.loc[df_ttvfi['date'].isin(last_6_dates), 'TTVFI'] = 'validate'

# Last 2 dates -> forecast
last_2_dates = sorted_dates[-2:]
df_ttvfi.loc[df_ttvfi['date'].isin(last_2_dates), 'TTVFI'] = 'forecast'

# Last 1 date -> incomeplete (preserving the typo from spec)
last_date = sorted_dates[-1:]
df_ttvfi.loc[df_ttvfi['date'].isin(last_date), 'TTVFI'] = 'incomeplete'

# ---- Drop columns, store as dt_monthly ----
drop_cols = ['arrest', 'domestic', 'count_previous', 'ran_num']
dt_monthly = df_ttvfi.drop(columns=drop_cols)

# ---- Drop records with delta_count = NaN ----
dt_monthly = dt_monthly.dropna(subset=['delta_count']).reset_index(drop=True)

# ---- Pretty print THEFT/ward=22 in dt_monthly ----
print('\n=== All THEFT records for ward=22 in dt_monthly ===')
subset = dt_monthly[(dt_monthly['primary_type'] == 'THEFT') & (dt_monthly['ward'] == 22)]
print(subset.to_string(index=False))

# ---- Save ----
dt_monthly.to_csv('data/crimes_monthly.csv', index=False)
print('\nSaved data/crimes_monthly.csv')
